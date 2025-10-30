# HER Pipeline Explanation - Complete Code Flow

This document explains **where** and **how** Hindsight Experience Replay (HER) happens in this project, with exact code references.

---

## 🔍 Key Question: Where is `buffer.sample()` called?

**Answer:** In `modules/agent/core.py`, line 126, inside the `train()` method:

```python
def train(self) -> Dict:
    metric_list = []
    utd_ratio = self.cfg.utd_ratio * self.cfg.episode_batch_size
    transitions = self.buffer.sample(self.cfg.batch_size * utd_ratio)  # ← HER HAPPENS HERE!
    # ... rest of training code
```

This `train()` method is called from `train.py` line 115:

```python
policy_metrics = policy.train()  # ← Calls Agent.train()
```

---

## 📊 Complete Pipeline Flow

### **PHASE 1: Environment Interaction & Episode Collection**

**File:** `train.py` (main training loop)

```python
for epoch in range(cfg.n_epochs):
    for _ in range(cfg.n_cycles):
        # STEP 1: Collect episodes from environment
        train_episodes = rollout_worker.generate_rollout(train_mode=True)  # Line 97
```

**What happens:** `RolloutWorker.generate_rollout()` runs episodes in the environment.

---

**File:** `modules/rollout.py` (episode generation)

```python
def generate_rollout(self, train_mode: bool = False) -> Dict:
    ep_obs, ep_actions, ep_success, ep_rewards, ep_dones = [], [], [], [], []
    dict_obs, info = self.env.reset()  # Line 27
    
    obs = dict_obs["observation"]      # Robot state (x, y, theta)
    ag = dict_obs["achieved_goal"]     # Current position (x, y)
    g = dict_obs["desired_goal"]       # Target position (x, y)
    
    for _ in range(self.env_params["max_episode_steps"]):  # Line 32
        action = self.policy.act(obs.copy(), ag.copy(), g.copy(), train_mode)  # Line 33
        observation_new, reward, terminated, truncated, info = self.env.step(action)  # Line 37
        
        # Store all data from this step
        ep_obs.append(obs.copy())
        ep_actions.append(action.copy())
        ep_ag.append(ag.copy())  # ← Store achieved goals!
        ep_g.append(g.copy())    # ← Store desired goals!
        # ... more storage
        
        obs = obs_new
        ag = ag_new
    
    # Final observation and achieved_goal
    ep_obs.append(obs.copy())
    ep_ag.append(ag.copy())
    
    # Return episode data as numpy arrays
    return {
        'obs': np.stack(ep_obs, axis=1),      # Shape: (batch, T+1, obs_dim)
        'action': np.stack(ep_actions, axis=1), # Shape: (batch, T, action_dim)
        'ag': np.stack(ep_ag, axis=1),         # Shape: (batch, T+1, goal_dim)
        'g': np.stack(ep_g, axis=1),           # Shape: (batch, T, goal_dim)
        # ... more keys
    }
```

**Data Structure Example:**
```
Episode with T=3 steps:
ag = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]  # T+1 timesteps (includes final)
g  = [[gx, gy], [gx, gy], [gx, gy]]            # T timesteps (no final)
```

---

### **PHASE 2: Storing Episodes in Buffer**

**File:** `train.py`

```python
        # STEP 2: Store episodes in replay buffer
        policy.store(train_episodes)  # Line 107
```

**File:** `modules/agent/core.py`

```python
def store(self, episodes: Dict[str, np.ndarray]) -> None:
    self.buffer.store_episode(episode_batch=episodes)  # Line 99
```

**File:** `modules/buffer.py`

```python
class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.sample_func = sample_func  # ← This is HerSampler.sample_her_transitions!
        # ... initialize storage arrays
        
    def store_episode(self, episode_batch: Dict) -> None:
        batch_size = len(episode_batch["reward"])
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            for key, val in episode_batch.items():
                if key in self.buffer.keys():
                    self.buffer[key][idxs] = val  # Store raw episodes
```

**Important:** Episodes are stored **WITHOUT** any HER processing. Just raw data.

---

### **PHASE 3: Training - Sampling with HER**

**File:** `train.py`

```python
        # STEP 3: Train the policy
        policy_metrics = policy.train()  # Line 115
```

**File:** `modules/agent/core.py`

```python
def train(self) -> Dict:
    utd_ratio = self.cfg.utd_ratio * self.cfg.episode_batch_size
    
    # ═══════════════════════════════════════
    # ★★★ THIS IS WHERE HER HAPPENS! ★★★
    # ═══════════════════════════════════════
    transitions = self.buffer.sample(self.cfg.batch_size * utd_ratio)  # Line 126
    
    # Now we have HER-processed transitions, train on them
    for i in range(utd_ratio):
        mini_batch = jax.tree_util.tree_map(_slice, transitions)
        metric_list.append(self._update_networks(mini_batch))
```

---

### **PHASE 4: Buffer.sample() → HER Processing**

**File:** `modules/buffer.py`

```python
def sample(self, batch_size: int) -> Dict:
    with self.lock:
        temp_buffers = jax.tree_map(
            lambda arr: arr[: self.current_size], 
            self.buffer
        )
    
    # Add next_obs and next_ag for transition format
    temp_buffers["next_obs"] = temp_buffers["obs"][:, 1:, :]
    temp_buffers["next_ag"] = temp_buffers["ag"][:, 1:, :]
    
    # ═══════════════════════════════════════════════
    # ★★★ CALL THE HER SAMPLER! ★★★
    # ═══════════════════════════════════════════════
    transitions = self.sample_func(temp_buffers, batch_size)  # Line 48
    return transitions
```

**What is `self.sample_func`?**

It's set during buffer initialization in `modules/agent/sac.py`:

```python
class SAC(Agent):
    def __init__(self, ...):
        self.her_module = HerSampler(self.cfg, compute_rew)  # Line 75
        self.buffer = ReplayBuffer(
            env_params=self.env_params,
            buffer_size=self.cfg.buffer_size,
            sample_func=self.her_module.sample_her_transitions,  # ← This function!
        )
```

---

### **PHASE 5: HER Magic - Goal Relabeling**

**File:** `modules/hindsight.py`

```python
def sample_her_transitions(self, episode_batch: Dict, batch_size_in_transitions: int) -> Dict:
    """
    This is the HEART of HER!
    
    Input: Raw episode data from buffer
    Output: Transitions with relabeled goals
    """
    t = episode_batch["action"].shape[1]  # Max timesteps
    rollout_batch_size = episode_batch["action"].shape[0]  # Number of episodes
    batch_size = batch_size_in_transitions
    
    # ─────────────────────────────────────────────────
    # STEP 1: Randomly sample transitions
    # ─────────────────────────────────────────────────
    episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    t_samples = np.random.randint(t, size=batch_size)
    
    # Extract transitions: (s_t, a_t, s_{t+1}, g, ag_t, ag_{t+1})
    transitions = jax.tree_map(
        lambda x: x[episode_idxs, t_samples], 
        episode_batch
    )
    
    # ─────────────────────────────────────────────────
    # STEP 2: Select which transitions to relabel
    # ─────────────────────────────────────────────────
    # future_p is the probability of using HER (typically 0.8 if replay_k=4)
    her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
    
    # ─────────────────────────────────────────────────
    # STEP 3: Sample future achieved goals
    # ─────────────────────────────────────────────────
    # For each HER transition, pick a random future timestep
    future_offset = np.random.uniform(size=batch_size) * (t - t_samples)
    future_offset = future_offset.astype(int)
    future_t = (t_samples + 1 + future_offset)[her_indexes]
    
    # Get the achieved_goal from that future timestep
    future_ag = episode_batch["ag"][episode_idxs[her_indexes], future_t]
    
    # ─────────────────────────────────────────────────
    # STEP 4: Replace desired_goal with future achieved_goal
    # ─────────────────────────────────────────────────
    transitions["g"][her_indexes] = future_ag  # ← THE KEY OPERATION!
    
    # ─────────────────────────────────────────────────
    # STEP 5: Recalculate rewards with new goals
    # ─────────────────────────────────────────────────
    transitions["reward"] = np.expand_dims(
        np.array([
            self.reward_func(next_ag, g, None)  # compute_reward from env
            for next_ag, g in zip(transitions["next_ag"], transitions["g"])
        ]),
        1,
    )
    
    return transitions
```

---

## 🎯 Visual Example: HER in Action

### Original Episode (Failed to reach goal):

```
Goal: [18, 2]

t=0: obs=[1, 2, 0], ag=[1, 2], action=[0.5, 0]
  ↓
t=1: obs=[2, 2.5, 0], ag=[2, 2.5], action=[0.3, 0.1]
  ↓
t=2: obs=[3, 3, 0.1], ag=[3, 3], action=[0.2, 0]
  ↓
t=3: obs=[4, 3.5, 0.1], ag=[4, 3.5]

Reward at each step: -14, -13.5, -13, -12.5 (negative distances)
Episode failed! ❌
```

### HER Relabeling:

**Sample transition at t=1:**
- Original: `(obs=[2, 2.5, 0], action=[0.3, 0.1], next_obs=[3, 3, 0.1], goal=[18, 2], reward=-13.5)`
- HER selects future time t=3
- Future achieved_goal = [4, 3.5]
- **Replace goal:** `goal = [4, 3.5]` (pretend we wanted to reach [4, 3.5])
- **Recalculate reward:** `distance([3, 3], [4, 3.5]) = 1.12` → reward = -1.12
- New transition: `(obs=[2, 2.5, 0], action=[0.3, 0.1], next_obs=[3, 3, 0.1], goal=[4, 3.5], reward=-1.12)`

**Result:** The agent learns that this action was "good" for reaching [4, 3.5]! ✅

---

## 📋 Class Responsibilities Summary

### 1. **RolloutWorker** (`modules/rollout.py`)
- **Purpose:** Interact with environment and collect episodes
- **Inputs:** Environment, policy
- **Outputs:** Episode dictionaries with `obs`, `action`, `ag`, `g`, `reward`
- **Key Method:** `generate_rollout()`

### 2. **ReplayBuffer** (`modules/buffer.py`)
- **Purpose:** Store episodes and sample transitions
- **Stores:** Raw episode data (no processing)
- **Key Methods:**
  - `store_episode()`: Add episodes to buffer
  - `sample()`: Sample transitions (delegates to HER)
- **Important:** `sample_func` is injected during initialization!

### 3. **HerSampler** (`modules/hindsight.py`)
- **Purpose:** Apply HER goal relabeling
- **Key Method:** `sample_her_transitions()`
- **Algorithm:**
  1. Randomly sample transitions from episodes
  2. For some transitions (controlled by `replay_k`), replace goal with future achieved_goal
  3. Recalculate rewards based on new goals
  4. Return modified transitions

### 4. **Agent (SAC/DDPG)** (`modules/agent/`)
- **Purpose:** Policy learning
- **Key Methods:**
  - `store()`: Pass episodes to buffer
  - `train()`: Sample from buffer (triggers HER) and update networks
  - `_update_networks()`: Actual gradient updates
- **Connection to HER:** Creates `HerSampler` and passes it to `ReplayBuffer`

---

## 🔗 Dependency Injection Pattern

The project uses **dependency injection** to connect HER:

```python
# In SAC.__init__() (modules/agent/sac.py)

# Create HER sampler with reward function
self.her_module = HerSampler(cfg, compute_rew)

# Inject HER's sampling function into buffer
self.buffer = ReplayBuffer(
    env_params=env_params,
    buffer_size=cfg.buffer_size,
    sample_func=self.her_module.sample_her_transitions,  # ← Injection!
)
```

This means:
- `ReplayBuffer` doesn't know about HER specifically
- It just calls whatever `sample_func` it was given
- This makes the code modular - you could swap HER for other sampling strategies

---

## ⚙️ Configuration System

### **What is `cfg`?**

`cfg` is a **Hydra configuration object** (type: `DictConfig` from `omegaconf` library) that contains all hyperparameters for training.

**Definition:** In `train.py`, line 257:

```python
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    check_hydra_config(cfg, comm)
    launch(cfg, comm)  # cfg is passed to launch()
```

The `@hydra.main` decorator:
1. Loads `conf/config.yaml` as the base configuration
2. Merges all files specified in the `defaults:` section
3. Creates a `DictConfig` object
4. Passes it to `main(cfg)`

### **Configuration Files Structure**

```
conf/
├── config.yaml           # Main config (loaded first)
├── misc.yaml            # Hardware & logging settings
├── agent/
│   ├── sac.yaml        # SAC algorithm parameters
│   └── ddpg.yaml       # DDPG algorithm parameters
└── hindsight/
    └── her.yaml        # HER parameters
```

### **Main Configuration (`conf/config.yaml`)**

```yaml
# Training Schedule
n_epochs: 10              # Number of training epochs
n_cycles: 50              # Episodes per epoch (per worker)
utd_ratio: 10             # Update-to-data ratio
batch_size: 256           # Training batch size
n_test_rollouts: 20       # Evaluation episodes

# Environment
env_name: 'FetchPush-v2'
seed: null                # Random seed (null = random)

# Replay Buffer
buffer_size: 1_000_000    # Max transitions in buffer
gamma: 0.98               # Discount factor

# Normalization
clip_range: 5
normalize_goal: True
done_signal: False        # Use done signal in TD target
```

### **HER Configuration (`conf/hindsight/her.yaml`)**

```yaml
name: her
replay_strategy: future  # Use future achieved goals
replay_k: 4              # Ratio of HER to original transitions
```

**What `replay_k=4` means:**
- For every 1 original transition, add 4 HER transitions
- Total: 80% of training uses HER goals, 20% uses original goals
- Calculated as: `future_p = 1 - (1 / (1 + 4)) = 0.8`

### **Misc Configuration (`conf/misc.yaml`)**

```yaml
episode_batch_size: 4     # Number of parallel environments
save_dir: 'parker/'       # Where to save models/logs
use_wandb: False          # Enable Weights & Biases logging
logging_formats:
  - stdout
  - log
  - csv
```

### **Agent Configuration (`conf/agent/sac.yaml`)**

```yaml
name: sac
automatic_entropy_tuning: True

actor:
  lr: 0.001               # Learning rate
  hidden_size: [256, 256] # Network architecture
  activation: relu

critic:
  lr: 0.001
  ensemble_size: 2        # Number of Q-networks
  dropout: 0.01
  tau: 0.005              # Soft update coefficient
  hidden_size: [256, 256]

temperature:
  lr: 0.001
  alpha: 1.0              # Initial entropy coefficient
```

### **Accessing Configuration Values**

In code, you access config values using dot notation:

```python
# Training parameters
epochs = cfg.n_epochs           # → 10
cycles = cfg.n_cycles           # → 50
batch = cfg.batch_size          # → 256

# Agent parameters
agent_name = cfg.agent.name     # → "sac"
actor_lr = cfg.agent.actor.lr  # → 0.001

# HER parameters
her_k = cfg.hindsight.replay_k  # → 4
strategy = cfg.hindsight.replay_strategy  # → "future"

# Environment
env = cfg.env_name              # → "FetchPush-v2"
parallel = cfg.episode_batch_size  # → 4
```

### **Command-line Overrides**

You can override any config value from the command line:

```bash
# Override single values
python train.py n_epochs=100
python train.py batch_size=512

# Override nested values
python train.py agent.actor.lr=0.0001

# Override multiple values
python train.py n_epochs=50 batch_size=512 seed=42

# Switch agent type
python train.py agent=ddpg

# Combine overrides
python train.py agent=ddpg n_epochs=100 hindsight.replay_k=8
```

### **Training Math (Using Default Config)**

With the default configuration:

```
Total environment steps per epoch:
  = n_cycles × episode_batch_size × max_episode_steps
  = 50 × 4 × 50
  = 10,000 steps per epoch

Total training for 10 epochs:
  = 10 × 10,000
  = 100,000 environment steps

Gradient updates per epoch:
  = n_cycles × utd_ratio × episode_batch_size
  = 50 × 10 × 4
  = 2,000 gradient updates per epoch

Total gradient updates:
  = 10 × 2,000
  = 20,000 gradient updates
```

---

## 🎓 Key Insights

1. **HER is applied during sampling, not storage**
   - Episodes are stored as-is
   - HER relabeling happens when `buffer.sample()` is called
   - This allows flexible replay ratios

2. **The reward function must be recomputable**
   - `compute_reward(achieved_goal, desired_goal, info)` is required
   - It must work with any goal, not just the original
   - This is why your environment needs this method!

3. **Two types of transitions are mixed**
   - Original transitions (20%): Learn to reach actual goals
   - HER transitions (80%): Learn general goal-reaching skills

4. **Training efficiency**
   - Every failed episode generates useful training data
   - Agent learns from "what it actually did" not "what it failed to do"
   - Dramatically speeds up learning for sparse reward tasks

---

## 🚀 Adapting Your Environment Checklist

For your robot obstacle avoidance environment:

- [ ] Return dict observations: `{"observation": ..., "achieved_goal": ..., "desired_goal": ...}`
- [ ] Implement `compute_reward(achieved_goal, desired_goal, info)` method
- [ ] Make sure `achieved_goal` is the (x, y) positions of robots
- [ ] Make sure `desired_goal` is the target (x, y) positions
- [ ] Test that reward can be computed from goals alone (no access to obstacles)

The last point is important - HER will change goals, so rewards must be calculable without knowing the full state!

