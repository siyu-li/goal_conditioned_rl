"""
PolicyManager: Manages loading and selecting RL policies for different robot subsets.

Usage:
    manager = PolicyManager(policy_dir="policies/")
    manager.load_all_policies()
    
    # Get policy for controlling robots [0, 2]
    policy = manager.get_policy([0, 2])
    action = policy.get_action(observation)
"""
import os
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState

from modules.mpi_utils.normalizer import Normalizer
from modules.networks import GaussianActor


@dataclass
class PolicyConfig:
    """Configuration for a loaded policy."""
    controlled_indices: Tuple[int, ...]
    obs_dim: int
    goal_dim: int
    action_dim: int
    model_path: str


class LoadedPolicy:
    """
    A loaded RL policy that can be used for inference.
    Wraps actor network with normalizers for direct action computation.
    """
    
    def __init__(
        self,
        actor_state: TrainState,
        actor_network: nn.Module,
        o_norm_stats: Dict,
        g_norm_stats: Dict,
        config: PolicyConfig,
        clip_range: float = 5.0
    ):
        self.actor_state = actor_state
        self.actor_network = actor_network
        self.config = config
        self.clip_range = clip_range
        
        # Setup normalizers
        self.o_norm = Normalizer(
            size=config.obs_dim,
            mean=o_norm_stats.get('mean'),
            std=o_norm_stats.get('std'),
            default_clip_range=clip_range
        )
        self.g_norm = Normalizer(
            size=config.goal_dim,
            mean=g_norm_stats.get('mean'),
            std=g_norm_stats.get('std'),
            default_clip_range=clip_range
        )
        
        # JIT compile action selection
        self._get_action_jit = jax.jit(self._get_action_impl)
    
    def _get_action_impl(self, params, obs: jax.Array, goal: jax.Array, deterministic: bool = True):
        """Internal JIT-compiled action selection."""
        actor_input = jnp.concatenate([obs, goal], axis=-1)
        mean, log_std = self.actor_network.apply(params, actor_input)
        
        if deterministic:
            # Use mean action
            action = jnp.tanh(mean)
        else:
            # Sample from distribution
            std = jnp.exp(log_std)
            noise = jax.random.normal(jax.random.PRNGKey(0), mean.shape)
            action = jnp.tanh(mean + std * noise)
        
        return action
    
    def get_action(
        self,
        observation: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action from the policy.
        
        Args:
            observation: Current observation (unnormalized)
            achieved_goal: Current achieved goal (unnormalized)
            desired_goal: Desired goal (unnormalized)
            deterministic: If True, use mean action; else sample
            
        Returns:
            action: Action array in [-1, 1]
        """
        # Normalize inputs
        obs_norm = self.o_norm.normalize(observation)
        goal_norm = self.g_norm.normalize(desired_goal)
        
        # Convert to JAX arrays
        obs_jax = jnp.array(obs_norm, dtype=jnp.float32)
        goal_jax = jnp.array(goal_norm, dtype=jnp.float32)
        
        # Add batch dimension if needed
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None, :]
            goal_jax = goal_jax[None, :]
        
        # Get action
        action = self._get_action_jit(
            self.actor_state.params,
            obs_jax,
            goal_jax,
            deterministic
        )
        
        # Remove batch dimension and convert to numpy
        action = np.array(action).squeeze(0)
        return action
    
    @property
    def controlled_robots(self) -> Tuple[int, ...]:
        """Return the robot indices this policy controls."""
        return self.config.controlled_indices


class PolicyManager:
    """
    Manages multiple RL policies for different robot subsets.
    
    Directory structure expected:
        policy_dir/
            1r_0/model_final.pkl       # 1 robot, index 0
            1r_1/model_final.pkl       # 1 robot, index 1
            2r_0_1/model_final.pkl     # 2 robots, indices 0,1
            2r_0_2/model_final.pkl     # 2 robots, indices 0,2
            ...
    """
    
    def __init__(
        self,
        policy_dir: str,
        cfg: Optional[Dict] = None,
        env_params: Optional[Dict] = None
    ):
        """
        Initialize PolicyManager.
        
        Args:
            policy_dir: Directory containing policy subdirectories
            cfg: Optional config dict (for network architecture)
            env_params: Optional env params dict (for network dimensions)
        """
        self.policy_dir = policy_dir
        self.cfg = cfg or self._default_cfg()
        self.env_params = env_params
        self.policies: Dict[Tuple[int, ...], LoadedPolicy] = {}
        self._policy_configs: Dict[Tuple[int, ...], PolicyConfig] = {}
    
    @staticmethod
    def _default_cfg() -> Dict:
        """Default configuration for networks."""
        return {
            'agent': {
                'actor': {
                    'hidden_layers': [256, 256, 256],
                    'log_std_min': -20,
                    'log_std_max': 2,
                }
            },
            'layer_norm': True,
        }
    
    @staticmethod
    def indices_to_key(indices: List[int]) -> Tuple[int, ...]:
        """Convert list of indices to hashable tuple key."""
        return tuple(sorted(indices))
    
    @staticmethod
    def key_to_dirname(key: Tuple[int, ...]) -> str:
        """Convert key to directory name format."""
        n = len(key)
        indices_str = "_".join(map(str, key))
        return f"{n}r_{indices_str}"
    
    @staticmethod
    def dirname_to_key(dirname: str) -> Optional[Tuple[int, ...]]:
        """Parse directory name to key. Returns None if invalid format."""
        try:
            # Expected format: "Nr_i1_i2_..._iN" e.g., "2r_0_1"
            parts = dirname.split('_')
            if not parts[0].endswith('r'):
                return None
            n = int(parts[0][:-1])
            indices = tuple(int(p) for p in parts[1:])
            if len(indices) != n:
                return None
            return indices
        except (ValueError, IndexError):
            return None
    
    def discover_policies(self) -> List[Tuple[int, ...]]:
        """
        Scan policy directory and return available policy keys.
        
        Returns:
            List of tuples representing available robot index combinations
        """
        available = []
        if not os.path.isdir(self.policy_dir):
            return available
        
        for dirname in os.listdir(self.policy_dir):
            dirpath = os.path.join(self.policy_dir, dirname)
            if not os.path.isdir(dirpath):
                continue
            
            key = self.dirname_to_key(dirname)
            if key is None:
                continue
            
            # Check if model file exists
            model_path = os.path.join(dirpath, "model_final.pkl")
            if os.path.isfile(model_path):
                available.append(key)
        
        return sorted(available, key=lambda x: (len(x), x))
    
    def load_policy(
        self,
        controlled_indices: List[int],
        model_name: str = "model_final.pkl"
    ) -> LoadedPolicy:
        """
        Load a single policy for the given robot indices.
        
        Args:
            controlled_indices: List of robot indices this policy controls
            model_name: Name of the model file
            
        Returns:
            LoadedPolicy instance
        """
        key = self.indices_to_key(controlled_indices)
        
        # Return cached if available
        if key in self.policies:
            return self.policies[key]
        
        # Find policy directory
        dirname = self.key_to_dirname(key)
        policy_dir = os.path.join(self.policy_dir, dirname)
        model_path = os.path.join(policy_dir, model_name)
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Policy not found for robots {controlled_indices} at {model_path}"
            )
        
        # Load checkpoint
        with open(model_path, 'rb') as f:
            ckpt = pickle.load(f)
        
        # Determine dimensions from normalizer stats
        o_norm_stats = ckpt['normalizer'].get('o_norm', {})
        g_norm_stats = ckpt['normalizer'].get('g_norm', {})
        
        obs_dim = len(o_norm_stats.get('mean', np.zeros(1)))
        goal_dim = len(g_norm_stats.get('mean', np.zeros(1)))
        
        # Action dimension: 3 per controlled robot (dx, dy, dtheta)
        action_dim = len(key) * 3
        
        config = PolicyConfig(
            controlled_indices=key,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            model_path=model_path
        )
        
        # Create actor network
        env_params = self.env_params or {
            'obs': obs_dim,
            'goal': goal_dim,
            'action': (action_dim,)
        }
        
        actor_network = GaussianActor(action_dim, self.cfg, env_params)
        
        # Create dummy state and restore
        dummy_obs = jnp.zeros((1, obs_dim + goal_dim))
        actor_key = jax.random.PRNGKey(0)
        
        actor_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network.init(actor_key, dummy_obs),
            tx=flax.optax.adam(1e-3)  # Dummy optimizer, not used for inference
        )
        
        # Restore from checkpoint
        actor_state = flax.serialization.from_bytes(actor_state, ckpt['actor'])
        
        # Create LoadedPolicy
        policy = LoadedPolicy(
            actor_state=actor_state,
            actor_network=actor_network,
            o_norm_stats=o_norm_stats,
            g_norm_stats=g_norm_stats,
            config=config
        )
        
        # Cache
        self.policies[key] = policy
        self._policy_configs[key] = config
        
        return policy
    
    def load_all_policies(self) -> int:
        """
        Load all available policies from the policy directory.
        
        Returns:
            Number of policies loaded
        """
        available = self.discover_policies()
        for key in available:
            try:
                self.load_policy(list(key))
            except Exception as e:
                print(f"Warning: Failed to load policy for {key}: {e}")
        return len(self.policies)
    
    def get_policy(self, controlled_indices: List[int]) -> LoadedPolicy:
        """
        Get a policy for the given robot indices.
        
        Args:
            controlled_indices: List of robot indices
            
        Returns:
            LoadedPolicy instance
            
        Raises:
            KeyError if policy not loaded
        """
        key = self.indices_to_key(controlled_indices)
        if key not in self.policies:
            # Try to load on demand
            return self.load_policy(controlled_indices)
        return self.policies[key]
    
    def has_policy(self, controlled_indices: List[int]) -> bool:
        """Check if a policy exists for the given indices."""
        key = self.indices_to_key(controlled_indices)
        if key in self.policies:
            return True
        # Check if file exists
        dirname = self.key_to_dirname(key)
        model_path = os.path.join(self.policy_dir, dirname, "model_final.pkl")
        return os.path.isfile(model_path)
    
    def list_loaded_policies(self) -> List[Tuple[int, ...]]:
        """Return list of loaded policy keys."""
        return list(self.policies.keys())
    
    def get_policy_info(self, controlled_indices: List[int]) -> Optional[PolicyConfig]:
        """Get configuration info for a policy."""
        key = self.indices_to_key(controlled_indices)
        return self._policy_configs.get(key)


# Convenience function for quick policy loading
def load_policy_for_inference(
    model_path: str,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    cfg: Optional[Dict] = None
) -> LoadedPolicy:
    """
    Quick helper to load a single policy file for inference.
    
    Args:
        model_path: Path to model_*.pkl file
        obs_dim: Observation dimension
        goal_dim: Goal dimension
        action_dim: Action dimension
        cfg: Optional network config
        
    Returns:
        LoadedPolicy ready for inference
    """
    cfg = cfg or PolicyManager._default_cfg()
    
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    env_params = {
        'obs': obs_dim,
        'goal': goal_dim,
        'action': (action_dim,)
    }
    
    actor_network = GaussianActor(action_dim, cfg, env_params)
    
    dummy_obs = jnp.zeros((1, obs_dim + goal_dim))
    actor_key = jax.random.PRNGKey(0)
    
    actor_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network.init(actor_key, dummy_obs),
        tx=flax.optax.adam(1e-3)
    )
    actor_state = flax.serialization.from_bytes(actor_state, ckpt['actor'])
    
    o_norm_stats = ckpt['normalizer'].get('o_norm', {})
    g_norm_stats = ckpt['normalizer'].get('g_norm', {})
    
    config = PolicyConfig(
        controlled_indices=(),
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        model_path=model_path
    )
    
    return LoadedPolicy(
        actor_state=actor_state,
        actor_network=actor_network,
        o_norm_stats=o_norm_stats,
        g_norm_stats=g_norm_stats,
        config=config
    )
