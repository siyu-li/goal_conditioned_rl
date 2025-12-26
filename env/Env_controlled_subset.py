"""
Child environment controlling a subset of robots.
Remaining robots are stationary obstacles.
Compatible with HER and goal-conditioned RL.

This is a STANDALONE environment for training RL policies.
It manages all 6 robots internally but only controls a subset.
Uncontrolled robots act as stationary obstacles.

Usage for Training:
    env = ControlledSubsetEnv(
        controlled_robot_indices=[0, 2],  # Control robots 0 and 2
        obstacle_config_path='configs/obstacles_default.yaml',
        max_episode_steps=50
    )
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import yaml


class ControlledSubsetEnv(gym.Env):
    """
    Environment that controls a subset of 6 robots.
    Uncontrolled robots are stationary obstacles.
    
    Observation per controlled robot:
    - robot_x, robot_y (absolute position)
    - cos(robot_theta), sin(robot_theta)
    - For each other controlled robot: distance, cos(bearing), sin(bearing)
    - For each uncontrolled robot: distance, cos(bearing), sin(bearing)
    - For each static obstacle: distance, cos(bearing), sin(bearing)
    
    Plus absolute goal positions at the end.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 controlled_robot_indices,
                 obstacle_config_path=None,
                 max_episode_steps=50,
                 action_coupled=False,
                 robot_radius=0.5,
                 render_mode=None):
        """
        Args:
            controlled_robot_indices: List of robot indices to control (e.g., [0, 2])
            obstacle_config_path: Path to YAML file with obstacle configuration
            max_episode_steps: Episode length
            action_coupled: If True, use shared distance for all controlled robots
            robot_radius: Radius of each robot
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()
        
        # Robot configuration
        self.num_total_robots = 6  # Always 6 robots total
        self.controlled_indices = sorted(controlled_robot_indices)
        self.num_controlled = len(self.controlled_indices)
        self.uncontrolled_indices = [i for i in range(6) if i not in self.controlled_indices]
        self.num_uncontrolled = len(self.uncontrolled_indices)
        
        self.robot_radius = float(robot_radius)
        self.max_step_size = 0.5
        
        # Episode configuration
        self.max_episode_steps = max_episode_steps
        self.action_coupled = action_coupled
        self.render_mode = render_mode

        # Workspace boundaries
        self.x_min, self.x_max = 0.0, 18.0
        self.y_min, self.y_max = 0.0, 18.0

        # Load obstacles from config
        self.obstacles = []
        if obstacle_config_path:
            self._load_obstacle_map(obstacle_config_path)

        # State: [x, y, theta] for ALL 6 robots (controlled + uncontrolled)
        self.state = np.zeros((6, 3), dtype=np.float32)
        
        # Goals only for controlled robots
        self.goal_position = np.zeros((self.num_controlled, 2), dtype=np.float32)

        # Action space: only for controlled robots
        if self.action_coupled:
            # [shared_distance, dtheta_0, dtheta_1, ...]
            action_low = np.array([0.0] + [-np.pi] * self.num_controlled, dtype=np.float32)
            action_high = np.array([self.max_step_size] + [np.pi] * self.num_controlled, dtype=np.float32)
        else:
            # [d_0, dtheta_0, d_1, dtheta_1, ...]
            action_low = np.tile([0.0, -np.pi], self.num_controlled).astype(np.float32)
            action_high = np.tile([self.max_step_size, np.pi], self.num_controlled).astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation space
        obs_low, obs_high = self._build_observation_bounds()
        
        # Goals only for controlled robots
        goal_low = np.tile([self.x_min, self.y_min], self.num_controlled).astype(np.float32)
        goal_high = np.tile([self.x_max, self.y_max], self.num_controlled).astype(np.float32)
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

        # Reward parameters (same as Env_v6)
        self.goal_threshold = 1.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.collision_penalty = -100.0
        self.proximity_threshold = 3.0
        self.max_proximity_penalty = -10.0

        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        if self.render_mode == "human":
            plt.ion()

    def _load_obstacle_map(self, config_path):
        """Load obstacles from YAML configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.obstacles = []
            if 'obstacles' in config:
                for obs in config['obstacles']:
                    if obs['type'] == 'circle':
                        self.obstacles.append({
                            'type': 'circle',
                            'x': obs['x'],
                            'y': obs['y'],
                            'radius': obs['radius']
                        })
            
            # Update workspace if specified
            if 'workspace' in config:
                ws = config['workspace']
                self.x_min = ws.get('x_min', self.x_min)
                self.x_max = ws.get('x_max', self.x_max)
                self.y_min = ws.get('y_min', self.y_min)
                self.y_max = ws.get('y_max', self.y_max)
                
        except Exception as e:
            print(f"Warning: Could not load obstacle config from {config_path}: {e}")
            self.obstacles = []

    def _build_observation_bounds(self):
        """
        Build observation space bounds.
        
        Per controlled robot:
        - x, y (absolute position)
        - cos(theta), sin(theta)
        - For each other controlled robot: distance, cos(bearing), sin(bearing)
        - For each uncontrolled robot: distance, cos(bearing), sin(bearing)
        - For each static obstacle: distance, cos(bearing), sin(bearing)
        
        Note: Goals are NOT included in observation array (they're in achieved_goal/desired_goal).
        """
        obs_low = []
        obs_high = []
        max_distance = np.sqrt(2) * max(self.x_max, self.y_max)

        # Per controlled robot observations
        for _ in range(self.num_controlled):
            # Absolute position (x, y)
            obs_low.extend([self.x_min, self.y_min])
            obs_high.extend([self.x_max, self.y_max])
            
            # Robot orientation as cos(theta) and sin(theta)
            obs_low.extend([-1.0, -1.0])
            obs_high.extend([1.0, 1.0])
            
            # For each other controlled robot: distance, cos(bearing), sin(bearing)
            for _ in range(self.num_controlled - 1):
                obs_low.extend([0.0, -1.0, -1.0])
                obs_high.extend([max_distance, 1.0, 1.0])
            
            # For each uncontrolled robot: distance, cos(bearing), sin(bearing)
            for _ in range(self.num_uncontrolled):
                obs_low.extend([0.0, -1.0, -1.0])
                obs_high.extend([max_distance, 1.0, 1.0])
            
            # For each static obstacle: distance, cos(bearing), sin(bearing)
            num_obstacles = len(self.obstacles)
            for _ in range(num_obstacles):
                obs_low.extend([0.0, -1.0, -1.0])
                obs_high.extend([max_distance, 1.0, 1.0])

        return np.array(obs_low, dtype=np.float32), np.array(obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Sample random collision-free positions for all 6 robots
        self.state = self._sample_random_state()
        
        # Sample goals for controlled robots
        self._sample_goals()
        
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        
        obs = self._get_obs()
        info = {"is_success": False, "collision": False}
        return obs, info

    def _sample_random_position(self):
        """Sample a random (x, y) position within workspace boundaries."""
        x = np.random.uniform(self.x_min + self.robot_radius, self.x_max - self.robot_radius)
        y = np.random.uniform(self.y_min + self.robot_radius, self.y_max - self.robot_radius)
        return np.array([x, y], dtype=np.float32)

    def _is_position_valid(self, pos, existing_positions):
        """Check if position is valid (no collision with obstacles or existing positions)."""
        x, y = pos
        
        # Check boundaries
        if not (self.x_min + self.robot_radius <= x <= self.x_max - self.robot_radius and
                self.y_min + self.robot_radius <= y <= self.y_max - self.robot_radius):
            return False
        
        # Check obstacle collision
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                dist = np.linalg.norm(pos - np.array([obs['x'], obs['y']]))
                if dist < self.robot_radius + obs['radius']:
                    return False
        
        # Check collision with existing positions
        for existing_pos in existing_positions:
            if np.linalg.norm(pos - existing_pos) < 2 * self.robot_radius:
                return False
        
        return True

    def _sample_random_state(self, max_attempts=1000):
        """Sample random collision-free initial state for all 6 robots."""
        state = np.zeros((6, 3), dtype=np.float32)
        positions = []
        
        for i in range(6):
            valid_position_found = False
            for attempt in range(max_attempts):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, positions):
                    positions.append(pos)
                    state[i, 0] = pos[0]
                    state[i, 1] = pos[1]
                    state[i, 2] = np.random.uniform(-np.pi, np.pi)
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                raise RuntimeError(
                    f"Failed to find valid initial position for robot {i} after {max_attempts} attempts"
                )
        
        return state

    def _sample_goals(self, max_attempts=1000):
        """Sample collision-free goals for controlled robots."""
        goals = []
        existing = list(self.state[:, :2])  # Avoid robot positions
        
        for i in range(self.num_controlled):
            for _ in range(max_attempts):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, existing + goals):
                    goals.append(pos)
                    break
            else:
                goals.append(self._sample_random_position())
        
        self.goal_position = np.array(goals, dtype=np.float32)

    def step(self, action):
        """Execute one step."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        proposed_state = self.state.copy()
        
        # Apply action to controlled robots
        if self.action_coupled:
            shared_distance = float(action[0])
            for idx, robot_i in enumerate(self.controlled_indices):
                dtheta = float(action[1 + idx])
                theta = self.state[robot_i, 2]
                
                dx = shared_distance * np.cos(theta)
                dy = shared_distance * np.sin(theta)
                
                proposed_state[robot_i, 0] = np.clip(
                    self.state[robot_i, 0] + dx, self.x_min, self.x_max)
                proposed_state[robot_i, 1] = np.clip(
                    self.state[robot_i, 1] + dy, self.y_min, self.y_max)
                proposed_state[robot_i, 2] = self._wrap_angle(theta + dtheta)
        else:
            for idx, robot_i in enumerate(self.controlled_indices):
                d = float(action[2 * idx])
                dtheta = float(action[2 * idx + 1])
                theta = self.state[robot_i, 2]
                
                dx = d * np.cos(theta)
                dy = d * np.sin(theta)
                
                proposed_state[robot_i, 0] = np.clip(
                    self.state[robot_i, 0] + dx, self.x_min, self.x_max)
                proposed_state[robot_i, 1] = np.clip(
                    self.state[robot_i, 1] + dy, self.y_min, self.y_max)
                proposed_state[robot_i, 2] = self._wrap_angle(theta + dtheta)

        # Uncontrolled robots stay stationary

        collision = self._check_collision(proposed_state)
        
        if not collision:
            self.state = proposed_state

        success = self._check_success()
        
        obs = self._get_obs()
        info = {"is_success": bool(success), "collision": bool(collision)}

        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            info["time_limit"] = True

        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        if isinstance(reward, np.ndarray):
            reward = float(reward.squeeze())

        terminated = bool(self.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info))
        truncated = bool(self.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info))

        self.episode_reward += reward
        self.episode_length += 1
        
        if terminated or truncated:
            info["episode"] = {"r": float(self.episode_reward), "l": int(self.episode_length)}
            self.episode_reward = 0.0
            self.episode_length = 0
            self.step_count = 0

        if self.render_mode == "human":
            if collision:
                # Render proposed state with collision highlighting
                self.render(state=proposed_state, highlight_collision=True)
            else:
                self.render()

        return obs, reward, terminated, truncated, info

    def _check_collision(self, state):
        """Check for collisions in proposed state."""
        positions = state[:, :2]
        
        # Check boundary for controlled robots only
        for robot_i in self.controlled_indices:
            x, y = positions[robot_i]
            if not (self.x_min + self.robot_radius <= x <= self.x_max - self.robot_radius and
                    self.y_min + self.robot_radius <= y <= self.y_max - self.robot_radius):
                return True
        
        # Check inter-robot collisions (between all robots)
        for i in range(6):
            for j in range(i + 1, 6):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2 * self.robot_radius:
                    return True
        
        # Check obstacle collisions for controlled robots only
        for robot_i in self.controlled_indices:
            pos = positions[robot_i]
            for obs in self.obstacles:
                if obs['type'] == 'circle':
                    dist = np.linalg.norm(pos - np.array([obs['x'], obs['y']]))
                    if dist < self.robot_radius + obs['radius']:
                        return True
        
        return False

    def _check_success(self):
        """Check if all controlled robots reached their goals."""
        for idx, robot_i in enumerate(self.controlled_indices):
            pos = self.state[robot_i, :2]
            goal = self.goal_position[idx]
            if np.linalg.norm(pos - goal) > self.goal_threshold:
                return False
        return True

    def _compute_obstacle_info_for_robot(self, robot_idx):
        """Compute distance and bearing to other controlled robots, uncontrolled robots, and static obstacles."""
        robot_pos = self.state[robot_idx, :2]
        info = []
        
        # Other controlled robots (exclude self)
        for controlled_i in self.controlled_indices:
            if controlled_i == robot_idx:
                continue
            other_robot_pos = self.state[controlled_i, :2]
            rel_vec = other_robot_pos - robot_pos
            distance = np.linalg.norm(rel_vec)
            
            if distance > 1e-6:
                bearing = np.arctan2(rel_vec[1], rel_vec[0])
                info.extend([distance, np.cos(bearing), np.sin(bearing)])
            else:
                info.extend([0.0, 1.0, 0.0])
        
        # Uncontrolled robots (stationary obstacles)
        for uncontrolled_i in self.uncontrolled_indices:
            obstacle_pos = self.state[uncontrolled_i, :2]
            rel_vec = obstacle_pos - robot_pos
            distance = np.linalg.norm(rel_vec)
            
            if distance > 1e-6:
                bearing = np.arctan2(rel_vec[1], rel_vec[0])
                info.extend([distance, np.cos(bearing), np.sin(bearing)])
            else:
                info.extend([0.0, 1.0, 0.0])
        
        # Static obstacles
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                obstacle_pos = np.array([obs['x'], obs['y']])
                rel_vec = obstacle_pos - robot_pos
                distance = np.linalg.norm(rel_vec)
                
                if distance > 1e-6:
                    bearing = np.arctan2(rel_vec[1], rel_vec[0])
                    info.extend([distance, np.cos(bearing), np.sin(bearing)])
                else:
                    info.extend([0.0, 1.0, 0.0])
        
        return info

    def _get_obs(self):
        """Build observation dict matching Env_v6 structure."""
        obs = []
        
        # Per controlled robot observations
        for robot_i in self.controlled_indices:
            x, y, theta = self.state[robot_i]
            
            # Absolute position
            obs.extend([x, y])
            
            # Robot orientation
            obs.extend([np.cos(theta), np.sin(theta)])
            
            # Obstacle information (uncontrolled robots + static obstacles)
            obstacle_info = self._compute_obstacle_info_for_robot(robot_i)
            obs.extend(obstacle_info)
        
        # Achieved and desired goals
        achieved_goal = np.array([self.state[i, :2] for i in self.controlled_indices]).flatten()
        desired_goal = self.goal_position.flatten()
        
        return {
            "observation": np.array(obs, dtype=np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward (HER compatible)."""
        achieved = self._reshape_goals(achieved_goal)
        desired = self._reshape_goals(desired_goal)
        batch_size = achieved.shape[0]

        # Distance-based reward
        distances = np.linalg.norm(achieved - desired, axis=2).sum(axis=1)
        rewards = self.step_penalty - 0.05 * distances

        # Proximity penalty
        proximity_penalties = self._compute_proximity_penalty()
        if isinstance(proximity_penalties, (int, float)):
            proximity_penalties = np.full(batch_size, proximity_penalties, dtype=np.float32)
        rewards += proximity_penalties

        # Success bonus
        success_mask = np.all(np.linalg.norm(achieved - desired, axis=2) <= self.goal_threshold, axis=1)
        rewards = np.where(success_mask, self.goal_reward, rewards)

        # Collision penalty
        collision_mask = self._extract_flag_from_info(info, "collision", False, batch_size)
        rewards = np.where(collision_mask, self.collision_penalty, rewards)

        rewards = rewards.astype(np.float32)
        if rewards.size == 1:
            return float(rewards[0])
        return rewards

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Terminate on success or collision."""
        success_mask = self._success_mask(achieved_goal, desired_goal)
        batch_size = success_mask.shape[0]
        collision_mask = self._extract_flag_from_info(info, "collision", False, batch_size)
        terminated = np.logical_or(success_mask, collision_mask)
        return bool(terminated[0]) if terminated.size == 1 else terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Truncate on time limit."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        if achieved.ndim == 1:
            achieved = achieved[None, :]
        batch_size = achieved.shape[0]
        truncated = self._extract_flag_from_info(info, "time_limit", False, batch_size)
        return bool(truncated[0]) if truncated.size == 1 else truncated

    def _reshape_goals(self, goals):
        goals = np.asarray(goals, dtype=np.float32)
        if goals.ndim == 1:
            goals = goals[None, :]
        return goals.reshape(goals.shape[0], self.num_controlled, 2)

    def _success_mask(self, achieved_goal, desired_goal):
        achieved = self._reshape_goals(achieved_goal)
        desired = self._reshape_goals(desired_goal)
        within_threshold = np.linalg.norm(achieved - desired, axis=2) <= self.goal_threshold
        return np.all(within_threshold, axis=1)

    def _compute_proximity_penalty(self):
        """Compute proximity penalty for controlled robots."""
        positions = self.state[:, :2]
        penalty = 0.0
        
        # Check proximity between controlled robots and all other robots
        for robot_i in self.controlled_indices:
            for robot_j in range(6):
                if robot_i == robot_j:
                    continue
                    
                dist = np.linalg.norm(positions[robot_i] - positions[robot_j])
                if dist < self.proximity_threshold:
                    min_distance = 2 * self.robot_radius
                    if dist < min_distance:
                        dist = min_distance
                    penalty_scale = (self.proximity_threshold - dist) / (self.proximity_threshold - min_distance)
                    penalty += self.max_proximity_penalty * penalty_scale
        
        return penalty

    @staticmethod
    def _extract_flag_from_info(info, key, default, batch_size):
        if info is None:
            return np.full(batch_size, default, dtype=bool)
        if isinstance(info, dict):
            return np.full(batch_size, info.get(key, default), dtype=bool)
        return np.array([item.get(key, default) for item in info], dtype=bool, copy=False)

    @staticmethod
    def _wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def render(self, state=None, highlight_collision=False):
        """
        Render the environment.
        
        Args:
            state: Optional state array (6, 3) to render. If None, uses self.state
            highlight_collision: If True, render all controlled robots in red
        
        Returns:
            frame if render_mode is "rgb_array", otherwise None
        """
        frame = self._render_frame(state=state, highlight_collision=highlight_collision)
        if self.render_mode == "human":
            plt.figure("ControlledSubsetEnv")
            plt.clf()
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(1.0 / self.metadata["render_fps"])
        return frame if self.render_mode == "rgb_array" else None

    def _render_frame(self, state=None, highlight_collision=False):
        """
        Render with controlled robots in color, uncontrolled in gray.
        
        Args:
            state: State array (6, 3) to render. If None, uses self.state
            highlight_collision: If True, render all controlled robots in red
        """
        if state is None:
            state = self.state
            
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.x_min - 0.5, self.x_max + 0.5)
        ax.set_ylim(self.y_min - 0.5, self.y_max + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        # Draw static obstacles
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                ax.add_patch(Circle((obs['x'], obs['y']), obs['radius'], 
                                   color='gray', alpha=0.5))

        # Draw uncontrolled robots (gray, stationary)
        for robot_i in self.uncontrolled_indices:
            x, y, theta = state[robot_i]
            
            ax.add_patch(Circle((x, y), self.robot_radius, color='lightgray', alpha=0.7, 
                                edgecolor='darkgray', linewidth=2))
            ax.text(x, y, str(robot_i), ha='center', va='center', fontsize=9, color='gray')
            
            hdx = 0.3 * np.cos(theta)
            hdy = 0.3 * np.sin(theta)
            ax.arrow(x, y, hdx, hdy, head_width=0.1, head_length=0.05, 
                    fc='darkgray', ec='darkgray', alpha=0.5)

        # Draw controlled robots (colored, or red if collision)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for idx, robot_i in enumerate(self.controlled_indices):
            x, y, theta = state[robot_i]
            
            # Use red for collision highlighting, otherwise use normal colors
            if highlight_collision:
                color = 'red'
            else:
                color = colors[idx % len(colors)]
            
            ax.add_patch(Circle((x, y), self.robot_radius, color=color, alpha=0.8))
            
            hdx = 0.5 * np.cos(theta)
            hdy = 0.5 * np.sin(theta)
            ax.arrow(x, y, hdx, hdy, head_width=0.15, head_length=0.1, fc='black', ec='black')
            
            goal_x, goal_y = self.goal_position[idx]
            ax.add_patch(Circle((goal_x, goal_y), 0.25, color=color, alpha=0.3))
            ax.plot(goal_x, goal_y, 'x', color=color, markersize=12, markeredgewidth=3)
            
            ax.text(x, y, str(robot_i), ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold')

        title = f"Controlled: {self.controlled_indices} | Step: {self.step_count}"
        if highlight_collision:
            title += " | COLLISION!"
        ax.set_title(title)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return frame

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close("ControlledSubsetEnv")
