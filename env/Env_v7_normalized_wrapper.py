"""
Normalization wrapper for Env_v7 that normalizes all positions and distances to a consistent scale.

This wrapper:
1. Normalizes robot positions in observations to [-1, 1]
2. Normalizes goal positions in observations to [-1, 1]
3. Normalizes distances (to other robots/obstacles) using the SAME SCALE as positions
   - E.g., if workspace is [0,10] → positions map to [-1,1] (scale: 2/10 = 0.2)
   - Then distances are also scaled by 0.2 (distance of 10 → 2.0, distance of 5 → 1.0)
4. Normalizes achieved_goal and desired_goal to [-1, 1] (for HER compatibility)
5. Adjusts goal_threshold and proximity_threshold to normalized space

Key benefits:
- Consistent coordinate frame: positions and distances use the same scale
- Better neural network convergence with normalized inputs
- Maintains HER compatibility (goals remain in same coordinate system)
- Preserves distance-based reward calculations (relative magnitudes maintained)

Example normalization (workspace [0, 10] x [0, 10]):
- Position (5.0, 5.0) → (0.0, 0.0)  [center]
- Position (10.0, 10.0) → (1.0, 1.0)  [corner]
- Distance 5.0 → 1.0
- Distance 10.0 → 2.0
- Distance 3.0 → 0.6
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NormalizedEnvV7Wrapper(gym.Wrapper):
    """Wrapper that normalizes positions and distances for Env_v7."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Store workspace dimensions for normalization
        self.x_min = env.x_min
        self.x_max = env.x_max
        self.y_min = env.y_min
        self.y_max = env.y_max
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        
        # Position normalization scale: positions map from [0, range] to [-1, 1] (span of 2)
        # So the scale factor is: 2 / range
        self.position_scale = 2.0 / self.x_range  # Assumes square workspace or use x_range as reference
        
        # Adjust thresholds to normalized space
        # Original goal_threshold in original space gets scaled by position_scale
        # E.g., goal_threshold = 1.0, position_scale = 0.2 → normalized_threshold = 0.2
        self.normalized_goal_threshold = self.env.goal_threshold * self.position_scale
        
        # Original proximity_threshold also gets scaled by position_scale
        self.normalized_proximity_threshold = self.env.proximity_threshold * self.position_scale
        
        # Calculate max normalized distance (must be set before building observation bounds)
        self.max_normalized_distance = self.env.max_distance * self.position_scale
        
        # Build normalized observation space
        obs_low, obs_high = self._build_normalized_observation_bounds()
        
        # Normalized goal space: [-1, 1] for each (x, y) position
        goal_low = np.tile([-1.0, -1.0], self.env.num_robots).astype(np.float32)
        goal_high = np.tile([1.0, 1.0], self.env.num_robots).astype(np.float32)
        
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
                "desired_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            }
        )

    def _build_normalized_observation_bounds(self):
        """Build observation bounds with normalized positions and distances."""
        # Per robot: [x, y, cos(theta), sin(theta), (dist, cos_bear, sin_bear) x (num_robots-1), (dist, cos_bear, sin_bear) x num_obstacles]
        per_robot = [
            [-1.0, 1.0], [-1.0, 1.0],  # position (x, y)
            [-1.0, 1.0], [-1.0, 1.0],  # orientation (cos, sin)
        ]
        # Other robots: distance + bearing
        for _ in range(self.env.num_robots - 1):
            per_robot.extend([[0.0, self.max_normalized_distance], [-1.0, 1.0], [-1.0, 1.0]])
        # Obstacles: distance + bearing
        for _ in range(self.env.num_obstacles):
            per_robot.extend([[0.0, self.max_normalized_distance], [-1.0, 1.0], [-1.0, 1.0]])
        
        # All robots
        all_bounds = per_robot * self.env.num_robots
        # Goals
        for _ in range(self.env.num_robots):
            all_bounds.extend([[-1.0, 1.0], [-1.0, 1.0]])
        
        bounds = np.array(all_bounds, dtype=np.float32)
        return bounds[:, 0], bounds[:, 1]  # low, high
    
    def normalize_position(self, x, y):
        """
        Normalize position from [x_min, x_max] x [y_min, y_max] to [-1, 1] x [-1, 1].
        
        Formula: normalized = 2 * (value - min) / (max - min) - 1
        """
        x_norm = 2.0 * (x - self.x_min) / self.x_range - 1.0
        y_norm = 2.0 * (y - self.y_min) / self.y_range - 1.0
        return x_norm, y_norm
    
    def denormalize_position(self, x_norm, y_norm):
        """
        Denormalize position from [-1, 1] x [-1, 1] back to original space.
        
        Formula: value = (normalized + 1) / 2 * (max - min) + min
        """
        x = (x_norm + 1.0) / 2.0 * self.x_range + self.x_min
        y = (y_norm + 1.0) / 2.0 * self.y_range + self.y_min
        return x, y
    
    def normalize_distance(self, distance):
        """
        Normalize distance using the same scale as positions.
        
        Since positions are scaled by position_scale (2 / range),
        distances should use the same scale for consistency.
        
        E.g., if workspace is [0, 10] and positions map to [-1, 1]:
        - position_scale = 2 / 10 = 0.2
        - distance of 10 → 10 * 0.2 = 2.0
        - distance of 5 → 5 * 0.2 = 1.0
        """
        return distance * self.position_scale
    
    def denormalize_distance(self, distance_norm):
        """
        Denormalize distance back to original space.
        
        Inverse of normalize_distance.
        """
        return distance_norm / self.position_scale
    
    def normalize_positions_array(self, positions):
        """
        Normalize an array of positions using vectorized operations.
        
        Args:
            positions: (..., 2) array with [x, y] positions
        
        Returns:
            normalized_positions: same shape with normalized values
        """
        positions = np.asarray(positions, dtype=np.float32)
        # Vectorized normalization: (pos - min) * (2 / range) - 1
        normalized = positions.copy()
        normalized[..., 0] = 2.0 * (positions[..., 0] - self.x_min) / self.x_range - 1.0
        normalized[..., 1] = 2.0 * (positions[..., 1] - self.y_min) / self.y_range - 1.0
        return normalized
    
    def denormalize_positions_array(self, positions_norm):
        """
        Denormalize an array of positions using vectorized operations.
        
        Args:
            positions_norm: (..., 2) array with normalized [x, y] positions
        
        Returns:
            positions: same shape with denormalized values
        """
        positions_norm = np.asarray(positions_norm, dtype=np.float32)
        # Vectorized denormalization: (pos_norm + 1) / 2 * range + min
        denormalized = positions_norm.copy()
        denormalized[..., 0] = (positions_norm[..., 0] + 1.0) / 2.0 * self.x_range + self.x_min
        denormalized[..., 1] = (positions_norm[..., 1] + 1.0) / 2.0 * self.y_range + self.y_min
        return denormalized
    
    def _normalize_observation(self, obs_dict):
        """
        Normalize observation dictionary from unwrapped environment.
        
        Normalizes:
        - Robot positions in observation to [-1, 1]
        - Distances to other robots/obstacles using same scale as positions
        - Goal positions in observation to [-1, 1]
        - achieved_goal to [-1, 1]
        - desired_goal to [-1, 1]
        """
        obs = obs_dict["observation"].copy()
        
        # Calculate observation structure offsets
        # Per robot: [x, y, cos(theta), sin(theta), (dist, cos_bear, sin_bear) x (num_robots-1), (dist, cos_bear, sin_bear) x num_obstacles]
        per_robot_other_robots = 3 * (self.env.num_robots - 1)
        per_robot_obstacles = 3 * self.env.num_obstacles
        per_robot_size = 4 + per_robot_other_robots + per_robot_obstacles
        
        # Normalize per-robot observations
        for robot_idx in range(self.env.num_robots):
            offset = robot_idx * per_robot_size
            
            # Normalize robot position (first 2 elements)
            x, y = obs[offset], obs[offset + 1]
            obs[offset], obs[offset + 1] = self.normalize_position(x, y)
            
            # Normalize distances to other robots
            for other_idx in range(self.env.num_robots - 1):
                dist_offset = offset + 4 + other_idx * 3
                obs[dist_offset] = self.normalize_distance(obs[dist_offset])
            
            # Normalize distances to obstacles
            for obs_idx in range(self.env.num_obstacles):
                dist_offset = offset + 4 + per_robot_other_robots + obs_idx * 3
                obs[dist_offset] = self.normalize_distance(obs[dist_offset])
        
        # Normalize goal positions at the end of observation
        goal_offset = self.env.num_robots * per_robot_size
        for goal_idx in range(self.env.num_robots):
            goal_x, goal_y = obs[goal_offset + goal_idx * 2], obs[goal_offset + goal_idx * 2 + 1]
            obs[goal_offset + goal_idx * 2], obs[goal_offset + goal_idx * 2 + 1] = self.normalize_position(goal_x, goal_y)
        
        # Normalize achieved_goal and desired_goal
        achieved_goal_norm = self.normalize_positions_array(
            obs_dict["achieved_goal"].reshape(self.env.num_robots, 2)
        ).flatten()
        
        desired_goal_norm = self.normalize_positions_array(
            obs_dict["desired_goal"].reshape(self.env.num_robots, 2)
        ).flatten()
        
        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal_norm.astype(np.float32),
            "desired_goal": desired_goal_norm.astype(np.float32),
        }
    
    def reset(self, **kwargs):
        """Reset environment and return normalized observation."""
        obs_dict, info = self.env.reset(**kwargs)
        normalized_obs = self._normalize_observation(obs_dict)
        return normalized_obs, info
    
    def step(self, action):
        """Step environment and return normalized observation."""
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        normalized_obs = self._normalize_observation(obs_dict)
        return normalized_obs, reward, terminated, truncated, info
    
    def _denormalize_goals(self, achieved_goal, desired_goal):
        """Helper to denormalize goals for compute methods."""
        achieved_denorm = self.denormalize_positions_array(
            np.asarray(achieved_goal).reshape(-1, self.env.num_robots, 2)
        ).reshape(np.asarray(achieved_goal).shape)
        
        desired_denorm = self.denormalize_positions_array(
            np.asarray(desired_goal).reshape(-1, self.env.num_robots, 2)
        ).reshape(np.asarray(desired_goal).shape)
        
        return achieved_denorm, desired_denorm
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward with normalized goals."""
        achieved_denorm, desired_denorm = self._denormalize_goals(achieved_goal, desired_goal)
        return self.env.compute_reward(achieved_denorm, desired_denorm, info)
    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Compute termination with normalized goals."""
        achieved_denorm, desired_denorm = self._denormalize_goals(achieved_goal, desired_goal)
        return self.env.compute_terminated(achieved_denorm, desired_denorm, info)
    
    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Compute truncation with normalized goals."""
        achieved_denorm, desired_denorm = self._denormalize_goals(achieved_goal, desired_goal)
        return self.env.compute_truncated(achieved_denorm, desired_denorm, info)
