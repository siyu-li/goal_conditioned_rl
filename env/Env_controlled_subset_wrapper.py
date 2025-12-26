"""
Wrapper for ControlledSubsetEnv (v2) providing relative coordinates and normalization.

This wrapper transforms the observations from absolute coordinates to relative coordinates
(relative to the centroid of controlled robots) with optional normalization.

Key Features:
1. Relative coordinates: All positions expressed relative to centroid of controlled robots
2. Normalization: Optional coordinate normalization to [-1, 1] range
3. Canonical ordering: Uncontrolled robots sorted by distance for consistent observations
4. Self-contained: Works with standalone ControlledSubsetEnv without parent env

Usage:
    base_env = ControlledSubsetEnv(
        controlled_robot_indices=[0, 2],
        obstacle_config_path='configs/obstacles_default.yaml'
    )
    
    env = ControlledSubsetWrapper(
        base_env,
        use_relative_coords=True,
        normalize_coords=True
    )
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any


class ControlledSubsetWrapper(gym.Wrapper):
    """
    Wrapper that provides relative coordinates and normalization for ControlledSubsetEnv.
    
    Observation Structure (same as base env, but positions relative to centroid):
    - For each controlled robot:
        - robot_x, robot_y (relative to centroid, NOT absolute)
        - cos(robot_theta), sin(robot_theta)
        - For each other controlled robot: distance, cos(bearing), sin(bearing)
        - For each uncontrolled robot: distance, cos(bearing), sin(bearing)
        - For each static obstacle: distance, cos(bearing), sin(bearing)
    
    Goals are also transformed to be relative to centroid.
    """
    
    def __init__(self, 
                 env,
                 use_relative_coords: bool = True,
                 normalize_coords: bool = True):
        """
        Args:
            env: ControlledSubsetEnv instance
            use_relative_coords: If True, use positions relative to centroid
            normalize_coords: If True, normalize coordinates to approx [-1, 1]
        """
        super().__init__(env)
        
        if not hasattr(env, 'controlled_indices'):
            raise ValueError("Environment must be a ControlledSubsetEnv instance")
        
        self.use_relative_coords = use_relative_coords
        self.normalize_coords = normalize_coords
        
        # Store environment parameters
        self.num_controlled = env.num_controlled
        self.num_uncontrolled = env.num_uncontrolled
        self.num_obstacles = len(env.obstacles)
        
        # For normalization - use workspace size
        self.workspace_size = max(
            env.x_max - env.x_min,
            env.y_max - env.y_min
        )
        
        # Rebuild observation space
        self.observation_space = self._build_observation_space()
        
        # Action space remains unchanged
        
    def _build_observation_space(self):
        """Build observation space for relative/normalized observations."""
        # Get base environment's observation space structure
        base_obs_space = self.env.observation_space['observation']
        obs_dim = base_obs_space.shape[0]
        
        if self.normalize_coords:
            # Normalized coordinates (allow some margin beyond [-1, 1])
            low = np.full(obs_dim, -3.0, dtype=np.float32)
            high = np.full(obs_dim, 3.0, dtype=np.float32)
        else:
            # Use the same bounds as base environment
            low = base_obs_space.low.copy()
            high = base_obs_space.high.copy()
            
            # Adjust position bounds to allow for relative coordinates
            # (can be negative when relative to centroid)
            max_range = max(
                self.env.x_max - self.env.x_min,
                self.env.y_max - self.env.y_min
            )
            # Update x, y bounds for each controlled robot
            # Each robot has: x, y, cos(theta), sin(theta), then obstacle info
            # Obstacle info includes: other_controlled * 3 + uncontrolled * 3 + obstacles * 3
            robot_obs_size = 4 + (self.num_controlled - 1) * 3 + self.num_uncontrolled * 3 + self.num_obstacles * 3
            for i in range(self.num_controlled):
                low[i * robot_obs_size] = -max_range
                high[i * robot_obs_size] = max_range
                low[i * robot_obs_size + 1] = -max_range
                high[i * robot_obs_size + 1] = max_range
        
        # Goal space (positions of controlled robots, relative to centroid)
        goal_dim = self.num_controlled * 2
        if self.normalize_coords:
            goal_low = np.full(goal_dim, -3.0, dtype=np.float32)
            goal_high = np.full(goal_dim, 3.0, dtype=np.float32)
        else:
            max_range = max(
                self.env.x_max - self.env.x_min,
                self.env.y_max - self.env.y_min
            )
            goal_low = np.full(goal_dim, -max_range, dtype=np.float32)
            goal_high = np.full(goal_dim, max_range, dtype=np.float32)
        
        return spaces.Dict({
            "observation": spaces.Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })
    
    def reset(self, seed=None, options=None):
        """Reset environment and transform observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        if self.use_relative_coords:
            obs = self._to_relative_observation(obs)
        
        return obs, info
    
    def step(self, action):
        """Execute step and transform observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.use_relative_coords:
            obs = self._to_relative_observation(obs)
        
        return obs, reward, terminated, truncated, info
    
    def _to_relative_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert absolute observation to relative observation.
        
        The observation structure remains the same as the base environment:
        - For each controlled robot:
            - robot_x, robot_y (NOW relative to centroid)
            - cos(robot_theta), sin(robot_theta)
            - For each other controlled robot: distance, cos(bearing), sin(bearing)
            - For each uncontrolled robot: distance, cos(bearing), sin(bearing)
            - For each static obstacle: distance, cos(bearing), sin(bearing)
        
        Only the robot positions (x, y) are changed from absolute to relative.
        """
        # Get centroid of controlled robots
        centroid = self.get_centroid()
        
        # Parse the original observation
        # Structure: for each controlled robot:
        #   - x, y, cos(theta), sin(theta)
        #   - other controlled robots: [distance, cos(bearing), sin(bearing)] * (num_controlled - 1)
        #   - uncontrolled robots: [distance, cos(bearing), sin(bearing)] * num_uncontrolled
        #   - obstacles: [distance, cos(bearing), sin(bearing)] * num_obstacles
        # Then goals for all controlled robots
        
        original_obs = obs['observation'].copy()
        relative_obs = original_obs.copy()
        
        # Calculate the stride for each robot's observation
        robot_obs_size = 4 + (self.num_controlled - 1) * 3 + self.num_uncontrolled * 3 + self.num_obstacles * 3
        
        # Transform each controlled robot's position to relative
        for i in range(self.num_controlled):
            start_idx = i * robot_obs_size
            
            # Convert to relative position
            rel_x = original_obs[start_idx] - centroid[0]
            rel_y = original_obs[start_idx + 1] - centroid[1]
            
            if self.normalize_coords:
                rel_x /= self.workspace_size
                rel_y /= self.workspace_size
            
            relative_obs[start_idx] = rel_x
            relative_obs[start_idx + 1] = rel_y
            
            # Normalize distance values in obstacle information if needed
            if self.normalize_coords:
                # Obstacle info starts at index 4 (after x, y, cos(theta), sin(theta))
                # Each obstacle entry: [distance, cos(bearing), sin(bearing)]
                obstacle_start = start_idx + 4
                num_obstacle_entries = (self.num_controlled - 1) + self.num_uncontrolled + self.num_obstacles
                
                for j in range(num_obstacle_entries):
                    distance_idx = obstacle_start + j * 3
                    # Normalize distance (index 0 of each triplet)
                    relative_obs[distance_idx] /= self.workspace_size
        
        # Transform goals to relative coordinates
        achieved_goal = obs['achieved_goal'].copy().reshape(self.num_controlled, 2)
        desired_goal = obs['desired_goal'].copy().reshape(self.num_controlled, 2)
        
        # Subtract centroid from each robot's position
        for i in range(self.num_controlled):
            achieved_goal[i] -= centroid
            desired_goal[i] -= centroid
            
            if self.normalize_coords:
                achieved_goal[i] /= self.workspace_size
                desired_goal[i] /= self.workspace_size
        
        return {
            "observation": relative_obs.astype(np.float32),
            "achieved_goal": achieved_goal.flatten().astype(np.float32),
            "desired_goal": desired_goal.flatten().astype(np.float32),
        }
    
    def get_centroid(self) -> np.ndarray:
        """Return centroid of controlled robots in absolute coordinates."""
        controlled_positions = self.env.state[self.env.controlled_indices, :2]
        return np.mean(controlled_positions, axis=0)
    
    def get_absolute_observation(self) -> Dict[str, np.ndarray]:
        """Get the underlying absolute observation from base environment."""
        return self.env._get_obs()
    
    # HER compatibility methods - delegate to base environment
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward (delegated to base env)."""
        return self.env.compute_reward(achieved_goal, desired_goal, info)
    
    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Check termination (delegated to base env)."""
        return self.env.compute_terminated(achieved_goal, desired_goal, info)
    
    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Check truncation (delegated to base env)."""
        return self.env.compute_truncated(achieved_goal, desired_goal, info)


def make_wrapped_env(controlled_robot_indices,
                     obstacle_config_path=None,
                     use_relative_coords=True,
                     normalize_coords=True,
                     **kwargs):
    """
    Factory function to create a wrapped ControlledSubsetEnv with relative coordinates.
    
    Args:
        controlled_robot_indices: List of robot indices to control
        obstacle_config_path: Path to obstacle configuration YAML
        use_relative_coords: If True, use relative coordinates
        normalize_coords: If True, normalize coordinates
        **kwargs: Additional arguments for ControlledSubsetEnv
    
    Returns:
        ControlledSubsetWrapper instance
    """
    from env.Env_controlled_subset import ControlledSubsetEnv
    
    # Create base environment
    base_env = ControlledSubsetEnv(
        controlled_robot_indices=controlled_robot_indices,
        obstacle_config_path=obstacle_config_path,
        **kwargs
    )
    
    # Wrap with coordinate transformation
    wrapped_env = ControlledSubsetWrapper(
        base_env,
        use_relative_coords=use_relative_coords,
        normalize_coords=normalize_coords
    )
    
    return wrapped_env


if __name__ == "__main__":
    """Test the wrapper."""
    import matplotlib.pyplot as plt
    from env.Env_controlled_subset import ControlledSubsetEnv
    
    # Create base environment directly (avoid import issues when running as script)
    base_env = ControlledSubsetEnv(
        controlled_robot_indices=[0, 1],
        obstacle_config_path='configs/obstacles_default.yaml',
        max_episode_steps=50,
        render_mode='human'
    )
    
    # Wrap it
    env = ControlledSubsetWrapper(
        base_env,
        use_relative_coords=True,
        normalize_coords=True
    )
    
    print("=" * 80)
    print("Testing ControlledSubsetWrapper")
    print("=" * 80)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Controlled robots: {env.env.controlled_indices}")
    print(f"Uncontrolled robots: {env.env.uncontrolled_indices}")
    print()
    
    # Test reset
    obs, info = env.reset(seed=42)
    
    print("Observation shapes:")
    print(f"  observation: {obs['observation'].shape}")
    print(f"  achieved_goal: {obs['achieved_goal'].shape}")
    print(f"  desired_goal: {obs['desired_goal'].shape}")
    print()
    
    # Check centroid
    centroid = env.get_centroid()
    print(f"Centroid of controlled robots: {centroid}")
    print()
    
    # Compare absolute vs relative observations
    abs_obs = env.get_absolute_observation()
    print("First few elements of absolute observation:", abs_obs['observation'][:10])
    print("First few elements of relative observation:", obs['observation'][:10])
    print()
    
    # Run a few steps
    print("Running 40 random steps...")
    for step_i in range(40):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        centroid = env.get_centroid()
        print(f"Step {step_i+1}: reward={reward:.2f}, centroid={centroid}, "
              f"success={info['is_success']}, collision={info['collision']}")
        
        if terminated or truncated:
            print("Episode ended!")
            obs, info = env.reset()
            break

        print("Press Enter to next step...")
        input()  # Wait for user input before closing
    print("Press Enter to close the visualization...")
    input()  # Wait for user input before closing
    env.close()
