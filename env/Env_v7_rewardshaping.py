## Multi-robot individually controlled environment with centralized state/action space
## and relative position observations for HER and goal-conditioned RL compatibility.
## Step reward is step_penalty - distance_to_goal_reward + collision_penalty + goal_reward
## Supports arbitrary number of robots specified via num_robots parameter.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow


class CentralizedMultiRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps, num_robots=3, action_coupled=False, robot_radius=0.5, num_obstacles=0, render_mode=None):
        super().__init__()
        # Episode configuration
        self.max_episode_steps = max_episode_steps
        self.num_robots = int(num_robots)
        self.robot_radius = float(robot_radius)
        self.num_obstacles = int(num_obstacles)  # Number of obstacles (0, 1, 2, or 3)
        self.obstacle_radius = float(robot_radius)  # Obstacles have same size as robots
        self.render_mode = render_mode
        self.action_coupled = bool(action_coupled)

        # Workspace boundaries for x/y coordinates
        self.x_min, self.x_max = 0.0, 10.0
        self.y_min, self.y_max = 0.0, 10.0
        self.max_distance = np.sqrt(2) * max(self.x_max, self.y_max)
        # Actions depend on action_coupled:
        # If coupled: [shared_distance, dtheta_robot1, dtheta_robot2, ..., dtheta_robotN]
        # If uncoupled: [distance_robot1, dtheta_robot1, distance_robot2, dtheta_robot2, ..., distance_robotN, dtheta_robotN]
        self.max_step_size = 0.5
        if self.action_coupled:
            # One shared distance for all robots + heading change per robot
            action_low = np.array([0.0] + [-np.pi] * self.num_robots, dtype=np.float32)
            action_high = np.array([self.max_step_size] + [np.pi] * self.num_robots, dtype=np.float32)
        else:
            # Each robot has its own distance and heading change
            action_low = np.tile([0.0, -np.pi], self.num_robots).astype(np.float32)
            action_high = np.tile([self.max_step_size, np.pi], self.num_robots).astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation space
        obs_low, obs_high = self._build_observation_bounds()
        

        goal_low = np.tile([self.x_min, self.y_min], self.num_robots).astype(np.float32)
        goal_high = np.tile([self.x_max, self.y_max], self.num_robots).astype(np.float32)
        
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                "achieved_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
                "desired_goal": spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            }
        )

        self.goal_threshold = 1.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.collision_penalty = -100.0
        
        # Robot-to-robot proximity penalty parameters
        self.proximity_threshold = 3  # Distance threshold for proximity penalty
        self.max_proximity_penalty = -10.0  # Maximum penalty when robots are at collision distance
        
        # Obstacle proximity penalty parameters
        self.obstacle_proximity_threshold = 3.0  # Distance threshold for obstacle proximity penalty
        self.max_obstacle_proximity_penalty = -10.0  # Maximum penalty when robot is at collision distance with obstacle

        # Initialize obstacles first, then state and goals (to avoid circular dependency)
        self.obstacles = self._sample_random_obstacles()  # Initialize obstacles first
        self.state = self._sample_random_state()  # Then sample robot positions avoiding obstacles
        self.goal_position = self._sample_random_goals()  # Then sample goals avoiding obstacles
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        if self.render_mode == "human":
            plt.ion()

    def _build_observation_bounds(self):
        """
        Build observation bounds for the new observation structure.
        
        Per robot observation:
        - robot_x, robot_y (absolute position)
        - cos(robot_theta), sin(robot_theta)
        - For each OTHER robot: distance, cos(bearing), sin(bearing)
        - For each OBSTACLE: distance, cos(bearing), sin(bearing)
        
        Plus absolute goal positions at the end.
        """
        obs_low = []
        obs_high = []
        
        max_distance = np.sqrt(2) * max(self.x_max, self.y_max)
        
        # Per robot observations
        for _ in range(self.num_robots):
            # Absolute position (x, y)
            obs_low.extend([self.x_min, self.y_min])
            obs_high.extend([self.x_max, self.y_max])
            
            # Robot orientation as cos(theta) and sin(theta)
            obs_low.extend([-1.0, -1.0])
            obs_high.extend([1.0, 1.0])
            
            # For each OTHER robot: distance, cos(bearing), sin(bearing)
            for _ in range(self.num_robots - 1):
                obs_low.extend([0.0, -1.0, -1.0])
                obs_high.extend([max_distance, 1.0, 1.0])
            
            # For each OBSTACLE: distance, cos(bearing), sin(bearing)
            for _ in range(self.num_obstacles):
                obs_low.extend([0.0, -1.0, -1.0])
                obs_high.extend([max_distance, 1.0, 1.0])
        
        # Goals as absolute positions
        for _ in range(self.num_robots):
            obs_low.extend([self.x_min, self.y_min])
            obs_high.extend([self.x_max, self.y_max])
        
        return np.array(obs_low, dtype=np.float32), np.array(obs_high, dtype=np.float32)

    def step(self, action):
        # Enforce action bounds so policy errors cannot break simulation
        action = np.clip(action, self.action_space.low, self.action_space.high)
        proposed_state = self.state.copy()
        
        if self.action_coupled:
            # Extract shared distance (first action element)
            shared_distance = float(action[0])
            
            # Apply shared distance and individual heading changes to each robot
            for i in range(self.num_robots):
                dtheta = float(action[1 + i])  # Heading change for robot i
                theta = self.state[i, 2]
                new_x = self.state[i, 0] + shared_distance * np.cos(theta)
                new_y = self.state[i, 1] + shared_distance * np.sin(theta)
                new_theta = self._wrap_angle(theta + dtheta)
                proposed_state[i, 0] = np.clip(new_x, self.x_min, self.x_max)
                proposed_state[i, 1] = np.clip(new_y, self.y_min, self.y_max)
                proposed_state[i, 2] = new_theta
        else:
            # Each robot has its own distance and heading change
            for i in range(self.num_robots):
                d = float(action[2 * i])
                dtheta = float(action[2 * i + 1])
                theta = self.state[i, 2]
                new_x = self.state[i, 0] + d * np.cos(theta)
                new_y = self.state[i, 1] + d * np.sin(theta)
                new_theta = self._wrap_angle(theta + dtheta)
                proposed_state[i, 0] = np.clip(new_x, self.x_min, self.x_max)
                proposed_state[i, 1] = np.clip(new_y, self.y_min, self.y_max)
                proposed_state[i, 2] = new_theta

        # Reject moves that would collide with other robots
        collision = not self.is_valid_state(proposed_state) or not self.is_valid_motion(proposed_state)
        if not collision:
            self.state = proposed_state

        success = self.is_goal(self.state)
        obs = self._get_obs()
        info = {"is_success": bool(success), "collision": bool(collision)}

        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            info["time_limit"] = True

        # Delegates reward logic to compute_reward for HER compatibility
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
            self.render()
        # print(f"Step: {self.step_count}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Return per-sample dense reward following the multi-goal API contract."""
        achieved = self._reshape_goals(achieved_goal)
        desired = self._reshape_goals(desired_goal)
        batch_size = achieved.shape[0]

        distances = np.linalg.norm(achieved - desired, axis=2).sum(axis=1)
        rewards = self.step_penalty - 0.05 * distances

        # Add proximity penalty for robots that are too close to each other
        proximity_penalties = self._compute_proximity_penalty(achieved)
        rewards += proximity_penalties
        
        # Add proximity penalty for robots that are too close to obstacles
        obstacle_proximity_penalties = self._compute_obstacle_proximity_penalty(achieved)
        rewards += obstacle_proximity_penalties

        success_mask = np.all(np.linalg.norm(achieved - desired, axis=2) <= self.goal_threshold, axis=1)
        rewards = np.where(success_mask, self.goal_reward, rewards)

        collision_mask = self._extract_flag_from_info(info, "collision", False, batch_size)
        rewards = np.where(collision_mask, self.collision_penalty, rewards)

        rewards = rewards.astype(np.float32)
        if rewards.size == 1:
            return float(rewards[0])
        return rewards

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Terminate when all robots reach goals or a collision occurs."""
        success_mask = self._success_mask(achieved_goal, desired_goal)
        batch_size = success_mask.shape[0]
        collision_mask = self._extract_flag_from_info(info, "collision", False, batch_size)
        terminated = np.logical_or(success_mask, collision_mask)
        return bool(terminated[0]) if terminated.size == 1 else terminated

    def compute_truncated(self, achieved_goal, desired_goal, info):
        """Episode truncation is controlled by the time_limit flag set in step."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        if achieved.ndim == 1:
            achieved = achieved[None, :]
        batch_size = achieved.shape[0]
        truncated = self._extract_flag_from_info(info, "time_limit", False, batch_size)
        return bool(truncated[0]) if truncated.size == 1 else truncated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Sample obstacles first, then robot positions and goals that avoid obstacles
        self.obstacles = self._sample_random_obstacles()  # Randomize obstacles first
        self.state = self._sample_random_state()  # Then sample robot positions avoiding obstacles
        self.goal_position = self._sample_random_goals()  # Then sample goals avoiding obstacles
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        observation = self._get_obs()
        info = {"is_success": False, "collision": False}
        return observation, info

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "human":
            plt.figure("CentralizedMultiRobotEnv")
            plt.clf()
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(1.0 / self.metadata["render_fps"])
        return frame if self.render_mode == "rgb_array" else None

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close("CentralizedMultiRobotEnv")


    def is_valid_state(self, state):
        """Check boundary, inter-robot collision, and robot-obstacle collision constraints."""
        for i in range(self.num_robots):
            x, y = state[i, :2]
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                return False
        
        # Check inter-robot collisions
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if np.linalg.norm(state[i, :2] - state[j, :2]) < 2 * self.robot_radius:
                    return False
        
        # Check robot-obstacle collisions
        for i in range(self.num_robots):
            for obs_pos in self.obstacles:
                if np.linalg.norm(state[i, :2] - obs_pos) < (self.robot_radius + self.obstacle_radius):
                    return False
        
        return True

    def is_valid_motion(self, proposed_state):
        """Reject straight-line motions that would intersect other robots or obstacles."""
        for i in range(self.num_robots):
            mid_x = 0.5 * (proposed_state[i, 0] + self.state[i, 0])
            mid_y = 0.5 * (proposed_state[i, 1] + self.state[i, 1])

        # Check inter-robot collisions
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                mid_x_i = 0.5 * (proposed_state[i, 0] + self.state[i, 0])
                mid_y_i = 0.5 * (proposed_state[i, 1] + self.state[i, 1])
                mid_x_j = 0.5 * (proposed_state[j, 0] + self.state[j, 0])
                mid_y_j = 0.5 * (proposed_state[j, 1] + self.state[j, 1])
                if np.linalg.norm([mid_x_i - mid_x_j, mid_y_i - mid_y_j]) < 2 * self.robot_radius:
                    return False
        
        # Check robot-obstacle collisions during motion
        for i in range(self.num_robots):
            mid_x_i = 0.5 * (proposed_state[i, 0] + self.state[i, 0])
            mid_y_i = 0.5 * (proposed_state[i, 1] + self.state[i, 1])
            for obs_pos in self.obstacles:
                if np.linalg.norm([mid_x_i - obs_pos[0], mid_y_i - obs_pos[1]]) < (self.robot_radius + self.obstacle_radius):
                    return False
        
        return True

    def is_goal(self, state):
        """Return True if all robots are within the goal threshold."""
        distances = np.linalg.norm(state[:, :2] - self.goal_position, axis=1)
        return bool(np.all(distances <= self.goal_threshold))

    def _sample_random_position(self):
        """Sample a random (x, y) position within workspace boundaries."""
        x = np.random.uniform(self.x_min, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max)
        return np.array([x, y], dtype=np.float32)

    def _is_position_valid(self, pos, existing_positions, check_obstacles=False):
        """
        Check if a position is valid (no collision with existing positions or obstacles).
        
        Args:
            pos: (x, y) position to check
            existing_positions: list of existing (x, y) positions to check against
            check_obstacles: if True, also check collision with obstacles
        """
        x, y = pos
        
        # Check boundaries
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False
    
        
        # Check collision with existing positions (other robots)
        for existing_pos in existing_positions:
            if np.linalg.norm(pos - existing_pos) < 2 * self.robot_radius:
                return False
        
        # Check collision with obstacles if requested
        if check_obstacles and self.num_obstacles > 0:
            for obs_pos in self.obstacles:
                if np.linalg.norm(pos - obs_pos) < (self.robot_radius + self.obstacle_radius):
                    return False
        
        return True

    def _sample_random_state(self, max_attempts=1000):
        """
        Sample random collision-free initial state for all robots.
        
        Returns:
            state: (num_robots, 3) array with [x, y, theta] for each robot
        """
        state = np.zeros((self.num_robots, 3), dtype=np.float32)
        positions = []
        
        for i in range(self.num_robots):
            valid_position_found = False
            for attempt in range(max_attempts):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, positions, check_obstacles=True):
                    positions.append(pos)
                    state[i, 0] = pos[0]
                    state[i, 1] = pos[1]
                    state[i, 2] = np.random.uniform(-np.pi, np.pi)  # Random heading
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                raise RuntimeError(
                    f"Failed to find valid initial position for robot {i} after {max_attempts} attempts"
                )
        
        return state

    def _sample_random_goals(self, max_attempts=1000):
        """
        Sample random collision-free goal positions for all robots.
        
        Returns:
            goals: (num_robots, 2) array with [x, y] goal for each robot
        """
        goals = np.zeros((self.num_robots, 2), dtype=np.float32)
        positions = []
        
        for i in range(self.num_robots):
            valid_position_found = False
            for attempt in range(max_attempts):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, positions, check_obstacles=True):
                    positions.append(pos)
                    goals[i] = pos
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                raise RuntimeError(
                    f"Failed to find valid goal position for robot {i} after {max_attempts} attempts"
                )
        
        return goals

    def _sample_random_obstacles(self, max_attempts=1000):
        """
        Sample random positions for obstacles that don't collide with other obstacles.
        This method should be called BEFORE sampling robot states and goals.
        
        Returns:
            obstacles: (num_obstacles, 2) array with [x, y] position for each obstacle
        """
        if self.num_obstacles == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        obstacles = np.zeros((self.num_obstacles, 2), dtype=np.float32)
        existing_obstacle_positions = []
        
        for i in range(self.num_obstacles):
            valid_position_found = False
            for attempt in range(max_attempts):
                pos = self._sample_random_position()
                # Only check that obstacle doesn't overlap with other obstacles
                # (Don't check robots/goals since they haven't been sampled yet)
                if self._is_position_valid(pos, existing_obstacle_positions, check_obstacles=False):
                    existing_obstacle_positions.append(pos)
                    obstacles[i] = pos
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                raise RuntimeError(
                    f"Failed to find valid position for obstacle {i} after {max_attempts} attempts"
                )
        
        return obstacles

    def distance_norm(self, state):
        """Aggregate distance metric used for shaping/diagnostics."""
        return float(np.linalg.norm(state[:, :2] - self.goal_position, axis=1).sum())

    def _compute_proximity_penalty(self, state):
        """
        Compute proximity penalty for robots that are closer than the threshold.
        The penalty increases as robots get closer, with maximum penalty at collision distance.
        
        Args:
            state: (num_robots, 3) or (batch_size, num_robots, 2) array with robot positions
        
        Returns:
            penalty: scalar or array of proximity penalties
        """
        # Handle both single state and batch of states
        if state.ndim == 2:
            # Single state: (num_robots, 3) or (num_robots, 2)
            positions = state[:, :2] if state.shape[1] >= 2 else state
            batch_mode = False
        else:
            # Batch of states: (batch_size, num_robots, 2)
            positions = state
            batch_mode = True
        
        if batch_mode:
            batch_size = positions.shape[0]
            penalties = np.zeros(batch_size, dtype=np.float32)
            
            for b in range(batch_size):
                penalty = 0.0
                for i in range(self.num_robots):
                    for j in range(i + 1, self.num_robots):
                        dist = np.linalg.norm(positions[b, i] - positions[b, j])
                        if dist < self.proximity_threshold:
                            # Linear penalty that increases as distance decreases
                            # At collision distance (2 * robot_radius), penalty is max_proximity_penalty
                            # At proximity_threshold, penalty is 0
                            min_distance = 2 * self.robot_radius
                            if dist < min_distance:
                                dist = min_distance  # Clamp to avoid division issues
                            
                            # Scale penalty: 0 at threshold, max at min_distance
                            penalty_scale = (self.proximity_threshold - dist) / (self.proximity_threshold - min_distance)
                            penalty += self.max_proximity_penalty * penalty_scale
                
                penalties[b] = penalty
            
            return penalties
        else:
            # Single state
            penalty = 0.0
            for i in range(self.num_robots):
                for j in range(i + 1, self.num_robots):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.proximity_threshold:
                        min_distance = 2 * self.robot_radius
                        if dist < min_distance:
                            dist = min_distance
                        
                        penalty_scale = (self.proximity_threshold - dist) / (self.proximity_threshold - min_distance)
                        penalty += self.max_proximity_penalty * penalty_scale
            
            return penalty

    def _compute_obstacle_proximity_penalty(self, state):
        """
        Compute proximity penalty for robots that are too close to obstacles.
        The penalty increases as robots get closer to obstacles, with maximum penalty at collision distance.
        
        Args:
            state: (num_robots, 3) or (batch_size, num_robots, 2) array with robot positions
        
        Returns:
            penalty: scalar or array of obstacle proximity penalties
        """
        # If no obstacles, return zero penalty
        if self.num_obstacles == 0:
            if state.ndim == 2:
                return 0.0
            else:
                return np.zeros(state.shape[0], dtype=np.float32)
        
        # Handle both single state and batch of states
        if state.ndim == 2:
            # Single state: (num_robots, 3) or (num_robots, 2)
            positions = state[:, :2] if state.shape[1] >= 2 else state
            batch_mode = False
        else:
            # Batch of states: (batch_size, num_robots, 2)
            positions = state
            batch_mode = True
        
        # Minimum safe distance (robot radius + obstacle radius)
        min_distance = self.robot_radius + self.obstacle_radius
        
        if batch_mode:
            batch_size = positions.shape[0]
            penalties = np.zeros(batch_size, dtype=np.float32)
            
            for b in range(batch_size):
                penalty = 0.0
                for i in range(self.num_robots):
                    for obs_pos in self.obstacles:
                        dist = np.linalg.norm(positions[b, i] - obs_pos)
                        if dist < self.obstacle_proximity_threshold:
                            # Linear penalty that increases as distance decreases
                            # At collision distance (robot_radius + obstacle_radius), penalty is max
                            # At obstacle_proximity_threshold, penalty is 0
                            if dist < min_distance:
                                dist = min_distance  # Clamp to avoid division issues
                            
                            # Scale penalty: 0 at threshold, max at min_distance
                            penalty_scale = (self.obstacle_proximity_threshold - dist) / (self.obstacle_proximity_threshold - min_distance)
                            penalty += self.max_obstacle_proximity_penalty * penalty_scale
                
                penalties[b] = penalty
            
            return penalties
        else:
            # Single state
            penalty = 0.0
            for i in range(self.num_robots):
                for obs_pos in self.obstacles:
                    dist = np.linalg.norm(positions[i] - obs_pos)
                    if dist < self.obstacle_proximity_threshold:
                        if dist < min_distance:
                            dist = min_distance
                        
                        penalty_scale = (self.obstacle_proximity_threshold - dist) / (self.obstacle_proximity_threshold - min_distance)
                        penalty += self.max_obstacle_proximity_penalty * penalty_scale
            
            return penalty

    def _compute_all_robot_info(self):
        """
        For each robot, compute the distance and polar angle to ALL other robots.
        
        Returns:
            all_robot_info: (num_robots, num_robots-1, 3) array with [distance, cos(bearing), sin(bearing)]
                           for each robot pair
        """
        all_robot_info = np.zeros((self.num_robots, self.num_robots - 1, 3), dtype=np.float32)
        
        for i in range(self.num_robots):
            robot_pos = self.state[i, :2]
            other_idx = 0
            
            # Compute info for all other robots
            for j in range(self.num_robots):
                if i == j:
                    continue
                
                neighbor_pos = self.state[j, :2]
                # Calculate relative position vector from robot i to robot j
                rel_vec = neighbor_pos - robot_pos
                distance = np.linalg.norm(rel_vec)
                
                # Compute polar angle (bearing) of the relative vector
                bearing = np.arctan2(rel_vec[1], rel_vec[0])
                
                # Store distance and bearing (as cos and sin)
                all_robot_info[i, other_idx, 0] = distance
                all_robot_info[i, other_idx, 1] = np.cos(bearing)
                all_robot_info[i, other_idx, 2] = np.sin(bearing)
                
                other_idx += 1
        
        return all_robot_info
    
    def _compute_all_obstacle_info(self):
        """
        For each robot, compute the distance and polar angle to ALL obstacles.
        
        Returns:
            all_obstacle_info: (num_robots, num_obstacles, 3) array with [distance, cos(bearing), sin(bearing)]
                              for each robot-obstacle pair
        """
        if self.num_obstacles == 0:
            return np.zeros((self.num_robots, 0, 3), dtype=np.float32)
        
        all_obstacle_info = np.zeros((self.num_robots, self.num_obstacles, 3), dtype=np.float32)
        
        for i in range(self.num_robots):
            robot_pos = self.state[i, :2]
            
            # Compute info for all obstacles
            for j in range(self.num_obstacles):
                obstacle_pos = self.obstacles[j]
                # Calculate relative position vector from robot i to obstacle j
                rel_vec = obstacle_pos - robot_pos
                distance = np.linalg.norm(rel_vec)
                
                # Compute polar angle (bearing) of the relative vector
                bearing = np.arctan2(rel_vec[1], rel_vec[0])
                
                # Store distance and bearing (as cos and sin)
                all_obstacle_info[i, j, 0] = distance
                all_obstacle_info[i, j, 1] = np.cos(bearing)
                all_obstacle_info[i, j, 2] = np.sin(bearing)
        
        return all_obstacle_info

    def _get_obs(self):
        """
        Constructs observation dictionary with absolute positions.
        
        New observation structure per robot:
        - robot_x, robot_y (absolute position)
        - cos(robot_theta), sin(robot_theta)
        - For each OTHER robot: distance, cos(bearing), sin(bearing)
        - For each OBSTACLE: distance, cos(bearing), sin(bearing)
        
        Plus absolute goal positions at the end.
        """
        obs = []
        
        # Compute info for all robot pairs
        all_robot_info = self._compute_all_robot_info()
        
        # Compute info for all robot-obstacle pairs
        all_obstacle_info = self._compute_all_obstacle_info()
        
        # Per robot observations
        for robot_idx in range(self.num_robots):
            robot_x, robot_y, robot_theta = self.state[robot_idx]
            
            # Absolute position (x, y)
            obs.extend([robot_x, robot_y])
            
            # Robot orientation as cos(theta) and sin(theta)
            obs.extend([np.cos(robot_theta), np.sin(robot_theta)])
            
            # Information about all other robots
            for other_idx in range(self.num_robots - 1):
                obs.extend([
                    all_robot_info[robot_idx, other_idx, 0],  # distance
                    all_robot_info[robot_idx, other_idx, 1],  # cos(bearing)
                    all_robot_info[robot_idx, other_idx, 2]   # sin(bearing)
                ])
            
            # Information about all obstacles
            for obstacle_idx in range(self.num_obstacles):
                obs.extend([
                    all_obstacle_info[robot_idx, obstacle_idx, 0],  # distance
                    all_obstacle_info[robot_idx, obstacle_idx, 1],  # cos(bearing)
                    all_obstacle_info[robot_idx, obstacle_idx, 2]   # sin(bearing)
                ])
        
        # Goals as absolute positions
        for goal_idx in range(self.num_robots):
            goal_x, goal_y = self.goal_position[goal_idx]
            obs.extend([goal_x, goal_y])
        
        return {
            "observation": np.array(obs, dtype=np.float32).copy(),
            "achieved_goal": self.state[:, :2].flatten().astype(np.float32).copy(),
            "desired_goal": self.goal_position.flatten().astype(np.float32).copy(),
        }

    def _reshape_goals(self, goals):
        goals = np.asarray(goals, dtype=np.float32)
        if goals.ndim == 1:
            goals = goals[None, :]
        return goals.reshape(goals.shape[0], self.num_robots, 2)

    def _success_mask(self, achieved_goal, desired_goal):
        # Evaluate success for each sample in a batch
        achieved = self._reshape_goals(achieved_goal)
        desired = self._reshape_goals(desired_goal)
        within_threshold = np.linalg.norm(achieved - desired, axis=2) <= self.goal_threshold
        return np.all(within_threshold, axis=1)

    @staticmethod
    def _wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _extract_flag_from_info(info, key, default, batch_size):
        if info is None:
            return np.full(batch_size, default, dtype=bool)
        if isinstance(info, dict):
            return np.full(batch_size, info.get(key, default), dtype=bool)
        return np.array([item.get(key, default) for item in info], dtype=bool, copy=False)

    def _render_frame(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(self.x_min - self.robot_radius, self.x_max + self.robot_radius)
        ax.set_ylim(self.y_min - self.robot_radius, self.y_max + self.robot_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Draw obstacles
        for obs_x, obs_y in self.obstacles:
            ax.add_patch(Circle((obs_x, obs_y), self.obstacle_radius, color="gray", alpha=0.7, zorder=1))

        # Draw robots and goals
        for idx, (x, y, theta) in enumerate(self.state):
            ax.add_patch(Circle((x, y), self.robot_radius, color="blue", alpha=0.8, zorder=2))
            dx = 0.5 * np.cos(theta)
            dy = 0.5 * np.sin(theta)
            ax.add_patch(Arrow(x, y, dx, dy, width=0.1, color="red", zorder=3))
            goal_x, goal_y = self.goal_position[idx]
            ax.add_patch(Circle((goal_x, goal_y), 0.15, color="green", alpha=0.9, zorder=2))

        fig.canvas.draw()
        # Use buffer_rgba() which is compatible with all backends including macOS
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB by dropping the alpha channel
        frame = frame[:, :, :3]
        plt.close(fig)
        return frame
