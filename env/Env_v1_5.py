## Two robot individually controlled (moving forward and turning) environment with centralized state/action space
## and relative position observations for HER and goal-conditioned RL compatibility.
## Step reward is step_penalty - distance_to_goal_reward + collision_penalty + goal_reward

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow


class CentralizedTwoRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, max_episode_steps, group_id, robot_radius=0.5, render_mode=None):
        super().__init__()
        # Episode configuration
        self.max_episode_steps = max_episode_steps
        self.group_id = group_id
        self.robot_radius = float(robot_radius)
        self.render_mode = render_mode

        # Workspace boundaries for x/y coordinates
        self.x_min, self.x_max = 0.0, 20.0
        self.y_min, self.y_max = 0.0, 20.0

        # Each group selects a fixed pair of robots from the template setup
        self.robot_groups = {0: [0, 2], 1: [1, 5], 2: [3, 4]}
        if group_id not in self.robot_groups:
            raise ValueError(f"Unknown group_id {group_id}. Available groups: {list(self.robot_groups)}")
        self.robots = self.robot_groups[group_id]
        self.num_robots = len(self.robots)

        # Actions are distance and heading change per robot
        self.max_step_size = 1.0
        action_low = np.tile([0.0, -np.pi], self.num_robots).astype(np.float32)
        action_high = np.tile([self.max_step_size, np.pi], self.num_robots).astype(np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Static circular obstacles (x, y, radius)
        self.obstacles = [(5.0, 15.0, 1.5), (15.0, 5.0, 1.5)]
        self.num_obstacles = len(self.obstacles)

        # Observation space: for each obstacle, relative positions of all robots and goals
        # Per obstacle: (robot_rel_x, robot_rel_y, robot_theta) * num_robots + (goal_rel_x, goal_rel_y) * num_robots
        # obs_dim = num_obstacles * (3 * num_robots + 2 * num_robots)
        obs_dim_per_obstacle = 3 * self.num_robots + 2 * self.num_robots
        obs_dim = obs_dim_per_obstacle * self.num_obstacles
        
        # Observation bounds: relative positions can span the entire workspace
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

        self._initial_state_template = np.array(
            [
                [1.0, 2.0, 0.0],
                [1.0, 5.2, 0.0],
                [1.0, 8.4, 0.0],
                [1.0, 11.6, 0.0],
                [1.0, 14.8, 0.0],
                [1.0, 18.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._goal_template = np.array(
            [
                [18.0, 2.0],
                [18.0, 5.2],
                [18.0, 8.4],
                [18.0, 11.6],
                [18.0, 14.8],
                [18.0, 18.0],
            ],
            dtype=np.float32,
        )

        self.goal_threshold = 1.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.collision_penalty = -100.0

        self.state = self._initial_state_template[self.robots].copy()
        self.goal_position = self._goal_template[self.robots].copy()
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        if self.render_mode == "human":
            plt.ion()

    def _build_observation_bounds(self):
        """Build observation bounds for relative positions."""
        obs_low = []
        obs_high = []
        for _ in range(self.num_obstacles):
            for _ in range(self.num_robots):
                # Robot relative position (x, y, theta)
                obs_low.extend([-self.x_max, -self.y_max, -np.pi])
                obs_high.extend([self.x_max, self.y_max, np.pi])
            for _ in range(self.num_robots):
                # Goal relative position (x, y)
                obs_low.extend([-self.x_max, -self.y_max])
                obs_high.extend([self.x_max, self.y_max])
        return np.array(obs_low, dtype=np.float32), np.array(obs_high, dtype=np.float32)

    def step(self, action):
        # Enforce action bounds so policy errors cannot break simulation
        action = np.clip(action, self.action_space.low, self.action_space.high)
        proposed_state = self.state.copy()
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

        # Reject moves that would collide with obstacles or other robots
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

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Return per-sample dense reward following the multi-goal API contract."""
        achieved = self._reshape_goals(achieved_goal)
        desired = self._reshape_goals(desired_goal)
        batch_size = achieved.shape[0]

        distances = np.linalg.norm(achieved - desired, axis=2).sum(axis=1)
        rewards = self.step_penalty - 0.05 * distances

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
        # Randomly sample collision-free initial state and goal positions
        self.state = self._sample_random_state()
        self.goal_position = self._sample_random_goals()
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        observation = self._get_obs()
        info = {"is_success": False, "collision": False}
        return observation, info

    def render(self):
        frame = self._render_frame()
        if self.render_mode == "human":
            # Render on a persistent matplotlib window for interactive debugging
            plt.figure("CentralizedTwoRobotEnv")
            plt.clf()
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(1.0 / self.metadata["render_fps"])
        return frame if self.render_mode == "rgb_array" else None

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close("CentralizedTwoRobotEnv")

    def is_valid_state(self, state):
        """Check boundary, obstacle, and inter-robot collision constraints."""
        for i in range(self.num_robots):
            x, y = state[i, :2]
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                return False
            for ox, oy, radius in self.obstacles:
                if np.linalg.norm([x - ox, y - oy]) < radius + self.robot_radius:
                    return False
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if np.linalg.norm(state[i, :2] - state[j, :2]) < 2 * self.robot_radius:
                    return False
        return True

    def is_valid_motion(self, proposed_state):
        """Reject straight-line motions that would intersect obstacles or other robots."""
        for i in range(self.num_robots):
            mid_x = 0.5 * (proposed_state[i, 0] + self.state[i, 0])
            mid_y = 0.5 * (proposed_state[i, 1] + self.state[i, 1])
            for ox, oy, radius in self.obstacles:
                if np.linalg.norm([mid_x - ox, mid_y - oy]) < radius + self.robot_radius:
                    return False
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                mid_x_i = 0.5 * (proposed_state[i, 0] + self.state[i, 0])
                mid_y_i = 0.5 * (proposed_state[i, 1] + self.state[i, 1])
                mid_x_j = 0.5 * (proposed_state[j, 0] + self.state[j, 0])
                mid_y_j = 0.5 * (proposed_state[j, 1] + self.state[j, 1])
                if np.linalg.norm([mid_x_i - mid_x_j, mid_y_i - mid_y_j]) < 2 * self.robot_radius:
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

    def _is_position_valid(self, pos, existing_positions):
        """
        Check if a position is valid (no collision with obstacles or existing positions).
        
        Args:
            pos: (x, y) position to check
            existing_positions: list of existing (x, y) positions to check against
        """
        x, y = pos
        
        # Check boundaries
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
            return False
        
        # Check collision with obstacles
        for ox, oy, radius in self.obstacles:
            if np.linalg.norm([x - ox, y - oy]) < radius + self.robot_radius:
                return False
        
        # Check collision with existing positions (other robots)
        for existing_pos in existing_positions:
            if np.linalg.norm(pos - existing_pos) < 2 * self.robot_radius:
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
                if self._is_position_valid(pos, positions):
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
                if self._is_position_valid(pos, positions):
                    positions.append(pos)
                    goals[i] = pos
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                raise RuntimeError(
                    f"Failed to find valid goal position for robot {i} after {max_attempts} attempts"
                )
        
        return goals


    def distance_norm(self, state):
        """Aggregate distance metric used for shaping/diagnostics."""
        return float(np.linalg.norm(state[:, :2] - self.goal_position, axis=1).sum())

    def _get_obs(self):
        """
        Constructs observation dictionary with relative positions.
        Observation includes relative positions of robots and goals to obstacles.
        """
        obs = []
        for obs_x, obs_y, _ in self.obstacles:
            for robot_idx in range(self.num_robots):
                # Robot relative to this obstacle
                obs.extend([
                    self.state[robot_idx, 0] - obs_x,
                    self.state[robot_idx, 1] - obs_y,
                    self.state[robot_idx, 2]
                ])
            for robot_idx in range(self.num_robots):
                # Goal relative to this obstacle
                obs.extend([
                    self.goal_position[robot_idx, 0] - obs_x,
                    self.goal_position[robot_idx, 1] - obs_y
                ])
        
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

        for obs_x, obs_y, obs_radius in self.obstacles:
            ax.add_patch(Circle((obs_x, obs_y), obs_radius, color="black", alpha=0.5))

        for idx, (x, y, theta) in enumerate(self.state):
            ax.add_patch(Circle((x, y), self.robot_radius, color="blue", alpha=0.8))
            dx = 0.5 * np.cos(theta)
            dy = 0.5 * np.sin(theta)
            ax.add_patch(Arrow(x, y, dx, dy, width=0.1, color="red"))
            goal_x, goal_y = self.goal_position[idx]
            ax.add_patch(Circle((goal_x, goal_y), 0.15, color="green", alpha=0.9))

        fig.canvas.draw()
        # Use buffer_rgba() which is compatible with all backends including macOS
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB by dropping the alpha channel
        frame = frame[:, :, :3]
        plt.close(fig)
        return frame
