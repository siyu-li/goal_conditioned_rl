"""
Parent environment with fixed 6 robots and obstacle map.
Used for RRT global planning and state synchronization.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Rectangle
import yaml


class AbstractSixRobotEnv(gym.Env):
    """
    Parent environment with fixed 6 robots and static obstacle map.
    State per robot: [x, y, theta]
    Last action stored separately as [dx, dy, dtheta] for reference.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, obstacle_config_path=None, robot_radius=0.5, render_mode=None):
        super().__init__()
        self.num_robots = 6  # Fixed
        self.robot_radius = float(robot_radius)
        self.render_mode = render_mode

        # Workspace boundaries
        self.x_min, self.x_max = 0.0, 10.0
        self.y_min, self.y_max = 0.0, 10.0
        self.max_step_size = 0.5

        # Load obstacles from config
        self.obstacles = []
        if obstacle_config_path:
            self._load_obstacle_map(obstacle_config_path)

        # State: [x, y, theta] for each robot
        self.state = np.zeros((6, 3), dtype=np.float32)
        
        # Last action: [dx, dy, dtheta] for each robot (for reference/visualization only)
        self.last_action = np.zeros((6, 3), dtype=np.float32)

        # Goal positions for all 6 robots
        self.goal_position = np.zeros((6, 2), dtype=np.float32)

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
                    elif obs['type'] == 'rectangle':
                        self.obstacles.append({
                            'type': 'rectangle',
                            'x': obs['x'],
                            'y': obs['y'],
                            'width': obs['width'],
                            'height': obs['height'],
                            'rotation': obs.get('rotation', 0.0)
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

    def get_full_state(self):
        """Return complete 6-robot state for RRT planning."""
        return self.state.copy()

    def set_full_state(self, state):
        """Set state from RRT waypoint."""
        self.state = np.array(state, dtype=np.float32).reshape(6, 3)

    def get_last_action(self):
        """Return last action [dx, dy, dtheta] for each robot."""
        return self.last_action.copy()

    def set_last_action(self, last_action):
        """Set last action for all robots."""
        self.last_action = np.array(last_action, dtype=np.float32).reshape(6, 3)

    def get_goals(self):
        """Return goal positions for all 6 robots."""
        return self.goal_position.copy()

    def set_goals(self, goals):
        """Set goal positions for all 6 robots."""
        self.goal_position = np.array(goals, dtype=np.float32).reshape(6, 2)

    def check_collision(self, state=None):
        """
        Check collisions with obstacles and between robots.
        Returns True if collision exists.
        """
        if state is None:
            state = self.state
        state = np.array(state).reshape(6, 3)
        positions = state[:, :2]

        # Check boundary collisions
        for i in range(6):
            x, y = positions[i]
            if not (self.x_min + self.robot_radius <= x <= self.x_max - self.robot_radius and
                    self.y_min + self.robot_radius <= y <= self.y_max - self.robot_radius):
                return True

        # Check inter-robot collisions
        for i in range(6):
            for j in range(i + 1, 6):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2 * self.robot_radius:
                    return True

        # Check obstacle collisions
        for i in range(6):
            pos = positions[i]
            for obs in self.obstacles:
                if obs['type'] == 'circle':
                    dist = np.linalg.norm(pos - np.array([obs['x'], obs['y']]))
                    if dist < self.robot_radius + obs['radius']:
                        return True
                elif obs['type'] == 'rectangle':
                    # Simple AABB check (ignoring rotation for simplicity)
                    ox, oy = obs['x'], obs['y']
                    w, h = obs['width'], obs['height']
                    # Check if robot center is within expanded rectangle
                    if (ox - self.robot_radius <= pos[0] <= ox + w + self.robot_radius and
                        oy - self.robot_radius <= pos[1] <= oy + h + self.robot_radius):
                        return True

        return False

    def propagate_robot(self, robot_idx, dt=1.0):
        """
        Propagate a single robot using its last action.
        Used for moving obstacle prediction (if needed).
        """
        dx, dy, dtheta = self.last_action[robot_idx]
        self.state[robot_idx, 0] += dx * dt
        self.state[robot_idx, 1] += dy * dt
        self.state[robot_idx, 2] = self._wrap_angle(self.state[robot_idx, 2] + dtheta * dt)
        
        # Clip to workspace
        self.state[robot_idx, 0] = np.clip(self.state[robot_idx, 0], self.x_min, self.x_max)
        self.state[robot_idx, 1] = np.clip(self.state[robot_idx, 1], self.y_min, self.y_max)

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

    def reset(self, seed=None, options=None):
        """Reset to random collision-free state."""
        super().reset(seed=seed)
        
        # Sample collision-free initial positions
        positions = []
        for i in range(6):
            for _ in range(1000):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, positions):
                    positions.append(pos)
                    break
            else:
                raise RuntimeError(f"Could not find valid position for robot {i}")
        
        # Set state
        for i in range(6):
            self.state[i, 0] = positions[i][0]
            self.state[i, 1] = positions[i][1]
            self.state[i, 2] = np.random.uniform(-np.pi, np.pi)
        
        # Reset last actions
        self.last_action = np.zeros((6, 3), dtype=np.float32)
        
        # Sample goals
        goal_positions = []
        for i in range(6):
            for _ in range(1000):
                pos = self._sample_random_position()
                if self._is_position_valid(pos, goal_positions):
                    goal_positions.append(pos)
                    break
            else:
                raise RuntimeError(f"Could not find valid goal for robot {i}")
        
        self.goal_position = np.array(goal_positions, dtype=np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Return observation dict."""
        return {
            "state": self.state.copy(),
            "last_action": self.last_action.copy(),
            "goals": self.goal_position.copy(),
        }

    @staticmethod
    def _wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        """Render the environment."""
        frame = self._render_frame()
        if self.render_mode == "human":
            plt.figure("AbstractSixRobotEnv")
            plt.clf()
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(1.0 / self.metadata["render_fps"])
        return frame if self.render_mode == "rgb_array" else None

    def _render_frame(self):
        """Render frame with robots, goals, and obstacles."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.x_min - 0.5, self.x_max + 0.5)
        ax.set_ylim(self.y_min - 0.5, self.y_max + 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        # Draw obstacles
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                ax.add_patch(Circle((obs['x'], obs['y']), obs['radius'], 
                                   color='gray', alpha=0.7))
            elif obs['type'] == 'rectangle':
                ax.add_patch(Rectangle((obs['x'], obs['y']), obs['width'], obs['height'],
                                       color='gray', alpha=0.7))

        # Draw robots and goals
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for idx in range(6):
            x, y, theta = self.state[idx]
            color = colors[idx % len(colors)]
            
            # Robot body
            ax.add_patch(Circle((x, y), self.robot_radius, color=color, alpha=0.7))
            
            # Robot heading arrow
            dx = 0.5 * np.cos(theta)
            dy = 0.5 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, fc='black', ec='black')
            
            # Goal
            goal_x, goal_y = self.goal_position[idx]
            ax.add_patch(Circle((goal_x, goal_y), 0.2, color=color, alpha=0.4))
            ax.plot(goal_x, goal_y, 'x', color=color, markersize=10, markeredgewidth=2)
            
            # Robot label
            ax.text(x, y, str(idx), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        ax.set_title(f"6-Robot Environment")
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return frame

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close("AbstractSixRobotEnv")
