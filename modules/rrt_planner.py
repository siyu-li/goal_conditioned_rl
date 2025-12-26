"""
RRT Planner for 6-robot configuration space.

Abstract implementation that can be extended for specific use cases.
Operates on the parent environment's full state space.

Usage:
    from env.Env_parent_6robots import AbstractSixRobotEnv
    
    env = AbstractSixRobotEnv(obstacle_config_path="configs/obstacles_default.yaml")
    planner = RRTPlanner(env, max_iter=5000, step_size=0.3)
    
    path = planner.plan(start_state, goal_state)
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RRTNode:
    """Node in RRT tree."""
    state: np.ndarray  # Full state: (6, 3) for [x, y, theta] per robot
    parent: Optional['RRTNode'] = None
    cost: float = 0.0  # Cost from start (for RRT*)
    children: List['RRTNode'] = field(default_factory=list)
    
    def __hash__(self):
        return id(self)


@dataclass
class RRTPlannerConfig:
    """Configuration for RRT planner."""
    max_iterations: int = 5000
    step_size: float = 0.3  # Max step per robot
    goal_sample_rate: float = 0.1  # Probability to sample goal
    goal_tolerance: float = 0.5  # Distance to consider goal reached
    collision_check_resolution: int = 5  # Interpolation points for collision
    rewire_radius: float = 1.0  # Radius for RRT* rewiring (0 to disable)


class RRTPlanner:
    """
    RRT planner for multi-robot configuration space.
    
    Configuration space: R^(6*3) = R^18 for 6 robots with [x, y, theta] each.
    """
    
    def __init__(
        self,
        env,  # AbstractSixRobotEnv
        config: Optional[RRTPlannerConfig] = None,
        collision_fn: Optional[Callable[[np.ndarray], bool]] = None
    ):
        """
        Initialize RRT planner.
        
        Args:
            env: Parent environment with collision checking
            config: Planner configuration
            collision_fn: Optional custom collision function
        """
        self.env = env
        self.config = config or RRTPlannerConfig()
        
        # Use environment's collision check or custom
        if collision_fn is not None:
            self._collision_fn = collision_fn
        else:
            self._collision_fn = self._env_collision_check
        
        # Workspace bounds from environment
        self.x_bounds = (env.x_min, env.x_max)
        self.y_bounds = (env.y_min, env.y_max)
        self.theta_bounds = (-np.pi, np.pi)
        
        # Tree storage
        self.nodes: List[RRTNode] = []
        self.start_node: Optional[RRTNode] = None
        self.goal_state: Optional[np.ndarray] = None
    
    def _env_collision_check(self, state: np.ndarray) -> bool:
        """Check collision using environment."""
        original_state = self.env.state.copy()
        self.env.state = state.reshape(6, 3)
        collision = self.env.check_collision()
        self.env.state = original_state
        return collision
    
    def _sample_random_state(self) -> np.ndarray:
        """Sample random state in configuration space."""
        state = np.zeros((6, 3), dtype=np.float32)
        for i in range(6):
            state[i, 0] = np.random.uniform(*self.x_bounds)
            state[i, 1] = np.random.uniform(*self.y_bounds)
            state[i, 2] = np.random.uniform(*self.theta_bounds)
        return state
    
    def _sample_state(self) -> np.ndarray:
        """Sample state with goal biasing."""
        if np.random.random() < self.config.goal_sample_rate and self.goal_state is not None:
            return self.goal_state.copy()
        return self._sample_random_state()
    
    def _distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute distance between two states.
        Uses weighted sum of position and angular distances.
        """
        s1 = state1.reshape(6, 3)
        s2 = state2.reshape(6, 3)
        
        # Position distance (Euclidean)
        pos_dist = np.sqrt(np.sum((s1[:, :2] - s2[:, :2])**2))
        
        # Angular distance (wrapped)
        theta_diff = s1[:, 2] - s2[:, 2]
        theta_diff = np.arctan2(np.sin(theta_diff), np.cos(theta_diff))
        ang_dist = np.sum(np.abs(theta_diff))
        
        return pos_dist + 0.1 * ang_dist  # Weight angular less
    
    def _nearest_node(self, state: np.ndarray) -> RRTNode:
        """Find nearest node in tree to given state."""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = self._distance(node.state, state)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_state: np.ndarray, to_state: np.ndarray) -> np.ndarray:
        """
        Steer from one state towards another, limited by step size.
        """
        diff = to_state - from_state
        dist = self._distance(from_state, to_state)
        
        if dist <= self.config.step_size:
            return to_state.copy()
        
        # Scale to step size
        ratio = self.config.step_size / dist
        new_state = from_state + ratio * diff
        
        # Wrap angles
        new_state = new_state.reshape(6, 3)
        new_state[:, 2] = np.arctan2(
            np.sin(new_state[:, 2]),
            np.cos(new_state[:, 2])
        )
        
        return new_state.flatten()
    
    def _collision_free_path(self, from_state: np.ndarray, to_state: np.ndarray) -> bool:
        """Check if path between two states is collision-free."""
        n_checks = self.config.collision_check_resolution
        
        for i in range(n_checks + 1):
            t = i / n_checks
            interp_state = (1 - t) * from_state + t * to_state
            
            # Wrap angles
            interp_state = interp_state.reshape(6, 3)
            interp_state[:, 2] = np.arctan2(
                np.sin(interp_state[:, 2]),
                np.cos(interp_state[:, 2])
            )
            
            if self._collision_fn(interp_state):
                return False
        
        return True
    
    def _near_nodes(self, state: np.ndarray, radius: float) -> List[RRTNode]:
        """Find all nodes within radius of state (for RRT*)."""
        near = []
        for node in self.nodes:
            if self._distance(node.state, state) <= radius:
                near.append(node)
        return near
    
    def _rewire(self, new_node: RRTNode, near_nodes: List[RRTNode]):
        """Rewire tree for RRT*."""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
            
            new_cost = new_node.cost + self._distance(new_node.state, near_node.state)
            
            if new_cost < near_node.cost:
                if self._collision_free_path(new_node.state, near_node.state):
                    # Rewire
                    if near_node.parent:
                        near_node.parent.children.remove(near_node)
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    new_node.children.append(near_node)
    
    def _extract_path(self, goal_node: RRTNode) -> List[np.ndarray]:
        """Extract path from start to goal node."""
        path = []
        node = goal_node
        while node is not None:
            path.append(node.state.reshape(6, 3))
            node = node.parent
        path.reverse()
        return path
    
    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        verbose: bool = False
    ) -> Optional[List[np.ndarray]]:
        """
        Plan path from start to goal.
        
        Args:
            start_state: Starting configuration (6, 3)
            goal_state: Goal configuration (6, 3)
            verbose: Print progress
            
        Returns:
            List of states from start to goal, or None if failed
        """
        start_state = start_state.flatten()
        goal_state = goal_state.flatten()
        self.goal_state = goal_state
        
        # Initialize tree with start
        self.start_node = RRTNode(state=start_state, cost=0.0)
        self.nodes = [self.start_node]
        
        # Check start/goal validity
        if self._collision_fn(start_state.reshape(6, 3)):
            if verbose:
                print("Start state in collision!")
            return None
        
        if self._collision_fn(goal_state.reshape(6, 3)):
            if verbose:
                print("Goal state in collision!")
            return None
        
        best_goal_node = None
        best_cost = float('inf')
        
        for iteration in range(self.config.max_iterations):
            # Sample
            random_state = self._sample_state()
            
            # Find nearest
            nearest_node = self._nearest_node(random_state)
            
            # Steer
            new_state = self._steer(nearest_node.state, random_state)
            
            # Check collision
            if not self._collision_free_path(nearest_node.state, new_state):
                continue
            
            # Compute cost
            new_cost = nearest_node.cost + self._distance(nearest_node.state, new_state)
            
            # RRT* - find best parent
            if self.config.rewire_radius > 0:
                near_nodes = self._near_nodes(new_state, self.config.rewire_radius)
                
                best_parent = nearest_node
                best_new_cost = new_cost
                
                for near_node in near_nodes:
                    candidate_cost = near_node.cost + self._distance(near_node.state, new_state)
                    if candidate_cost < best_new_cost:
                        if self._collision_free_path(near_node.state, new_state):
                            best_parent = near_node
                            best_new_cost = candidate_cost
                
                nearest_node = best_parent
                new_cost = best_new_cost
            
            # Add node
            new_node = RRTNode(state=new_state, parent=nearest_node, cost=new_cost)
            nearest_node.children.append(new_node)
            self.nodes.append(new_node)
            
            # RRT* - rewire
            if self.config.rewire_radius > 0:
                self._rewire(new_node, near_nodes)
            
            # Check goal
            dist_to_goal = self._distance(new_state, goal_state)
            if dist_to_goal <= self.config.goal_tolerance:
                if new_cost < best_cost:
                    best_goal_node = new_node
                    best_cost = new_cost
                    
                    if verbose:
                        print(f"Iteration {iteration}: Found path with cost {best_cost:.2f}")
                
                # Early termination for basic RRT
                if self.config.rewire_radius == 0:
                    break
            
            if verbose and iteration % 500 == 0:
                print(f"Iteration {iteration}, nodes: {len(self.nodes)}")
        
        if best_goal_node is None:
            if verbose:
                print(f"Failed to find path after {self.config.max_iterations} iterations")
            return None
        
        # Connect to exact goal if within tolerance
        final_node = RRTNode(
            state=goal_state,
            parent=best_goal_node,
            cost=best_cost + self._distance(best_goal_node.state, goal_state)
        )
        
        return self._extract_path(final_node)
    
    def smooth_path(
        self,
        path: List[np.ndarray],
        iterations: int = 50
    ) -> List[np.ndarray]:
        """
        Smooth path using shortcutting.
        
        Args:
            path: List of states
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        path = [s.flatten() for s in path]
        
        for _ in range(iterations):
            if len(path) <= 2:
                break
            
            # Random shortcut
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            
            if self._collision_free_path(path[i], path[j]):
                path = path[:i+1] + path[j:]
        
        return [s.reshape(6, 3) for s in path]
    
    def get_tree_stats(self) -> dict:
        """Return statistics about the RRT tree."""
        return {
            'num_nodes': len(self.nodes),
            'max_depth': self._compute_max_depth(),
        }
    
    def _compute_max_depth(self) -> int:
        """Compute maximum depth of tree."""
        max_depth = 0
        for node in self.nodes:
            depth = 0
            current = node
            while current.parent is not None:
                depth += 1
                current = current.parent
            max_depth = max(max_depth, depth)
        return max_depth


class BiRRTPlanner(RRTPlanner):
    """
    Bidirectional RRT planner.
    Grows trees from both start and goal, connects when they meet.
    """
    
    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        verbose: bool = False
    ) -> Optional[List[np.ndarray]]:
        """Plan using bidirectional RRT."""
        start_state = start_state.flatten()
        goal_state = goal_state.flatten()
        
        # Check validity
        if self._collision_fn(start_state.reshape(6, 3)):
            return None
        if self._collision_fn(goal_state.reshape(6, 3)):
            return None
        
        # Initialize both trees
        start_tree = [RRTNode(state=start_state, cost=0.0)]
        goal_tree = [RRTNode(state=goal_state, cost=0.0)]
        
        trees = [start_tree, goal_tree]
        tree_idx = 0  # Alternate between trees
        
        for iteration in range(self.config.max_iterations):
            current_tree = trees[tree_idx]
            other_tree = trees[1 - tree_idx]
            
            # Sample and extend current tree
            random_state = self._sample_random_state().flatten()
            nearest = min(current_tree, key=lambda n: self._distance(n.state, random_state))
            new_state = self._steer(nearest.state, random_state)
            
            if not self._collision_free_path(nearest.state, new_state):
                tree_idx = 1 - tree_idx
                continue
            
            new_node = RRTNode(state=new_state, parent=nearest)
            current_tree.append(new_node)
            
            # Try to connect to other tree
            other_nearest = min(other_tree, key=lambda n: self._distance(n.state, new_state))
            
            if self._distance(new_state, other_nearest.state) <= self.config.goal_tolerance:
                if self._collision_free_path(new_state, other_nearest.state):
                    if verbose:
                        print(f"Connected at iteration {iteration}")
                    
                    # Extract and merge paths
                    path_start = self._extract_path_from_tree(new_node if tree_idx == 0 else other_nearest)
                    path_goal = self._extract_path_from_tree(other_nearest if tree_idx == 0 else new_node)
                    path_goal.reverse()
                    
                    return path_start + path_goal[1:]
            
            tree_idx = 1 - tree_idx
            
            if verbose and iteration % 500 == 0:
                print(f"Iteration {iteration}, tree sizes: {len(start_tree)}, {len(goal_tree)}")
        
        return None
    
    def _extract_path_from_tree(self, node: RRTNode) -> List[np.ndarray]:
        """Extract path from tree root to node."""
        path = []
        current = node
        while current is not None:
            path.append(current.state.reshape(6, 3))
            current = current.parent
        path.reverse()
        return path
