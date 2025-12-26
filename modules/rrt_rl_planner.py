"""
RRT-RL Hierarchical Planner

Combines RRT global planning with RL local policies:
1. RRT plans waypoints in the 6-robot configuration space
2. RL policies execute local motions between waypoints
3. PolicyManager selects appropriate policy based on robot subset

Usage:
    from env.Env_parent_6robots import AbstractSixRobotEnv
    from modules.policy_manager import PolicyManager
    from modules.rrt_rl_planner import RRTRLPlanner
    
    # Setup
    env = AbstractSixRobotEnv(obstacle_config_path="configs/obstacles_default.yaml")
    policy_manager = PolicyManager("policies/")
    policy_manager.load_all_policies()
    
    # Plan and execute
    planner = RRTRLPlanner(env, policy_manager)
    success = planner.plan_and_execute(start_state, goal_state)
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from modules.rrt_planner import RRTPlanner, RRTPlannerConfig, RRTNode
from modules.policy_manager import PolicyManager, LoadedPolicy


class ExecutionStatus(Enum):
    """Status of local policy execution."""
    SUCCESS = "success"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    STUCK = "stuck"


@dataclass
class WaypointResult:
    """Result of executing motion to a waypoint."""
    status: ExecutionStatus
    final_state: np.ndarray
    steps_taken: int
    cumulative_reward: float


@dataclass
class RRTRLConfig:
    """Configuration for RRT-RL hierarchical planner."""
    # RRT config
    rrt_max_iterations: int = 5000
    rrt_step_size: float = 0.5
    rrt_goal_sample_rate: float = 0.1
    rrt_goal_tolerance: float = 0.3
    
    # RL execution config
    max_steps_per_waypoint: int = 100
    waypoint_tolerance: float = 0.3
    stuck_threshold: int = 20  # Steps without progress
    progress_threshold: float = 0.01  # Min distance change to count as progress
    
    # Policy selection
    default_controlled_robots: List[int] = None  # If None, control all 6
    
    # Replanning
    replan_on_failure: bool = True
    max_replan_attempts: int = 3


class LocalPolicyExecutor:
    """
    Executes RL policies to move between waypoints.
    Wraps the child environment and policy interaction.
    """
    
    def __init__(
        self,
        parent_env,
        policy_manager: PolicyManager,
        config: RRTRLConfig
    ):
        self.parent_env = parent_env
        self.policy_manager = policy_manager
        self.config = config
        
        # Import here to avoid circular imports
        from env.Env_controlled_subset import ControlledSubsetEnv
        self.ControlledSubsetEnv = ControlledSubsetEnv
    
    def execute_to_waypoint(
        self,
        target_state: np.ndarray,
        controlled_indices: List[int],
        uncontrolled_velocities: Optional[np.ndarray] = None,
        render: bool = False
    ) -> WaypointResult:
        """
        Execute RL policy to reach target waypoint.
        
        Args:
            target_state: Target state (6, 3) for all robots
            controlled_indices: Which robots the policy controls
            uncontrolled_velocities: Optional velocities for uncontrolled robots
            render: Whether to render during execution
            
        Returns:
            WaypointResult with execution status
        """
        # Get policy for this robot subset
        try:
            policy = self.policy_manager.get_policy(controlled_indices)
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: No policy for robots {controlled_indices}: {e}")
            return WaypointResult(
                status=ExecutionStatus.STUCK,
                final_state=self.parent_env.state.copy(),
                steps_taken=0,
                cumulative_reward=0.0
            )
        
        # Create child environment for this subset
        child_env = self.ControlledSubsetEnv(
            self.parent_env,
            controlled_robot_indices=controlled_indices,
            uncontrolled_velocity=uncontrolled_velocities
        )
        
        # Set goal for controlled robots
        controlled_goals = target_state[controlled_indices, :2]
        child_env.set_goals_for_controlled(controlled_goals)
        
        # Execute policy
        steps = 0
        cumulative_reward = 0.0
        stuck_counter = 0
        prev_distance = self._compute_distance_to_target(
            self.parent_env.state, target_state, controlled_indices
        )
        
        obs, _ = child_env.reset(reset_state=False)  # Keep current state
        
        while steps < self.config.max_steps_per_waypoint:
            # Get action from policy
            action = policy.get_action(
                observation=obs['observation'],
                achieved_goal=obs['achieved_goal'],
                desired_goal=obs['desired_goal'],
                deterministic=True
            )
            
            # Step environment
            obs, reward, terminated, truncated, info = child_env.step(action)
            cumulative_reward += reward
            steps += 1
            
            if render:
                child_env.render()
            
            # Check collision
            if info.get('collision', False):
                return WaypointResult(
                    status=ExecutionStatus.COLLISION,
                    final_state=self.parent_env.state.copy(),
                    steps_taken=steps,
                    cumulative_reward=cumulative_reward
                )
            
            # Check if reached waypoint
            current_distance = self._compute_distance_to_target(
                self.parent_env.state, target_state, controlled_indices
            )
            
            if current_distance < self.config.waypoint_tolerance:
                return WaypointResult(
                    status=ExecutionStatus.SUCCESS,
                    final_state=self.parent_env.state.copy(),
                    steps_taken=steps,
                    cumulative_reward=cumulative_reward
                )
            
            # Check if stuck
            if abs(current_distance - prev_distance) < self.config.progress_threshold:
                stuck_counter += 1
                if stuck_counter >= self.config.stuck_threshold:
                    return WaypointResult(
                        status=ExecutionStatus.STUCK,
                        final_state=self.parent_env.state.copy(),
                        steps_taken=steps,
                        cumulative_reward=cumulative_reward
                    )
            else:
                stuck_counter = 0
            
            prev_distance = current_distance
            
            if terminated or truncated:
                break
        
        # Timeout
        return WaypointResult(
            status=ExecutionStatus.TIMEOUT,
            final_state=self.parent_env.state.copy(),
            steps_taken=steps,
            cumulative_reward=cumulative_reward
        )
    
    def _compute_distance_to_target(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        controlled_indices: List[int]
    ) -> float:
        """Compute distance to target for controlled robots."""
        current_pos = current_state[controlled_indices, :2]
        target_pos = target_state[controlled_indices, :2]
        return np.sqrt(np.sum((current_pos - target_pos)**2))


class RRTRLPlanner:
    """
    Hierarchical RRT-RL planner.
    
    Uses RRT to find global path through configuration space,
    then executes RL policies to traverse between waypoints.
    """
    
    def __init__(
        self,
        parent_env,
        policy_manager: PolicyManager,
        config: Optional[RRTRLConfig] = None
    ):
        """
        Initialize RRT-RL planner.
        
        Args:
            parent_env: AbstractSixRobotEnv instance
            policy_manager: PolicyManager with loaded policies
            config: Planner configuration
        """
        self.parent_env = parent_env
        self.policy_manager = policy_manager
        self.config = config or RRTRLConfig()
        
        # Default to controlling all robots
        if self.config.default_controlled_robots is None:
            self.config.default_controlled_robots = list(range(6))
        
        # Setup RRT planner
        rrt_config = RRTPlannerConfig(
            max_iterations=self.config.rrt_max_iterations,
            step_size=self.config.rrt_step_size,
            goal_sample_rate=self.config.rrt_goal_sample_rate,
            goal_tolerance=self.config.rrt_goal_tolerance
        )
        self.rrt_planner = RRTPlanner(parent_env, config=rrt_config)
        
        # Setup local executor
        self.local_executor = LocalPolicyExecutor(
            parent_env, policy_manager, self.config
        )
        
        # Execution history
        self.waypoints: List[np.ndarray] = []
        self.execution_history: List[WaypointResult] = []
    
    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        verbose: bool = False
    ) -> Optional[List[np.ndarray]]:
        """
        Plan global path using RRT.
        
        Args:
            start_state: Starting state (6, 3)
            goal_state: Goal state (6, 3)
            verbose: Print progress
            
        Returns:
            List of waypoint states, or None if planning failed
        """
        path = self.rrt_planner.plan(start_state, goal_state, verbose=verbose)
        
        if path is not None:
            # Optionally smooth path
            path = self.rrt_planner.smooth_path(path, iterations=30)
            self.waypoints = path
        
        return path
    
    def execute(
        self,
        waypoints: Optional[List[np.ndarray]] = None,
        controlled_indices: Optional[List[int]] = None,
        render: bool = False,
        verbose: bool = False
    ) -> Tuple[bool, List[WaypointResult]]:
        """
        Execute along waypoints using RL policies.
        
        Args:
            waypoints: List of waypoint states (uses self.waypoints if None)
            controlled_indices: Which robots to control (uses config default if None)
            render: Whether to render
            verbose: Print progress
            
        Returns:
            (success, list of WaypointResults)
        """
        waypoints = waypoints or self.waypoints
        controlled_indices = controlled_indices or self.config.default_controlled_robots
        
        if not waypoints:
            return False, []
        
        self.execution_history = []
        
        for i, target_waypoint in enumerate(waypoints[1:], start=1):
            if verbose:
                print(f"Executing to waypoint {i}/{len(waypoints)-1}")
            
            result = self.local_executor.execute_to_waypoint(
                target_state=target_waypoint,
                controlled_indices=controlled_indices,
                render=render
            )
            
            self.execution_history.append(result)
            
            if verbose:
                print(f"  Status: {result.status.value}, steps: {result.steps_taken}")
            
            if result.status != ExecutionStatus.SUCCESS:
                return False, self.execution_history
        
        return True, self.execution_history
    
    def plan_and_execute(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        controlled_indices: Optional[List[int]] = None,
        render: bool = False,
        verbose: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Full pipeline: plan with RRT, execute with RL.
        
        Args:
            start_state: Starting state (6, 3)
            goal_state: Goal state (6, 3)
            controlled_indices: Which robots to control
            render: Whether to render
            verbose: Print progress
            
        Returns:
            (success, info_dict)
        """
        # Set initial state
        self.parent_env.state = start_state.copy()
        
        info = {
            'planning_success': False,
            'execution_success': False,
            'waypoints': [],
            'replan_count': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'final_state': None
        }
        
        for attempt in range(self.config.max_replan_attempts + 1):
            if verbose:
                print(f"\nPlanning attempt {attempt + 1}")
            
            # Plan from current state
            current_state = self.parent_env.state.copy()
            waypoints = self.plan(current_state, goal_state, verbose=verbose)
            
            if waypoints is None:
                if verbose:
                    print("RRT planning failed")
                continue
            
            info['planning_success'] = True
            info['waypoints'] = waypoints
            
            # Execute
            success, results = self.execute(
                waypoints=waypoints,
                controlled_indices=controlled_indices,
                render=render,
                verbose=verbose
            )
            
            info['total_steps'] += sum(r.steps_taken for r in results)
            info['total_reward'] += sum(r.cumulative_reward for r in results)
            
            if success:
                info['execution_success'] = True
                info['final_state'] = self.parent_env.state.copy()
                return True, info
            
            # Replan if enabled
            if not self.config.replan_on_failure:
                break
            
            info['replan_count'] += 1
        
        info['final_state'] = self.parent_env.state.copy()
        return False, info
    
    def get_execution_stats(self) -> Dict:
        """Get statistics about last execution."""
        if not self.execution_history:
            return {}
        
        return {
            'num_waypoints': len(self.waypoints),
            'waypoints_reached': sum(
                1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS
            ),
            'total_steps': sum(r.steps_taken for r in self.execution_history),
            'total_reward': sum(r.cumulative_reward for r in self.execution_history),
            'failure_reasons': [
                r.status.value for r in self.execution_history 
                if r.status != ExecutionStatus.SUCCESS
            ]
        }


class AdaptiveRRTRLPlanner(RRTRLPlanner):
    """
    Adaptive RRT-RL planner that can switch between different robot subsets.
    
    Useful when different policies are available for different robot groupings,
    and you want to decompose the problem based on available policies.
    """
    
    def __init__(
        self,
        parent_env,
        policy_manager: PolicyManager,
        config: Optional[RRTRLConfig] = None
    ):
        super().__init__(parent_env, policy_manager, config)
        self.available_subsets = policy_manager.list_loaded_policies()
    
    def select_robot_subset(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray
    ) -> List[int]:
        """
        Select which robots to control for this motion segment.
        
        Override this method for custom selection strategies.
        
        Args:
            current_state: Current state (6, 3)
            target_state: Target state (6, 3)
            
        Returns:
            List of robot indices to control
        """
        # Default: find robots that need to move most
        distances = np.sqrt(np.sum(
            (current_state[:, :2] - target_state[:, :2])**2, axis=1
        ))
        
        # Try to find a matching policy
        for k in [6, 4, 3, 2, 1]:  # Prefer controlling more robots
            # Get k robots with largest distances
            top_k = np.argsort(distances)[-k:].tolist()
            top_k_key = tuple(sorted(top_k))
            
            if top_k_key in self.available_subsets:
                return list(top_k_key)
        
        # Fallback to default
        return self.config.default_controlled_robots
    
    def execute_adaptive(
        self,
        waypoints: Optional[List[np.ndarray]] = None,
        render: bool = False,
        verbose: bool = False
    ) -> Tuple[bool, List[WaypointResult]]:
        """
        Execute with adaptive robot subset selection per waypoint.
        """
        waypoints = waypoints or self.waypoints
        
        if not waypoints:
            return False, []
        
        self.execution_history = []
        
        for i, target_waypoint in enumerate(waypoints[1:], start=1):
            current_state = self.parent_env.state.copy()
            
            # Adaptively select robots
            controlled_indices = self.select_robot_subset(current_state, target_waypoint)
            
            if verbose:
                print(f"Waypoint {i}: controlling robots {controlled_indices}")
            
            result = self.local_executor.execute_to_waypoint(
                target_state=target_waypoint,
                controlled_indices=controlled_indices,
                render=render
            )
            
            self.execution_history.append(result)
            
            if result.status != ExecutionStatus.SUCCESS:
                return False, self.execution_history
        
        return True, self.execution_history


# Utility functions

def decompose_to_subproblems(
    goal_state: np.ndarray,
    current_state: np.ndarray,
    max_robots_per_group: int = 2
) -> List[List[int]]:
    """
    Decompose multi-robot planning into subproblems.
    
    Groups robots that need similar motions together.
    
    Args:
        goal_state: Goal state (6, 3)
        current_state: Current state (6, 3)
        max_robots_per_group: Maximum robots per group
        
    Returns:
        List of robot index lists for each subproblem
    """
    # Compute motion vectors
    motions = goal_state[:, :2] - current_state[:, :2]
    
    # Simple clustering by motion direction
    angles = np.arctan2(motions[:, 1], motions[:, 0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    # Group
    groups = []
    current_group = []
    
    for idx in sorted_indices:
        current_group.append(int(idx))
        if len(current_group) >= max_robots_per_group:
            groups.append(current_group)
            current_group = []
    
    if current_group:
        groups.append(current_group)
    
    return groups


def sequential_subset_execution(
    planner: RRTRLPlanner,
    goal_state: np.ndarray,
    robot_groups: List[List[int]],
    render: bool = False,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Execute motion for robot groups sequentially.
    
    Each group moves while others stay stationary.
    
    Args:
        planner: RRTRLPlanner instance
        goal_state: Final goal state
        robot_groups: List of robot index groups
        render: Whether to render
        verbose: Print progress
        
    Returns:
        (success, info_dict)
    """
    info = {
        'groups_completed': 0,
        'total_steps': 0,
        'group_results': []
    }
    
    for group_idx, robots in enumerate(robot_groups):
        if verbose:
            print(f"\nExecuting group {group_idx + 1}/{len(robot_groups)}: robots {robots}")
        
        # Create intermediate goal where only this group moves
        intermediate_goal = planner.parent_env.state.copy()
        for robot_idx in robots:
            intermediate_goal[robot_idx] = goal_state[robot_idx]
        
        # Plan and execute for this group
        success, group_info = planner.plan_and_execute(
            start_state=planner.parent_env.state.copy(),
            goal_state=intermediate_goal,
            controlled_indices=robots,
            render=render,
            verbose=verbose
        )
        
        info['group_results'].append({
            'robots': robots,
            'success': success,
            'steps': group_info.get('total_steps', 0)
        })
        info['total_steps'] += group_info.get('total_steps', 0)
        
        if success:
            info['groups_completed'] += 1
        else:
            return False, info
    
    return True, info
