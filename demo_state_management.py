"""
Example: RRT-RL State Management Demonstration

This script demonstrates the correct way to manage states between
parent environment (RRT planning) and child environments (RL execution).
"""
import numpy as np
import gymnasium as gym
from env.Env_parent_6robots import AbstractSixRobotEnv
from env.Env_controlled_subset import ControlledSubsetEnv


def example_1_training_with_randomization():
    """
    Example 1: Training child policies with randomized uncontrolled robots.
    
    This ensures the policy learns to handle diverse obstacle configurations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Training with Randomized Uncontrolled Robots")
    print("="*70)
    
    # Create environment (or use registered version)
    env = gym.make('SymmetricMultiRobot-2-v1')
    
    print("\nTraining setup:")
    print(f"  - Controlling: 2 robots (randomly selected each episode)")
    print(f"  - Uncontrolled: 4 robots (randomized positions each episode)")
    print(f"  - Result: Policy learns to handle diverse obstacle scenarios\n")
    
    # Demonstrate 3 episodes
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  - New random positions for all 6 robots")
        print(f"  - New random goals for controlled robots")
        print(f"  - Observation shape: {obs['observation'].shape}")
        
        # Simulate a few steps
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    print("\n✓ Training randomization ensures policy robustness")


def example_2_rrt_rl_state_sync():
    """
    Example 2: Proper state synchronization for RRT-RL execution.
    
    Shows how to load RRT waypoints into child env and maintain state alignment.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: RRT-RL State Synchronization")
    print("="*70)
    
    # Setup parent and child environments
    parent_env = AbstractSixRobotEnv(
        obstacle_config_path='configs/obstacles_default.yaml',
        robot_radius=0.5
    )
    
    child_env = ControlledSubsetEnv(
        parent_env=parent_env,
        controlled_robot_indices=[0, 1, 2],  # Control first 3 robots
        max_episode_steps=50
    )
    
    print("\nSetup:")
    print(f"  - Parent env: 6 robots")
    print(f"  - Child env: Controls robots [0, 1, 2]")
    print(f"  - Uncontrolled: robots [3, 4, 5] (treated as obstacles)")
    
    # Simulate RRT planning
    print("\n1. RRT Planning Phase:")
    parent_env.reset(seed=42)
    initial_state = parent_env.get_full_state()
    print(f"   Initial state shape: {initial_state.shape}")
    print(f"   Initial positions:\n{initial_state[:, :2]}")
    
    # Simulate RRT returning a waypoint
    waypoint_state = initial_state.copy()
    waypoint_state[[0, 1, 2], :2] += np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
    
    print("\n2. Load RRT Waypoint into Child Env:")
    local_goals = waypoint_state[[0, 1, 2], :2]  # Goals for controlled robots
    
    obs, info = child_env.load_state_from_parent_and_reset(
        parent_state=waypoint_state,
        goals=local_goals,
        randomize_uncontrolled=False  # Keep uncontrolled from RRT
    )
    
    print(f"   Loaded waypoint into child env")
    print(f"   Local goals for robots [0,1,2]: {local_goals}")
    print(f"   Observation ready for policy")
    
    # Simulate policy execution
    print("\n3. Execute Policy:")
    for step in range(5):
        action = child_env.action_space.sample()  # Would be policy.predict(obs)
        obs, reward, terminated, truncated, info = child_env.step(action)
        
        if step == 0:
            print(f"   Step {step}: Taking actions for controlled robots")
    
    # Get final state and update parent
    print("\n4. Sync Final State Back to Parent:")
    final_state = child_env.get_current_parent_state()
    parent_env.set_full_state(final_state)
    
    print(f"   Final state synced to parent")
    print(f"   Parent and child now aligned")
    
    # Verify synchronization
    parent_state_check = parent_env.get_full_state()
    assert np.allclose(final_state, parent_state_check), "States not synced!"
    
    print("\n✓ State synchronization successful")


def example_3_switching_controlled_robots():
    """
    Example 3: Switching which robots are controlled during execution.
    
    Shows how to dynamically change controlled indices (e.g., for adaptive planning).
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Dynamic Controlled Robot Switching")
    print("="*70)
    
    # Setup
    parent_env = AbstractSixRobotEnv(
        obstacle_config_path='configs/obstacles_default.yaml',
        robot_radius=0.5
    )
    
    parent_env.reset(seed=123)
    current_state = parent_env.get_full_state()
    
    # Scenario 1: Control robots [0, 1]
    print("\nScenario 1: Control robots [0, 1]")
    child_env_1 = ControlledSubsetEnv(
        parent_env=parent_env,
        controlled_robot_indices=[0, 1]
    )
    
    goals_1 = current_state[[0, 1], :2] + np.array([[1.0, 0.0], [0.0, 1.0]])
    obs, info = child_env_1.load_state_from_parent_and_reset(
        parent_state=current_state,
        goals=goals_1,
        randomize_uncontrolled=False
    )
    
    print(f"  Controlling: {child_env_1.controlled_indices}")
    print(f"  Uncontrolled: {child_env_1.uncontrolled_indices}")
    
    # Execute a few steps
    for _ in range(3):
        action = child_env_1.action_space.sample()
        obs, reward, terminated, truncated, info = child_env_1.step(action)
    
    # Get updated state
    current_state = child_env_1.get_current_parent_state()
    parent_env.set_full_state(current_state)
    
    # Scenario 2: Switch to control robots [2, 3, 4]
    print("\nScenario 2: Switch to control robots [2, 3, 4]")
    child_env_2 = ControlledSubsetEnv(
        parent_env=parent_env,
        controlled_robot_indices=[2, 3, 4]
    )
    
    goals_2 = current_state[[2, 3, 4], :2] + np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    obs, info = child_env_2.load_state_from_parent_and_reset(
        parent_state=current_state,  # Use updated state from scenario 1
        goals=goals_2,
        randomize_uncontrolled=False
    )
    
    print(f"  Controlling: {child_env_2.controlled_indices}")
    print(f"  Uncontrolled: {child_env_2.uncontrolled_indices}")
    print(f"  Note: Robots [0, 1] now appear as obstacles in observation")
    
    # Execute a few steps
    for _ in range(3):
        action = child_env_2.action_space.sample()
        obs, reward, terminated, truncated, info = child_env_2.step(action)
    
    current_state = child_env_2.get_current_parent_state()
    
    print("\n✓ Successfully switched controlled robots while maintaining state")


def example_4_randomization_modes():
    """
    Example 4: Different randomization modes for different use cases.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Randomization Modes")
    print("="*70)
    
    parent_env = AbstractSixRobotEnv(
        obstacle_config_path='configs/obstacles_default.yaml',
        robot_radius=0.5
    )
    
    child_env = ControlledSubsetEnv(
        parent_env=parent_env,
        controlled_robot_indices=[0, 1]
    )
    
    # Mode 1: Full randomization (training)
    print("\nMode 1: Training - Full randomization")
    obs, info = child_env.reset()
    print("  - All 6 robots: random positions")
    print("  - Uncontrolled [2,3,4,5]: randomized")
    print("  - Use case: Training robust policies")
    
    # Mode 2: RRT execution (no randomization)
    print("\nMode 2: RRT-RL - Use RRT state as-is")
    parent_env.reset(seed=42)
    rrt_state = parent_env.get_full_state()
    goals = rrt_state[[0, 1], :2] + 1.0
    
    obs, info = child_env.load_state_from_parent_and_reset(
        parent_state=rrt_state,
        goals=goals,
        randomize_uncontrolled=False  # Keep RRT positions
    )
    print("  - Initial state: from RRT")
    print("  - Uncontrolled: fixed (from RRT)")
    print("  - Use case: Executing planned trajectories")
    
    # Mode 3: Partial randomization (testing robustness)
    print("\nMode 3: Testing - Randomize obstacles around RRT state")
    obs, info = child_env.load_state_from_parent_and_reset(
        parent_state=rrt_state,
        goals=goals,
        randomize_uncontrolled=True  # Randomize obstacles
    )
    print("  - Controlled: from RRT")
    print("  - Uncontrolled: randomized")
    print("  - Use case: Testing policy robustness to obstacle variations")
    
    print("\n✓ Flexible randomization for different scenarios")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("RRT-RL STATE MANAGEMENT EXAMPLES")
    print("="*70)
    
    # Run all examples
    example_1_training_with_randomization()
    example_2_rrt_rl_state_sync()
    example_3_switching_controlled_robots()
    example_4_randomization_modes()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Training: Randomize uncontrolled robots for diverse scenarios")
    print("2. RRT-RL: Use load_state_from_parent_and_reset() for clean state sync")
    print("3. Switching: Can dynamically change which robots are controlled")
    print("4. Flexibility: Different randomization modes for different needs")
    print("\nSee RRT_RL_STATE_MANAGEMENT.md for complete documentation")
    print("="*70 + "\n")
