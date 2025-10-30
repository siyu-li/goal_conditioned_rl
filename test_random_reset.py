"""
Test script to verify random initial state and goal sampling in the environment.
"""
import numpy as np
from env.Env_v1 import CentralizedTwoRobotEnv

def test_random_reset():
    """Test that reset generates different random configurations."""
    env = CentralizedTwoRobotEnv(max_episode_steps=100, group_id=0)
    
    print("Testing random reset functionality...")
    print(f"Number of robots: {env.num_robots}")
    print(f"Workspace: x=[{env.x_min}, {env.x_max}], y=[{env.y_min}, {env.y_max}]")
    print(f"Robot radius: {env.robot_radius}")
    print(f"Obstacles: {env.obstacles}")
    print()
    
    # Test multiple resets
    num_tests = 5
    states = []
    goals = []
    
    for i in range(num_tests):
        obs, info = env.reset()
        state = obs['observation'].reshape(env.num_robots, 3)
        goal = obs['desired_goal'].reshape(env.num_robots, 2)
        
        print(f"Reset {i+1}:")
        print(f"  Initial state:")
        for j, (x, y, theta) in enumerate(state):
            print(f"    Robot {j}: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
        print(f"  Goal positions:")
        for j, (gx, gy) in enumerate(goal):
            print(f"    Robot {j}: x={gx:.2f}, y={gy:.2f}")
        
        # Verify no collision with obstacles
        for robot_idx in range(env.num_robots):
            x, y = state[robot_idx, :2]
            for ox, oy, radius in env.obstacles:
                dist = np.linalg.norm([x - ox, y - oy])
                assert dist >= radius + env.robot_radius, \
                    f"Robot {robot_idx} collides with obstacle at ({ox}, {oy})"
        
        # Verify no inter-robot collisions in initial state
        for j in range(env.num_robots):
            for k in range(j + 1, env.num_robots):
                dist = np.linalg.norm(state[j, :2] - state[k, :2])
                assert dist >= 2 * env.robot_radius, \
                    f"Robots {j} and {k} are too close in initial state"
        
        # Verify no goal collisions with obstacles
        for robot_idx in range(env.num_robots):
            gx, gy = goal[robot_idx]
            for ox, oy, radius in env.obstacles:
                dist = np.linalg.norm([gx - ox, gy - oy])
                assert dist >= radius + env.robot_radius, \
                    f"Goal for robot {robot_idx} collides with obstacle at ({ox}, {oy})"
        
        # Verify no inter-goal collisions
        for j in range(env.num_robots):
            for k in range(j + 1, env.num_robots):
                dist = np.linalg.norm(goal[j] - goal[k])
                assert dist >= 2 * env.robot_radius, \
                    f"Goals for robots {j} and {k} are too close"
        
        states.append(state.copy())
        goals.append(goal.copy())
        print("  ✓ All collision checks passed!")
        print()
    
    # Verify that configurations are different across resets
    print("Checking that resets produce different configurations...")
    for i in range(num_tests - 1):
        state_diff = np.linalg.norm(states[i] - states[i+1])
        goal_diff = np.linalg.norm(goals[i] - goals[i+1])
        print(f"  Reset {i+1} vs {i+2}: state_diff={state_diff:.2f}, goal_diff={goal_diff:.2f}")
        assert state_diff > 0.1 or goal_diff > 0.1, \
            "Consecutive resets produced identical configurations"
    
    print("\n✓ All tests passed! Random reset is working correctly.")
    env.close()

if __name__ == "__main__":
    test_random_reset()
