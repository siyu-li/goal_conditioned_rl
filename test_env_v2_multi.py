"""
Test script for CentralizedMultiRobotEnv with different numbers of robots.
"""

import numpy as np
from env.Env_v2 import CentralizedMultiRobotEnv


def test_env_with_n_robots(num_robots, num_steps=5):
    """Test the environment with a specific number of robots."""
    print(f"\n{'='*60}")
    print(f"Testing with {num_robots} robots")
    print(f"{'='*60}")
    
    try:
        # Create environment
        env = CentralizedMultiRobotEnv(
            max_episode_steps=100,
            num_robots=num_robots,
            robot_radius=0.5,
            render_mode=None
        )
        
        print(f"✓ Environment created successfully")
        print(f"  - Action space: {env.action_space.shape}")
        print(f"  - Observation space: {env.observation_space['observation'].shape}")
        print(f"  - Achieved goal space: {env.observation_space['achieved_goal'].shape}")
        print(f"  - Desired goal space: {env.observation_space['desired_goal'].shape}")
        print(f"  - Number of robots: {env.num_robots}")
        
        # Reset environment
        obs, info = env.reset(seed=42)
        print(f"\n✓ Environment reset successfully")
        print(f"  - Initial state shape: {env.state.shape}")
        print(f"  - Goal position shape: {env.goal_position.shape}")
        print(f"  - Info: {info}")
        
        # Take a few random steps
        print(f"\n✓ Taking {num_steps} random steps:")
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {step+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                print(f"  Episode ended early: {info}")
                break
        
        # Test compute_reward with batch
        print(f"\n✓ Testing compute_reward with batch:")
        achieved = obs["achieved_goal"].reshape(1, -1)
        desired = obs["desired_goal"].reshape(1, -1)
        batch_reward = env.compute_reward(achieved, desired, info)
        print(f"  Batch reward shape: {np.asarray(batch_reward).shape}")
        print(f"  Batch reward value: {batch_reward}")
        
        # Test success condition
        print(f"\n✓ Testing success condition:")
        # Move all robots to their goals
        env.state[:, :2] = env.goal_position.copy()
        obs = env._get_obs()
        success_mask = env._success_mask(obs["achieved_goal"], obs["desired_goal"])
        print(f"  Success mask: {success_mask}")
        print(f"  Is goal: {env.is_goal(env.state)}")
        
        env.close()
        print(f"\n✓ All tests passed for {num_robots} robots!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error testing with {num_robots} robots:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests with different numbers of robots."""
    print("="*60)
    print("Testing CentralizedMultiRobotEnv with various robot counts")
    print("="*60)
    
    # Test with different numbers of robots
    test_cases = [1, 2, 3, 4, 5, 6, 8]
    results = {}
    
    for num_robots in test_cases:
        results[num_robots] = test_env_with_n_robots(num_robots)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for num_robots, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{num_robots} robots: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("="*60)


if __name__ == "__main__":
    main()
