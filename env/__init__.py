"""
Example: How to register your custom environment with Gymnasium.

This allows you to use your custom environment just like FetchPush-v2:
    python train.py env_name=CentralizedTwoRobot-v1
"""

from gymnasium.envs.registration import register

# Register your custom environment
register(
    id='CentralizedTwoRobot-v1',
    entry_point='env.Env_v1:CentralizedTwoRobotEnv',
    max_episode_steps=50,
    kwargs={
        'group_id': 0,
        'robot_radius': 0.5,
        'max_episode_steps': 1024,
    }
)

print("✅ CentralizedTwoRobot-v1 registered successfully!")
print("\nYou can now use it in training:")
print("  python train.py env_name=CentralizedTwoRobot-v1")
print("\nOr test it:")
print("  import gymnasium as gym")
print("  env = gym.make('CentralizedTwoRobot-v1')")
