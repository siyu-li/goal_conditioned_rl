#!/usr/bin/env python3
"""
Script to list all available Gymnasium environments.
Run this to see which environments are registered.
"""

import gymnasium as gym

# Get all registered environment IDs
all_envs = gym.envs.registry.keys()

# Filter for robotics environments
robotics_envs = [env for env in all_envs if 'Fetch' in env or 'Hand' in env]

print("=" * 60)
print("AVAILABLE GYMNASIUM-ROBOTICS ENVIRONMENTS:")
print("=" * 60)

for env_id in sorted(robotics_envs):
    print(f"  - {env_id}")

print("\n" + "=" * 60)
print(f"Total robotics environments: {len(robotics_envs)}")
print("=" * 60)

# Test if FetchPush-v2 is available
print("\nTesting FetchPush-v2:")
try:
    env = gym.make("FetchPush-v2")
    print(f"✅ FetchPush-v2 successfully created!")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Max episode steps: {env.spec.max_episode_steps}")
    env.close()
except Exception as e:
    print(f"❌ Failed to create FetchPush-v2: {e}")
