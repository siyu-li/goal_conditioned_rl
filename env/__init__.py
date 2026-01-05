"""
Example: How to register your custom environment with Gymnasium.

This allows you to use your custom environment just like FetchPush-v2:
    python train.py env_name=CentralizedTwoRobot-v1
"""

from gymnasium.envs.registration import register
from pathlib import Path
# =============================================================================
# Register CentralizedTwoRobot-v1 environment
# regular state representation
register(
    id='CentralizedTwoRobot-v1',
    entry_point='env.Env_v1:CentralizedTwoRobotEnv',
    max_episode_steps=50,
    kwargs={
        'group_id': 0,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedTwoRobot-v1_5 environment
# obstacle as anchor state representation
register(
    id='CentralizedTwoRobot-v1_5',
    entry_point='env.Env_v1_5:CentralizedTwoRobotEnv',
    max_episode_steps=50,
    kwargs={
        'group_id': 0,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedMultiRobotEnv-v2 environment
# obstacle as anchor state representation
register(
    id='CentralizedMultiRobotEnv-v2',
    entry_point='env.Env_v2:CentralizedMultiRobotEnv',
    max_episode_steps=50,
    kwargs={
        'num_robots': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)


# Register CentralizedThreeRobot-v2_3r environment
# obstacle as anchor state representation
# uncoupled action mode
register(
    id='CentralizedThreeRobot-v2_3r',
    entry_point='env.Env_v2_3r:CentralizedThreeRobotEnv',
    max_episode_steps=50,
    kwargs={
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedMultiRobotEnv-v3 environment
# with obstacle as anchor state representation
# coupled action mode
register(
    id='CentralizedMultiRobotEnv-v3',
    entry_point='env.Env_v3:CentralizedMultiRobotEnv',
    max_episode_steps=50,
    kwargs={
        'num_robots': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedMultiRobotEnv-v4 environment
# no obstacles, action can be coupled or uncoupled
register(
    id='CentralizedMultiRobotEnv-v4',
    entry_point='env.Env_v4:CentralizedMultiRobotEnv',
    max_episode_steps=50,
    kwargs={
        'num_robots': 4,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
        'action_coupled': False,
    }
)

# Register CentralizedMultiRobotEnv-v5 environment
# obstacle as anchor state representation + decouple action + neighbor awareness
register(
    id='CentralizedMultiRobotEnv-v5',
    entry_point='env.Env_v5:CentralizedMultiRobotEnv',
    max_episode_steps=50,
    kwargs={
        'num_robots': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedMultiRobotEnv-v6 environment
# no obstacle + couple/decouple action + neighbor awareness
register(
    id='CentralizedMultiRobotEnv-v6',
    entry_point='env.Env_v6:CentralizedMultiRobotEnv',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Register CentralizedMultiRobotEnv-v7 environment
# random obstacle (0,1,2,3) + couple/decouple action + neighbor awareness
# This uses a factory function to support dynamic kwargs from command line
def _make_v7_env(**kwargs):
    """
    Factory function for CentralizedMultiRobotEnv-v7 with dynamic kwargs.
    Supports command-line overrides like: num_obstacles=2 num_robots=4
    """
    from env.Env_v7 import CentralizedMultiRobotEnv
    
    # Default values (can be overridden via kwargs)
    default_kwargs = {
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
        'render_mode': None,
    }
    
    # Merge defaults with provided kwargs (kwargs take precedence)
    final_kwargs = {**default_kwargs, **kwargs}
    
    return CentralizedMultiRobotEnv(**final_kwargs)


# Base v7 environment (default: 3 robots, 1 obstacle)
register(
    id='CentralizedMultiRobotEnv-v7',
    entry_point='env:_make_v7_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Specific configurations for parallel training
# 3 robots with varying obstacles (0, 1, 2, 3)
register(
    id='CentralizedMultiRobotEnv-v7-0obs',
    entry_point='env:_make_v7_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 0,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-1obs',
    entry_point='env:_make_v7_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-2obs',
    entry_point='env:_make_v7_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 2,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-3obs',
    entry_point='env:_make_v7_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# =============================================================================
# NORMALIZED WRAPPER for CentralizedMultiRobotEnv-v7
# Normalizes all positions and distances to [-1, 1] or [0, 1]
# =============================================================================
def _make_v7_normalized_env(**kwargs):
    """
    Factory function for normalized CentralizedMultiRobotEnv-v7.
    
    Creates base v7 environment and wraps it with NormalizedEnvV7Wrapper.
    This normalizes:
    - Robot positions to [-1, 1]
    - Goal positions to [-1, 1]
    - Distances to [0, 1]
    - achieved_goal and desired_goal to [-1, 1]
    """
    from env.Env_v7 import CentralizedMultiRobotEnv
    from env.Env_v7_normalized_wrapper import NormalizedEnvV7Wrapper
    
    # Default values (can be overridden via kwargs)
    default_kwargs = {
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
        'render_mode': None,
    }
    
    # Merge defaults with provided kwargs (kwargs take precedence)
    final_kwargs = {**default_kwargs, **kwargs}
    
    # Create base environment
    base_env = CentralizedMultiRobotEnv(**final_kwargs)
    
    # Wrap with normalization
    return NormalizedEnvV7Wrapper(base_env)


# Base v7-normalized environment (default: 3 robots, 1 obstacle)
register(
    id='CentralizedMultiRobotEnv-v7-normalized',
    entry_point='env:_make_v7_normalized_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# Specific normalized configurations for parallel training
register(
    id='CentralizedMultiRobotEnv-v7-0obs-normalized',
    entry_point='env:_make_v7_normalized_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 0,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-1obs-normalized',
    entry_point='env:_make_v7_normalized_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-2obs-normalized',
    entry_point='env:_make_v7_normalized_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 2,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-3obs-normalized',
    entry_point='env:_make_v7_normalized_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# =============================================================================
# REWARD SHAPING VARIANT for CentralizedMultiRobotEnv-v7
# Adds obstacle proximity penalty for better learning with obstacles
# =============================================================================
def _make_v7_rewardshaping_env(**kwargs):
    """
    Factory function for CentralizedMultiRobotEnv-v7-rewardshaping with dynamic kwargs.
    
    This variant includes obstacle proximity penalty to provide gradual feedback
    for obstacle avoidance, improving learning when obstacles are present.
    """
    from env.Env_v7_rewardshaping import CentralizedMultiRobotEnv
    
    # Default values (can be overridden via kwargs)
    default_kwargs = {
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
        'render_mode': None,
    }
    
    # Merge defaults with provided kwargs (kwargs take precedence)
    final_kwargs = {**default_kwargs, **kwargs}
    
    return CentralizedMultiRobotEnv(**final_kwargs)


# Base v7-rewardshaping environment (default: 3 robots, 1 obstacle)
register(
    id='CentralizedMultiRobotEnv-v7-rewardshaping',
    entry_point='env:_make_v7_rewardshaping_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-1obs-rewardshaping',
    entry_point='env:_make_v7_rewardshaping_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 1,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-2obs-rewardshaping',
    entry_point='env:_make_v7_rewardshaping_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 2,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

register(
    id='CentralizedMultiRobotEnv-v7-3obs-rewardshaping',
    entry_point='env:_make_v7_rewardshaping_env',
    max_episode_steps=50,
    kwargs={
        'action_coupled': False,
        'num_robots': 3,
        'num_obstacles': 3,
        'robot_radius': 0.5,
        'max_episode_steps': 50,
    }
)

# =============================================================================
# CONTROLLED SUBSET ENVIRONMENTS (Self-contained, no parent env)
# =============================================================================
# These are standalone environments that manage all 6 robots internally.
# They can be used with or without the wrapper for relative coordinates.

def _make_controlled_subset(controlled_indices, use_relative_coords=True, normalize_coords=False, **kwargs):
    """
    Factory function for creating ControlledSubsetEnv with optional wrapper.
    
    Args:
        controlled_indices: List of robot indices to control
        use_wrapper: If True, wrap with ControlledSubsetWrapper for relative coords
        use_relative_coords: If True (and use_wrapper=True), use relative coordinates
        normalize_coords: If True (and use_wrapper=True), normalize coordinates
        **kwargs: Additional arguments for ControlledSubsetEnv
    """
    project_root = Path(__file__).resolve().parents[1]
    obstacle_cfg = kwargs.get('obstacle_config_path', 'configs/obstacles_default.yaml')
    obstacle_cfg_path = Path(obstacle_cfg)
    if not obstacle_cfg_path.is_absolute():
        obstacle_cfg_path = project_root / obstacle_cfg

    
    from env.Env_controlled_subset import ControlledSubsetEnv
    
    base_env = ControlledSubsetEnv(
        controlled_robot_indices=controlled_indices,
        obstacle_config_path=str(obstacle_cfg_path),
        max_episode_steps=kwargs.get('max_episode_steps', 50),
        action_coupled=kwargs.get('action_coupled', False),
        robot_radius=kwargs.get('robot_radius', 0.5),
        render_mode=kwargs.get('render_mode', None),
    )
    
    from env.Env_controlled_subset_wrapper import ControlledSubsetWrapper
    return ControlledSubsetWrapper(
        base_env,
        use_relative_coords=use_relative_coords,
        normalize_coords=normalize_coords,
    )


register(
    id='ControlledSubset1Robot-wrapped',
    entry_point='env:_make_controlled_subset',
    max_episode_steps=200,
    kwargs={
        'controlled_indices': [0],
        'use_relative_coords': True,
        'normalize_coords': True,
        'max_episode_steps': 200,
    }
)


register(
    id='ControlledSubset2Robot-wrapped',
    entry_point='env:_make_controlled_subset',
    max_episode_steps=200,
    kwargs={
        'controlled_indices': [0, 2],
        'use_relative_coords': True,
        'normalize_coords': True,
        'max_episode_steps': 200,
    }
)

register(
    id='ControlledSubset3Robot-wrapped',
    entry_point='env:_make_controlled_subset',
    max_episode_steps=200,
    kwargs={
        'controlled_indices': [0, 2, 4],
        'use_relative_coords': True,
        'normalize_coords': True,
        'max_episode_steps': 200,
    }
)
