"""
Training script for wrapped controlled subset environment policies.

This script trains RL policies for the ControlledSubset environments
with relative coordinates and normalization via the wrapper.

Key Concept:
- Train ONE policy per number of robots (1, 2, 3)
- Environments use wrapped observations (relative coords + normalization)
- Each policy controls a specific subset configuration

Usage:
    # Train all configurations (1, 2, 3 robots)
    python train_child_policies.py --all
    
    # Train specific number of robots
    python train_child_policies.py --num-robots 2 3
    
    # Train with specific agent and epochs
    python train_child_policies.py --num-robots 2 --agent sac --n-epochs 200
    
    # Dry run to see what would be trained
    python train_child_policies.py --all --dry-run
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_wrapped_env_name(num_robots: int) -> str:
    """
    Get the environment name for wrapped controlled subset training.
    
    Args:
        num_robots: Number of robots to control
    
    Returns:
        Environment name string
    """
    return f"ControlledSubset{num_robots}Robot-wrapped"


def train_policy(env_name: str,
                 agent: str = 'sac',
                 n_epochs: int = 200,
                 n_processes: int = 1,
                 save_dir: str = None,
                 extra_args: list = None) -> bool:
    """
    Train a single policy using the main training script.
    
    Args:
        env_name: Name of environment
        agent: 'sac' or 'ddpg'
        n_epochs: Number of training epochs
        n_processes: Number of MPI processes
        save_dir: Directory to save the model
        extra_args: Additional arguments for train.py
    
    Returns:
        True if training succeeded
    """
    cmd = []
    
    if n_processes > 1:
        cmd.extend(['mpirun', '-np', str(n_processes)])
    
    cmd.extend([
        'python', '-u', 'train.py',
        f'agent={agent}',
        f'env_name={env_name}',
        f'n_epochs={n_epochs}',
    ])
    
    if save_dir:
        cmd.append(f'save_dir={save_dir}')
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"Training: {env_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for {env_name}: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Could not find train.py or required dependencies: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train wrapped controlled subset environment policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train policies for 1, 2, and 3 robots
    python train_child_policies.py --all
    
    # Train only 2-robot policy
    python train_child_policies.py --num-robots 2
    
    # Train with more epochs
    python train_child_policies.py --num-robots 2 --n-epochs 500
    
    # Use DDPG instead of SAC
    python train_child_policies.py --num-robots 2 --agent ddpg
        """
    )
    
    parser.add_argument('--num-robots', type=int, nargs='+', default=None,
                       help='Number of robots to control (e.g., 1 2 3)')
    parser.add_argument('--all', action='store_true',
                       help='Train all configurations (1, 2, 3 robots)')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'ddpg'],
                       help='RL agent type (default: sac)')
    parser.add_argument('--n-epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--n-processes', type=int, default=1,
                       help='Number of MPI processes (default: 1)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Base directory to save policies (default: wrapped_policies/)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print training configurations without executing')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--extra-args', type=str, nargs='*', default=[],
                       help='Additional arguments to pass to train.py')
    
    args = parser.parse_args()
    
    # Determine which configurations to train
    if args.all:
        num_robots_list = [1, 2, 3]
    elif args.num_robots:
        num_robots_list = args.num_robots
    else:
        print("❌ Please specify --num-robots or use --all")
        parser.print_help()
        return
    
    
    # Generate training configurations
    configs = [(n, get_wrapped_env_name(n)) for n in num_robots_list]
    
    print("\n" + "=" * 70)
    print("WRAPPED CONTROLLED SUBSET POLICY TRAINING")
    print("=" * 70)
    print(f"\nTraining configurations ({len(configs)} policies):")
    print("-" * 70)
    for i, (num, env_name) in enumerate(configs, 1):
        print(f"  {i}. {env_name:40s} (controls {num} robot{'s' if num > 1 else ''})")
    print("-" * 70)
    print(f"\nSettings:")
    print(f"  Agent:      {args.agent}")
    print(f"  Epochs:     {args.n_epochs}")
    print(f"  Processes:  {args.n_processes}")
    print(f"  Save dir:   {args.save_dir or 'wrapped_policies/'}")
    print("-" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without training")
        return
    
    # Train each configuration
    extra_args = list(args.extra_args)
    if args.use_wandb:
        extra_args.append('use_wandb=true')
    
    success_count = 0
    failed = []
    
    for num_robots, env_name in configs:
        # Set save directory based on number of robots
        if args.save_dir:
            save_dir = f"{args.save_dir}/{num_robots}_robots"
        else:
            save_dir = f"wrapped_policies/{num_robots}_robots"
        
        success = train_policy(
            env_name=env_name,
            agent=args.agent,
            n_epochs=args.n_epochs,
            n_processes=args.n_processes,
            save_dir=save_dir,
            extra_args=extra_args
        )
        
        if success:
            success_count += 1
        else:
            failed.append(env_name)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"✓ Successful: {success_count}/{len(configs)}")
    
    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for env_name in failed:
            print(f"  - {env_name}")
    


if __name__ == '__main__':
    main()
