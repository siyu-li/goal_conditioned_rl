"""
Utility script to organize trained symmetric policies.

For symmetric policies, we only need to organize by number of robots,
not by specific robot indices, since one policy works for any subset.

Directory Structure:
    trained_policies/
    ├── 1_robot/
    │   ├── model_final.pkl
    │   └── .hydra/
    ├── 2_robots/
    │   ├── model_final.pkl
    │   └── .hydra/
    └── 3_robots/
        ├── model_final.pkl
        └── .hydra/

Usage:
    python organize_policies.py --source symmetric_policies/ --dest trained_policies/
    python organize_policies.py --source parker/ --dest trained_policies/  # legacy
"""
import os
import shutil
import argparse
from pathlib import Path
import yaml
import re


def parse_num_robots_from_env_name(env_name: str) -> int:
    """
    Parse the number of controlled robots from environment name.
    
    Examples:
        'SymmetricMultiRobot-2-v1' -> 2
        'Controlled2Robot_idx_0_2-v1' -> 2
        'CentralizedTwoRobot-v1' -> 2
    """
    # New symmetric format
    match = re.search(r'SymmetricMultiRobot-(\d+)', env_name)
    if match:
        return int(match.group(1))
    
    # Legacy format with explicit number
    match = re.search(r'Controlled(\d+)Robot', env_name)
    if match:
        return int(match.group(1))
    
    # Word-based naming
    word_to_num = {
        'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6
    }
    for word, num in word_to_num.items():
        if word in env_name:
            return num
    
    return None


def organize_policy(source_dir: Path, dest_base: Path, dry_run: bool = False):
    """
    Organize a single trained policy into structured format.
    
    For symmetric policies, we organize by number of robots only.
    
    Args:
        source_dir: Directory containing trained model
        dest_base: Base destination directory (e.g., trained_policies/)
        dry_run: If True, only print actions without executing
    
    Returns:
        True if successfully organized
    """
    # Load environment name from config
    hydra_config = source_dir / '.hydra' / 'config.yaml'
    
    if not hydra_config.exists():
        print(f"⚠️  No config found in {source_dir}, skipping")
        return False
    
    with open(hydra_config) as f:
        config = yaml.safe_load(f)
    
    env_name = config.get('env_name', '')
    
    # Parse number of robots
    num_robots = parse_num_robots_from_env_name(env_name)
    
    if num_robots is None:
        print(f"⚠️  Could not parse num_robots from {env_name}, skipping")
        return False
    
    # Construct destination path (organized by number only)
    robot_suffix = "robot" if num_robots == 1 else "robots"
    dest_dir = dest_base / f'{num_robots}_{robot_suffix}'
    
    # Check if model exists
    model_file = source_dir / 'model_final.pkl'
    if not model_file.exists():
        print(f"⚠️  No model_final.pkl in {source_dir}, skipping")
        return False
    
    # Copy files
    if dry_run:
        print(f"[DRY RUN] Would copy:")
        print(f"  {source_dir} -> {dest_dir}")
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        shutil.copy2(model_file, dest_dir / 'model_final.pkl')
        
        # Copy .hydra directory
        if (source_dir / '.hydra').exists():
            if (dest_dir / '.hydra').exists():
                shutil.rmtree(dest_dir / '.hydra')
            shutil.copytree(source_dir / '.hydra', dest_dir / '.hydra')
        
        print(f"✓ Copied {env_name} -> {dest_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Organize trained symmetric policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
    trained_policies/
    ├── 1_robot/
    │   └── model_final.pkl
    ├── 2_robots/
    │   └── model_final.pkl
    └── 3_robots/
        └── model_final.pkl
        """
    )
    parser.add_argument('--source', type=str, default='symmetric_policies/',
                       help='Source directory containing training runs')
    parser.add_argument('--dest', type=str, default='trained_policies/',
                       help='Destination base directory for organized policies')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print actions without executing')
    
    args = parser.parse_args()
    
    source_base = Path(args.source)
    dest_base = Path(args.dest)
    
    if not source_base.exists():
        print(f"❌ Source directory {source_base} does not exist")
        return
    
    print(f"Organizing symmetric policies from {source_base} to {dest_base}")
    print("=" * 70)
    
    # Find all training run directories
    success_count = 0
    skip_count = 0
    
    # Handle both flat and nested structures
    # Flat: symmetric_policies/1_robots/model_final.pkl
    # Nested: symmetric_policies/SymmetricMultiRobot-2-v1/sac_13-31-11/model_final.pkl
    
    for item in source_base.iterdir():
        if not item.is_dir():
            continue
        
        # Check if this directory contains a model directly
        if (item / 'model_final.pkl').exists():
            if organize_policy(item, dest_base, args.dry_run):
                success_count += 1
            else:
                skip_count += 1
        else:
            # Look for nested agent run directories
            for run_dir in item.iterdir():
                if not run_dir.is_dir():
                    continue
                
                if organize_policy(run_dir, dest_base, args.dry_run):
                    success_count += 1
                else:
                    skip_count += 1
    
    print("=" * 70)
    print(f"✓ Successfully organized: {success_count}")
    print(f"⚠️  Skipped: {skip_count}")
    
    if not args.dry_run and dest_base.exists():
        print(f"\nPolicies organized in: {dest_base}")
        print("\nAvailable symmetric policies:")
        print("-" * 40)
        
        for num_dir in sorted(dest_base.iterdir()):
            if num_dir.is_dir():
                model_path = num_dir / 'model_final.pkl'
                status = "✓" if model_path.exists() else "⚠️"
                print(f"  {status} {num_dir.name}")
        
        print("-" * 40)
        print("\nUsage:")
        print("  from modules.policy_manager import PolicyManager")
        print("  pm = PolicyManager('trained_policies/')")
        print("  policy = pm.get_policy(num_robots=2)")


if __name__ == '__main__':
    main()
