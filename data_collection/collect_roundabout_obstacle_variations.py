#!/usr/bin/env python3
"""
Automated data collection for roundabout navigation with obstacle variations.

This script systematically collects expert driving data for the roundabout_navigation
route with randomized obstacle placement and clean route runs. The collected data
will be used to fine-tune the SimLingo model for obstacle detection and avoidance.

Usage:
    python data_collection/collect_roundabout_obstacle_variations.py --num-runs 10
"""

import sys
import argparse
import random
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.collect_data import main as collect_data_main
from core.scene_loader import SceneDefinition, ActorDefinition


class ObstacleVariationGenerator:
    """Generates obstacle variations along the roundabout_navigation route."""
    
    # Route waypoint indices where obstacles can be placed
    # These are strategic positions along the roundabout navigation route
    OBSTACLE_POSITIONS = [
        {'name': 'early_roundabout', 'waypoint_idx': 25, 'description': 'Early in roundabout entrance'},
        {'name': 'mid_roundabout', 'waypoint_idx': 35, 'description': 'Middle of roundabout curve'},
        {'name': 'roundabout_exit', 'waypoint_idx': 50, 'description': 'Roundabout exit area'},
        {'name': 'straight_section', 'waypoint_idx': 65, 'description': 'Straight road section'},
        {'name': 'late_route', 'waypoint_idx': 75, 'description': 'Late in route'},
    ]
    
    # Available obstacle car variations
    OBSTACLE_VARIANTS = ['obstacle_car_var1', 'obstacle_car_var3', 'obstacle_car_var4']
    
    def __init__(self, route_waypoints: list):
        """
        Initialize obstacle variation generator.
        
        Args:
            route_waypoints: List of [x, y, z] waypoints from the route JSON
        """
        self.route_waypoints = route_waypoints
        
    def generate_clean_scene(self, run_number: int) -> SceneDefinition:
        """Generate a scene with no obstacles (clean route)."""
        scene_data = {
            "name": f"roundabout_clean_run{run_number:02d}",
            "description": f"Clean roundabout navigation (Run {run_number})",
            "ego_route": "roundabout_navigation",
            "actors": [],
            "actors_backup": []
        }
        return SceneDefinition(scene_data, scene_path="generated", actors=[])
    
    def generate_obstacle_scene(self, run_number: int, position_name: Optional[str] = None) -> SceneDefinition:
        """
        Generate a scene with an obstacle at a specific or random position.
        
        Args:
            run_number: Run number for naming
            position_name: Optional specific position name, or None for random
            
        Returns:
            SceneDefinition with obstacle placement
        """
        # Select position
        if position_name:
            position = next((p for p in self.OBSTACLE_POSITIONS if p['name'] == position_name), None)
            if not position:
                print(f"WARNING: Position '{position_name}' not found, using random")
                position = random.choice(self.OBSTACLE_POSITIONS)
        else:
            position = random.choice(self.OBSTACLE_POSITIONS)
        
        # Get waypoint at obstacle position
        waypoint_idx = position['waypoint_idx']
        if waypoint_idx >= len(self.route_waypoints):
            waypoint_idx = len(self.route_waypoints) - 1
            
        obstacle_location = self.route_waypoints[waypoint_idx]
        
        # Calculate heading from route direction (next waypoint - current waypoint)
        if waypoint_idx < len(self.route_waypoints) - 1:
            next_wp = self.route_waypoints[waypoint_idx + 1]
            dx = next_wp[0] - obstacle_location[0]
            dy = next_wp[1] - obstacle_location[1]
            
            import math
            heading_rad = math.atan2(dy, dx)
            heading_deg = math.degrees(heading_rad)
        else:
            heading_deg = 90.0  # Default heading
        
        # Randomly select obstacle variant
        obstacle_variant = random.choice(self.OBSTACLE_VARIANTS)
        
        # Create custom obstacle actor definition
        obstacle_actor_data = {
            "name": f"{obstacle_variant}_run{run_number:02d}",
            "description": f"Obstacle at {position['description']} (Run {run_number})",
            "type": "parked_vehicle", # Changed back to parked_vehicle to spawn a car mesh
            "actor_number": 900 + run_number,  # Unique actor number
            "location": [
                float(obstacle_location[0]),
                float(obstacle_location[1]),
                0.005
            ],
            "rotation": [0.0, 0.0, float(heading_deg)],
            "configuration": 0,
            "variant": obstacle_variant # Store variant info
        }
        
        # Wrap in ActorDefinition
        obstacle_actor_def = ActorDefinition(obstacle_actor_data, actor_path="generated")
        
        scene_name = f"roundabout_obstacle_{position['name']}_run{run_number:02d}"
        
        scene_data = {
            "name": scene_name,
            "description": f"Obstacle at {position['description']} (Run {run_number})",
            "ego_route": "roundabout_navigation",
            "actors": [obstacle_actor_data['name']], # List of actor names for reference
            "actors_backup": []
        }
        
        return SceneDefinition(
            scene_data, 
            scene_path="generated", 
            actors=[obstacle_actor_def]
        )


def load_route_waypoints(route_name: str = "roundabout_navigation") -> list:
    """Load route waypoints from config file."""
    import json
    project_root = Path(__file__).parent.parent
    route_path = project_root / "config" / "routes" / f"{route_name}.json"
    
    with open(route_path, 'r') as f:
        route_data = json.load(f)
    
    return route_data['waypoints']


def run_data_collection_variations(
    num_runs: int = 10,
    clean_ratio: float = 0.4,
    split: str = 'train',
    database_root: Optional[Path] = None,
    sequential: bool = False
):
    """
    Run automated data collection with obstacle variations.
    
    Args:
        num_runs: Total number of data collection runs
        clean_ratio: Ratio of clean (no obstacle) runs (0.0 to 1.0)
        split: Dataset split ('train' or 'val')
        database_root: Optional custom database root directory
        sequential: If True, use sequential obstacle positions instead of random
    """
    print("="*80)
    print("ROUNDABOUT OBSTACLE VARIATION DATA COLLECTION")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Total runs: {num_runs}")
    print(f"  - Clean route ratio: {clean_ratio*100:.0f}%")
    print(f"  - Obstacle route ratio: {(1-clean_ratio)*100:.0f}%")
    print(f"  - Dataset split: {split}")
    print(f"  - Sequential obstacles: {sequential}")
    print("="*80)
    
    # Load route waypoints
    print("\nLoading route configuration...")
    route_waypoints = load_route_waypoints()
    print(f"✓ Loaded {len(route_waypoints)} waypoints from roundabout_navigation route")
    
    # Initialize generator
    generator = ObstacleVariationGenerator(route_waypoints)
    
    # Calculate number of clean vs obstacle runs
    num_clean = int(num_runs * clean_ratio)
    num_obstacle = num_runs - num_clean
    
    print(f"\nPlanned runs:")
    print(f"  - Clean routes: {num_clean}")
    print(f"  - Obstacle routes: {num_obstacle}")
    print("="*80)
    
    # Create run schedule
    run_schedule = []
    
    # Add clean runs
    for i in range(num_clean):
        run_schedule.append(('clean', i + 1))
    
    # Add obstacle runs
    if sequential:
        # Use sequential obstacle positions
        positions = generator.OBSTACLE_POSITIONS
        for i in range(num_obstacle):
            position_name = positions[i % len(positions)]['name']
            run_schedule.append(('obstacle', i + 1, position_name))
    else:
        # Random obstacle positions
        for i in range(num_obstacle):
            run_schedule.append(('obstacle', i + 1, None))
    
    # Shuffle the schedule for variety
    random.shuffle(run_schedule)
    
    # Execute data collection runs
    successful_runs = 0
    failed_runs = 0
    
    for run_idx, run_config in enumerate(run_schedule, start=1):
        print("\n" + "="*80)
        print(f"RUN {run_idx}/{num_runs}")
        print("="*80)
        
        try:
            if run_config[0] == 'clean':
                run_number = run_config[1]
                scene_def = generator.generate_clean_scene(run_number)
                print(f"Type: CLEAN ROUTE (no obstacles)")
            else:
                run_number = run_config[1]
                position_name = run_config[2] if len(run_config) > 2 else None
                scene_def = generator.generate_obstacle_scene(run_number, position_name)
                
                if position_name:
                    print(f"Type: OBSTACLE at {position_name}")
                else:
                    print(f"Type: OBSTACLE at random position")
            
            print(f"Scene: {scene_def.name}")
            print(f"\nStarting teleop data collection in 3 seconds...")
            print("CONTROLS: Arrow keys to drive / B to brake / Q to quit")
            time.sleep(3)
            
            # Run data collection
            success = collect_data_main(
                scene_name=None,
                split=split,
                database_root=database_root,
                scene_definition_override=scene_def
            )
            
            if success:
                successful_runs += 1
                print(f"\n✓ Run {run_idx} completed successfully")
            else:
                failed_runs += 1
                print(f"\n✗ Run {run_idx} failed")
                
        except KeyboardInterrupt:
            print("\n\n⚠ Data collection interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error during run {run_idx}: {e}")
            failed_runs += 1
            continue
        
        # Small delay between runs
        if run_idx < num_runs:
            print(f"\nPreparing next run in 2 seconds...")
            time.sleep(2)
    
    # Final summary
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Total runs attempted: {run_idx}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/run_idx*100:.1f}%")
    print("="*80)
    
    if database_root:
        data_dir = database_root / "data" / "simlingo" / f"routes_{split}ing" / "qlabs"
    else:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "database" / "data" / "simlingo" / f"routes_{split}ing" / "qlabs"
    
    print(f"\nData saved to: {data_dir}")
    print("\nNext steps:")
    print("1. Verify data quality (check images and measurements)")
    print("2. Run training with: cd simlingo/simlingo_training && python train.py experiment=qlabs_roundabout_finetune")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Automated data collection for roundabout obstacle variations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 10 runs (40% clean, 60% obstacle, random positions)
  python data_collection/collect_roundabout_obstacle_variations.py --num-runs 10

  # Collect 15 runs with 50/50 clean/obstacle split
  python data_collection/collect_roundabout_obstacle_variations.py --num-runs 15 --clean-ratio 0.5

  # Collect validation data with sequential obstacle positions
  python data_collection/collect_roundabout_obstacle_variations.py --num-runs 8 --split val --sequential

  # Custom database location
  python data_collection/collect_roundabout_obstacle_variations.py --num-runs 10 --database /path/to/database
        """
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Total number of data collection runs (default: 10)'
    )
    
    parser.add_argument(
        '--clean-ratio',
        type=float,
        default=0.4,
        help='Ratio of clean (no obstacle) runs, between 0.0 and 1.0 (default: 0.4)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to save into (default: train)'
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default=None,
        help='Root directory for database (default: <repo>/database)'
    )
    
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Use sequential obstacle positions instead of random (useful for systematic coverage)'
    )
    
    args = parser.parse_args()
    
    # Validate clean ratio
    if not 0.0 <= args.clean_ratio <= 1.0:
        parser.error("--clean-ratio must be between 0.0 and 1.0")
    
    # Convert database path
    database_root = Path(args.database) if args.database else None
    
    # Run collection
    run_data_collection_variations(
        num_runs=args.num_runs,
        clean_ratio=args.clean_ratio,
        split=args.split,
        database_root=database_root,
        sequential=args.sequential
    )
