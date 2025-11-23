#!/usr/bin/env python3
"""
Script to collect data for roundabout navigation with varying static obstacle positions.
This script loads the 'roundabout_navigation' scene and injects a static vehicle
at different locations along the route for each run.
"""

import sys
import math
import copy
import time
from pathlib import Path

# Add parent directory to path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scene_loader import SceneLoader, ActorDefinition
from data_collection.collect_data import main as collect_data_main

def calculate_rotation(p1, p2):
    """Calculate yaw rotation in degrees from p1 to p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    yaw_rad = math.atan2(dy, dx)
    return math.degrees(yaw_rad)

def create_static_car_actor(location, rotation, actor_number=999):
    """Create an ActorDefinition for a static car."""
    actor_data = {
        "name": f"obstacle_car_{actor_number}",
        "description": "Static obstacle car",
        "type": "parked_vehicle", # or 'obstacle' if supported, but parked_vehicle is standard
        "actor_number": actor_number,
        "location": location,
        "rotation": [0.0, 0.0, rotation],
        "configuration": 0 # Color/model variant
    }
    # We pass a dummy path since it's dynamically created
    return ActorDefinition(actor_data, "dynamic_generated")

def main():
    print("Starting Roundabout Navigation Data Collection (Variations)")
    
    # Load the base scene
    loader = SceneLoader()
    base_scene_name = "03_roundabout_navigation"
    base_scene = loader.load_scene(base_scene_name)
    
    if not base_scene:
        print(f"Error: Could not load base scene '{base_scene_name}'")
        return

    # Define obstacle locations (Location [x, y, z], Rotation Yaw)
    # These are approximate locations on the route.
    # You may need to adjust these based on the exact lane center.
    variations = [
        # Variation 1: Adjusted to be further from spawn (Spawn is at y=18.5)
        # Old: [2.89, 22.69] -> Too close (4m)
        # New: Approx Waypoint 10-12, before the stop sign (Stop sign at y=29.0)
        {
            "location": [8.36, 28.99, 0.005],
            "rotation": 45.0 # Facing North-East (approaching roundabout)
        },
        # Variation 2: Entering roundabout (approx waypoint 15)
        # Stop sign is at [11.0, 29.0]. This is just after it.
        {
            "location": [12.98, 31.56, 0.005], 
            "rotation": 0.0 # Facing East
        },
        # Variation 3: In roundabout (approx waypoint 25)
        {
            "location": [21.00, 33.89, 0.005],
            "rotation": 90.0 # Facing North
        },
        # Variation 4: Exiting roundabout (approx waypoint 40)
        # Crosswalk is at y=43.6. This is after it.
        {
            "location": [15.67, 44.97, 0.005],
            "rotation": 180.0 # Facing West
        },
         # Variation 5: Straight after exit (approx waypoint 60)
        {
            "location": [-5.6, 44.97, 0.005],
            "rotation": 180.0 # Facing West
        }
    ]

    num_runs_per_variation = 3 # Adjust as needed
    
    for i, var in enumerate(variations):
        print(f"\n--- Variation {i+1}/{len(variations)} ---")
        print(f"Obstacle at: {var['location']}")
        
        for run in range(num_runs_per_variation):
            print(f"  Run {run+1}/{num_runs_per_variation}")
            
            # Create a deep copy of the scene to modify
            current_scene = copy.deepcopy(base_scene)
            
            # Create and add the obstacle actor
            obstacle = create_static_car_actor(var['location'], var['rotation'])
            current_scene.actors.append(obstacle)
            current_scene.parked_vehicles.append(obstacle) # Update the category list
            
            # Update scene name/description for logging (optional, but good for clarity)
            current_scene.name = f"{base_scene_name}_var{i+1}_run{run+1}"
            current_scene.description += f" (Obstacle at {var['location']})"
            
            # Run data collection
            # Note: The user needs to drive and finish the run.
            success = collect_data_main(scene_definition_override=current_scene, split='train')
            
            if not success:
                print("  Run failed or aborted.")
                # Optional: ask to retry or continue
            
            time.sleep(1) # Brief pause between runs

if __name__ == "__main__":
    main()
