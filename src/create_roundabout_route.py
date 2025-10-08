#!/usr/bin/env python3
"""
Create a roundabout route for QCar2 in QLabs Cityscape Lite.

Route description:
1. Start from spawn [0, -1.3] heading 90° (North)
2. Drive straight north
3. Enter roundabout and turn right (east)
4. Turn right again (south) 
5. Drive south on parallel road
6. End before reaching the southern turn

This uses the SDCSRoadMap to generate smooth waypoints.
"""

import sys
import os
import numpy as np

# Add python directory to path for HAL library
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python'))

from hal.products.mats import SDCSRoadMap


def create_roundabout_route():
    """
    Create a route that goes through the roundabout.
    
    Based on the map analysis:
    - Spawn: [0, -1.3] heading 90° (North)
    - The map shows nodes 0-23
    - We need to go: straight north → roundabout → turn right → south on parallel road
    
    Looking at the node positions and the map image:
    - Start near node 0 (bottom center)
    - Go north through nodes 1, 12, 11
    - Enter roundabout at node 18
    - Exit roundabout at node 16 (east side)
    - Turn to node 17 (heading south)
    - Go south through nodes 14, 15
    - End around node 5 or 4
    
    Returns:
        List of waypoints [[x, y, z], ...]
    """
    # Initialize SDCSRoadMap (right-hand traffic, full map)
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    
    # Define node sequence for the route
    # Based on map analysis and edge connections
    # Path: 0 → 1 → (through intersection) → roundabout → east → south → back
    
    # Looking at the map and node positions:
    # Node 0: [1134, 2299] - bottom center (near spawn)
    # Node 1: [1266, 2323] - slightly north-east of 0
    # Node 12: [1134, 1454] - north of 0
    # Node 18: [1523, 958] - roundabout area
    # Node 16: [1580, 540] - east side of roundabout
    # Node 17: [1854.4, 814.5] - turning point
    # Node 14: [2242, 905] - east road going south
    # Node 15: [2109, 1454] - south on east road
    # Node 4: [2242, 2323] - back to southern area
    
    # Try a simple path first: 0 → 12 → 18 → 16 → 17 → 14 → 15
    node_sequence = [0, 12, 18, 16, 17, 14, 15]
    
    print("="*80)
    print("Generating Roundabout Route")
    print("="*80)
    print(f"Node sequence: {node_sequence}")
    
    # Generate path using SDCSRoadMap
    try:
        waypoints_mm = roadmap.generate_path(node_sequence)
        print(f"Generated {len(waypoints_mm)} waypoints from SDCSRoadMap")
    except Exception as e:
        print(f"Error generating path: {e}")
        print("Trying alternative node sequence...")
        # Try alternative: 0 → 1 → 13 → 18 → 19 → 17 → 14
        node_sequence = [0, 1, 13, 18, 19, 17, 14]
        waypoints_mm = roadmap.generate_path(node_sequence)
        print(f"Generated {len(waypoints_mm)} waypoints from alternative path")
    
    # Convert from millimeters to meters
    waypoints_m = [[x/1000.0, y/1000.0, 0.0] for x, y, _ in waypoints_mm]
    
    # Print first few waypoints to understand the coordinate system
    print(f"\nFirst 5 waypoints (meters):")
    for i, wp in enumerate(waypoints_m[:5]):
        print(f"  [{i}]: [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}]")
    
    # Calculate offset to align with spawn location
    # Spawn is at [0, -1.3], and we want to start from there
    # Find the offset needed
    first_waypoint = np.array(waypoints_m[0][:2])
    spawn_location = np.array([0.0, -1.3])
    
    # Calculate offset
    offset = spawn_location - first_waypoint
    print(f"\nOffset to align with spawn: [{offset[0]:.3f}, {offset[1]:.3f}]")
    
    # Apply offset to all waypoints
    waypoints_aligned = []
    for wp in waypoints_m:
        aligned_wp = [wp[0] + offset[0], wp[1] + offset[1], wp[2]]
        waypoints_aligned.append(aligned_wp)
    
    # Downsample to ~2-5 meter spacing for smoother control
    waypoints_downsampled = downsample_waypoints(waypoints_aligned, target_spacing=3.0)
    
    # Add lead-in waypoints from spawn to first route waypoint if needed
    spawn_to_first = np.linalg.norm(
        np.array(waypoints_downsampled[0][:2]) - spawn_location
    )
    
    if spawn_to_first > 2.0:
        # Add intermediate waypoints
        num_intermediate = int(spawn_to_first / 2.0)
        lead_in = []
        for i in range(num_intermediate):
            t = (i + 1) / (num_intermediate + 1)
            x = spawn_location[0] + t * (waypoints_downsampled[0][0] - spawn_location[0])
            y = spawn_location[1] + t * (waypoints_downsampled[0][1] - spawn_location[1])
            lead_in.append([x, y, 0.0])
        waypoints_final = [[spawn_location[0], spawn_location[1], 0.0]] + lead_in + waypoints_downsampled
    else:
        waypoints_final = [[spawn_location[0], spawn_location[1], 0.0]] + waypoints_downsampled
    
    return waypoints_final


def downsample_waypoints(waypoints, target_spacing=3.0):
    """
    Downsample waypoints to achieve target spacing.
    
    Args:
        waypoints: List of [x, y, z] waypoints
        target_spacing: Target distance between waypoints (meters)
        
    Returns:
        Downsampled list of waypoints
    """
    if len(waypoints) <= 2:
        return waypoints
    
    downsampled = [waypoints[0]]
    accumulated_distance = 0.0
    
    for i in range(1, len(waypoints)):
        segment_distance = np.linalg.norm(
            np.array(waypoints[i][:2]) - np.array(waypoints[i-1][:2])
        )
        accumulated_distance += segment_distance
        
        if accumulated_distance >= target_spacing:
            downsampled.append(waypoints[i])
            accumulated_distance = 0.0
    
    # Always include the last waypoint
    if downsampled[-1] != waypoints[-1]:
        downsampled.append(waypoints[-1])
    
    return downsampled


def print_route_for_config(waypoints):
    """Print waypoints in the format needed for config.py"""
    print("\n" + "="*80)
    print("Roundabout Route for config.py")
    print("="*80)
    print(f"Total waypoints: {len(waypoints)}")
    print(f"Start: [{waypoints[0][0]:.3f}, {waypoints[0][1]:.3f}]")
    print(f"End: [{waypoints[-1][0]:.3f}, {waypoints[-1][1]:.3f}]")
    
    # Calculate total distance
    total_dist = 0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        total_dist += np.sqrt(dx**2 + dy**2)
    print(f"Total distance: {total_dist:.1f} meters")
    
    print("\n" + "="*80)
    print("Copy this into src/config.py:")
    print("="*80)
    print("\nself.route_waypoints = [")
    for i, wp in enumerate(waypoints):
        comment = ""
        if i == 0:
            comment = "  # Spawn location"
        elif i == len(waypoints) - 1:
            comment = "  # End of route"
        elif i % 10 == 0:
            comment = f"  # Waypoint {i}"
        print(f"    [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}],{comment}")
    print("]")
    
    # Print route description
    print("\n" + "="*80)
    print("Route Description:")
    print("="*80)
    print("- Type: Roundabout route with right turns")
    print(f"- Start: [{waypoints[0][0]:.3f}, {waypoints[0][1]:.3f}] heading 90° (North)")
    print(f"- End: [{waypoints[-1][0]:.3f}, {waypoints[-1][1]:.3f}]")
    print(f"- Total length: {total_dist:.1f} meters")
    print(f"- Total waypoints: {len(waypoints)}")
    print("- Path: Straight north → roundabout → right turn → south on parallel road")


if __name__ == "__main__":
    try:
        waypoints = create_roundabout_route()
        print_route_for_config(waypoints)
    except Exception as e:
        print(f"Error creating route: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print("Fallback: Creating manual route based on map coordinates")
        print("="*80)

        # Manual route based on map analysis and the provided map image
        # Map coordinates: X: -2 to 3, Y: -1 to 5 (meters)
        # Spawn: [0, -1.3] heading 90° (North)

        # Route description:
        # 1. Start at spawn [0, -1.3], go straight north along X=0
        # 2. Continue north to Y≈3.5 (approaching roundabout)
        # 3. Enter roundabout, curve right (east)
        # 4. Exit roundabout heading south along X≈2
        # 5. Drive south back to Y≈0.5
        # Total distance: ~20-25 meters

        spawn = [0.0, -1.3, 0.0]

        # Create waypoints with 1-2 meter spacing for smooth control
        manual_waypoints = [
            # Straight north section (X=0)
            [0.0, -1.3, 0.0],   # Spawn
            [0.0, -0.3, 0.0],   # 1m north
            [0.0, 0.7, 0.0],    # 2m north
            [0.0, 1.7, 0.0],    # 3m north
            [0.0, 2.7, 0.0],    # 4m north
            [0.0, 3.2, 0.0],    # Approaching roundabout

            # Roundabout entry and right turn
            [0.2, 3.5, 0.0],    # Start curving right
            [0.5, 3.7, 0.0],    # Entering roundabout
            [0.8, 3.8, 0.0],    # In roundabout
            [1.2, 3.8, 0.0],    # Continuing curve
            [1.5, 3.7, 0.0],    # Apex of turn
            [1.8, 3.5, 0.0],    # Exiting roundabout
            [2.0, 3.2, 0.0],    # Exit complete, heading south

            # Straight south section (X=2)
            [2.0, 2.7, 0.0],    # 1m south from exit
            [2.0, 2.2, 0.0],    # Continue south
            [2.0, 1.7, 0.0],    # Continue south
            [2.0, 1.2, 0.0],    # Continue south
            [2.0, 0.7, 0.0],    # Continue south
            [2.0, 0.5, 0.0],    # End point (before turn)
        ]

        print_route_for_config(manual_waypoints)

