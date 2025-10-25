#!/usr/bin/env python3
"""
Generate extended traffic_circle route: (14, 16, 17, 15, 5)
Starting before node 14 and extending to node 5
"""

import sys
import os
import json
import numpy as np

# Add parent directory and python directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python'))

from hal.products.mats import SDCSRoadMap


def downsample_waypoints(waypoints, target_spacing=1.0):
    """Downsample waypoints to achieve target spacing."""
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


def generate_route():
    """Generate the extended traffic circle route."""
    print("="*80)
    print("EXTENDED TRAFFIC CIRCLE ROUTE GENERATION")
    print("="*80)
    
    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    
    # Extended node sequence: 14 → 16 → 17 → 15 → 5
    node_sequence = [14, 16, 17, 15, 5]
    route_name = "traffic_circle"
    
    print(f"\nGenerating route: {route_name}")
    print(f"  Node sequence: {node_sequence}")
    
    try:
        # Generate path using SDCSRoadMap
        waypoints_scaled = roadmap.generate_path(node_sequence)
        waypoints_sdcs = waypoints_scaled.T
        
        # Convert to QLabs coordinates (multiply by 10)
        waypoints_qlabs = []
        for wp in waypoints_sdcs:
            x_qlabs = wp[0] * 10.0
            y_qlabs = wp[1] * 10.0
            waypoints_qlabs.append([x_qlabs, y_qlabs, 0.0])
        
        # Downsample to ~1m spacing
        downsampled = downsample_waypoints(waypoints_qlabs, target_spacing=1.0)
        
        # Prepend waypoints before node 14 to start earlier
        # Node 14 is at (22.55, 29.67), heading 90° (north)
        # Add 3 waypoints going south before node 14
        node14_pos = downsampled[0]
        prepend_waypoints = [
            [node14_pos[0], node14_pos[1] - 3.0, 0.0],
            [node14_pos[0], node14_pos[1] - 2.0, 0.0],
            [node14_pos[0], node14_pos[1] - 1.0, 0.0],
        ]
        
        final_waypoints = prepend_waypoints + downsampled
        
        # Get spawn location and heading
        spawn_location = final_waypoints[0]
        # Heading is 90° (north) same as node 14
        spawn_heading = np.pi / 2  # 90 degrees
        spawn_rotation = [0.0, 0.0, float(spawn_heading)]
        
        # Calculate route statistics
        total_distance = 0.0
        for i in range(len(final_waypoints) - 1):
            dx = final_waypoints[i+1][0] - final_waypoints[i][0]
            dy = final_waypoints[i+1][1] - final_waypoints[i][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        route_data = {
            "name": route_name,
            "node_sequence": node_sequence,
            "waypoints": final_waypoints,
            "spawn_location": spawn_location,
            "spawn_rotation": spawn_rotation,
            "num_waypoints": len(final_waypoints),
            "total_distance": float(total_distance)
        }
        
        # Save to JSON file
        routes_dir = os.path.join(parent_dir, "config", "routes")
        filename = f"{route_name}.json"
        filepath = os.path.join(routes_dir, filename)
        
        # Backup existing file
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(filepath, backup_path)
            print(f"  Backed up existing route to: {backup_path}")
        
        with open(filepath, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        print(f"  ✓ Saved: {filename}")
        print(f"    Waypoints: {route_data['num_waypoints']}")
        print(f"    Distance: {route_data['total_distance']:.1f}m")
        
        print("\n" + "="*80)
        print("SUCCESS")
        print("="*80)
        print(f"Route saved to: {filepath}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR generating route: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(generate_route())
