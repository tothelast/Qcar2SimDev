#!/usr/bin/env python3
"""
Generate multiple valid routes for the QLabs Cityscape map.
Creates route files in JSON format with ~1.0m waypoint spacing.
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


def generate_route(roadmap, node_sequence, route_name):
    """
    Generate a route from a node sequence.
    
    Args:
        roadmap: SDCSRoadMap instance
        node_sequence: List of node IDs
        route_name: Name for the route
        
    Returns:
        Dictionary with route data or None if generation fails
    """
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
        
        # Get spawn location and heading
        spawn_location = downsampled[0]
        start_node_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
        spawn_heading = start_node_pose[2]
        spawn_rotation = [0.0, 0.0, float(spawn_heading)]
        
        # Calculate route statistics
        total_distance = 0.0
        for i in range(len(downsampled) - 1):
            dx = downsampled[i+1][0] - downsampled[i][0]
            dy = downsampled[i+1][1] - downsampled[i][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        route_data = {
            "name": route_name,
            "node_sequence": node_sequence,
            "waypoints": downsampled,
            "spawn_location": spawn_location,
            "spawn_rotation": spawn_rotation,
            "num_waypoints": len(downsampled),
            "total_distance": float(total_distance)
        }
        
        return route_data
        
    except Exception as e:
        print(f"ERROR generating route {route_name}: {e}")
        return None


def main():
    print("="*80)
    print("ROUTE GENERATION - QLabs Cityscape")
    print("="*80)
    
    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    
    # Define 10 diverse routes
    routes_to_generate = [
        ([13, 19, 17, 20, 22], "roundabout_navigation"),
        ([0, 2, 4], "simple_straight"),
        ([14, 16, 17, 15], "traffic_circle"),
        ([11, 12, 0, 2, 4, 14], "long_route"),
        ([1, 7, 5], "short_route"),
        ([18, 11, 12], "kink_street"),
        ([8, 10, 1], "one_way_street"),
        ([13, 19, 17, 15, 6], "complex_route"),
        ([17, 20, 22, 9], "roundabout_exit"),
        ([12, 7, 14, 16, 17, 15, 6, 13], "full_circuit"),
    ]
    
    # Create routes directory
    routes_dir = os.path.join(parent_dir, "routes")
    os.makedirs(routes_dir, exist_ok=True)
    print(f"\nCreating routes in: {routes_dir}")
    
    # Generate each route
    successful = 0
    failed = 0
    
    for node_sequence, route_name in routes_to_generate:
        print(f"\nGenerating route: {route_name}")
        print(f"  Node sequence: {node_sequence}")
        
        route_data = generate_route(roadmap, node_sequence, route_name)
        
        if route_data:
            # Save to JSON file
            filename = f"{route_name}.json"
            filepath = os.path.join(routes_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(route_data, f, indent=2)
            
            print(f"  ✓ Saved: {filename}")
            print(f"    Waypoints: {route_data['num_waypoints']}")
            print(f"    Distance: {route_data['total_distance']:.1f}m")
            successful += 1
        else:
            print(f"  ✗ Failed to generate route")
            failed += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Successfully generated: {successful} routes")
    print(f"Failed: {failed} routes")
    print(f"Routes saved to: {routes_dir}")
    print("="*80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

