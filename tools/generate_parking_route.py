#!/usr/bin/env python3
"""
Generate a parking route that ends smoothly in a parking spot.
This script creates routes with custom parking approach waypoints.
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


def generate_smooth_parking_approach(road_waypoints, parking_target, parking_heading, 
                                     approach_distance=8.0, num_transition_points=8):
    """
    Generate smooth transition waypoints from road to parking spot.
    
    Args:
        road_waypoints: List of road waypoints leading to transition point
        parking_target: [x, y, z] target parking position
        parking_heading: heading angle in radians for the parked car
        approach_distance: distance before parking spot to start transition
        num_transition_points: number of waypoints for smooth transition
    
    Returns:
        List of transition waypoints
    """
    # Convert to numpy arrays
    last_road_waypoint = road_waypoints[-1]
    road_pos = np.array(last_road_waypoint[:2])
    park_pos = np.array(parking_target[:2])
    
    # Calculate road direction from last few waypoints
    if len(road_waypoints) >= 3:
        # Use last 3 waypoints to get road direction
        prev_wp = np.array(road_waypoints[-3][:2])
        road_direction = road_pos - prev_wp
        road_direction = road_direction / np.linalg.norm(road_direction)
        road_heading = np.arctan2(road_direction[1], road_direction[0])
    else:
        # Fallback: use direction to parking spot
        to_park = park_pos - road_pos
        road_heading = np.arctan2(to_park[1], to_park[0])
    
    # Generate smooth cubic bezier curve from road to parking
    # Control points for bezier curve
    p0 = road_pos
    p3 = park_pos
    
    # Control point 1: extend from last road waypoint along the ROAD direction
    # This keeps the car on the road path initially
    # Use larger control distance to keep straighter initially
    control_distance = 0.45 * np.linalg.norm(park_pos - road_pos)
    p1 = road_pos + control_distance * np.array([np.cos(road_heading), np.sin(road_heading)])
    
    # Control point 2: extend backward from parking spot along parking heading
    # Reduced to make sharper turn at the end
    p2 = park_pos - 0.3 * approach_distance * np.array([np.cos(parking_heading), 
                                                          np.sin(parking_heading)])
    
    # Generate bezier curve points
    transition_waypoints = []
    for i in range(num_transition_points):
        t = i / (num_transition_points - 1)
        
        # Cubic bezier formula
        point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
        
        transition_waypoints.append([float(point[0]), float(point[1]), 0.0])
    
    return transition_waypoints


def generate_parking_route(roadmap, node_sequence, parking_target, parking_heading, 
                           route_name, approach_distance=8.0):
    """
    Generate a route that ends with a smooth parking maneuver.
    
    Args:
        roadmap: SDCSRoadMap instance
        node_sequence: List of node IDs for the road portion
        parking_target: [x, y, z] target parking position in QLabs coordinates
        parking_heading: heading angle in degrees for the parked position
        route_name: Name for the route
        approach_distance: distance before parking to start smooth transition
        
    Returns:
        Dictionary with route data or None if generation fails
    """
    try:
        # Generate path using SDCSRoadMap for the road portion
        waypoints_scaled = roadmap.generate_path(node_sequence)
        waypoints_sdcs = waypoints_scaled.T
        
        # Convert to QLabs coordinates (multiply by 10)
        waypoints_qlabs = []
        for wp in waypoints_sdcs:
            x_qlabs = wp[0] * 10.0
            y_qlabs = wp[1] * 10.0
            waypoints_qlabs.append([x_qlabs, y_qlabs, 0.0])
        
        # Downsample to ~1m spacing
        road_waypoints = downsample_waypoints(waypoints_qlabs, target_spacing=1.0)
        
        # Find where to cut off the road waypoints and start parking approach
        # We need to stay on the road path longer to avoid cutting the curb
        # Calculate distance along the route, not straight-line distance
        parking_pos = np.array(parking_target[:2])
        
        # Find the point on the road closest to the parking target
        min_distance = float('inf')
        closest_idx = len(road_waypoints) - 1
        
        for i, wp in enumerate(road_waypoints):
            wp_pos = np.array(wp[:2])
            distance = np.linalg.norm(parking_pos - wp_pos)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # Start transition MUCH LATER - keep car on straight road as long as possible
        # Continue well past the closest point before starting the turn
        # This makes the car stay on the centerline longer as shown in the reference image
        waypoints_past_closest = min(15, len(road_waypoints) - closest_idx - 1)
        transition_start_idx = min(closest_idx + waypoints_past_closest, len(road_waypoints) - 1)
        filtered_road_waypoints = road_waypoints[:transition_start_idx + 1]
        
        # If we're too close to the end, use more waypoints
        if len(filtered_road_waypoints) < 10:
            filtered_road_waypoints = road_waypoints
        
        # Generate smooth parking approach
        parking_heading_rad = np.radians(parking_heading)
        
        transition_waypoints = generate_smooth_parking_approach(
            filtered_road_waypoints,
            parking_target, 
            parking_heading_rad,
            approach_distance=approach_distance,
            num_transition_points=10
        )
        
        # Combine road and parking waypoints
        all_waypoints = filtered_road_waypoints[:-1] + transition_waypoints
        
        # Ensure final spacing is maintained
        final_waypoints = downsample_waypoints(all_waypoints, target_spacing=1.0)
        
        # Make sure we end at the exact parking target
        if final_waypoints[-1] != parking_target:
            # Check if last waypoint is very close to target
            last_wp = np.array(final_waypoints[-1][:2])
            target_pos = np.array(parking_target[:2])
            if np.linalg.norm(target_pos - last_wp) < 0.5:
                # Replace last waypoint with exact target
                final_waypoints[-1] = parking_target
            else:
                # Append the target
                final_waypoints.append(parking_target)
        
        # Get spawn location and heading from first waypoint
        spawn_location = final_waypoints[0]
        start_node_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
        spawn_heading = start_node_pose[2]
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
            "parking_target": parking_target,
            "parking_heading": parking_heading,
            "waypoints": final_waypoints,
            "spawn_location": spawn_location,
            "spawn_rotation": spawn_rotation,
            "num_waypoints": len(final_waypoints),
            "total_distance": float(total_distance)
        }
        
        return route_data
        
    except Exception as e:
        print(f"ERROR generating parking route {route_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("PARKING ROUTE GENERATION - QLabs Cityscape")
    print("="*80)
    
    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    
    # Define parking spot behind spot4
    # From reference table:
    # Road Parking 1 (spot4): X=-13.093, Y=-7.572, Yaw=-42°
    # Road Parking 2 (target): X=-18.078, Y=-2.879, Yaw=-42°
    # This is the parking spot behind/adjacent to spot4
    
    spot4_pos = np.array([-13.093, -7.572])
    spot4_heading_deg = -42.0
    
    # Target parking spot (Road Parking 2 from the reference table)
    parking_target = [-18.078, -2.879, 0.005]
    parking_heading_deg = -42.0
    
    print(f"\nParking Spot Configuration:")
    print(f"  Spot4 vehicle (Road Parking 1): {spot4_pos}, heading: {spot4_heading_deg}°")
    print(f"  Target parking spot (Road Parking 2): [{parking_target[0]:.3f}, {parking_target[1]:.3f}], heading: {parking_heading_deg}°")
    
    offset = np.array(parking_target[:2]) - spot4_pos
    offset_distance = np.linalg.norm(offset)
    print(f"  Distance from spot4: {offset_distance:.2f}m")
    
    # Generate route: nodes 18 -> 11 -> 12 -> 8, then smooth parking approach
    node_sequence = [18, 11, 12, 8]
    route_name = "kink_street"
    
    print(f"\nGenerating parking route: {route_name}")
    print(f"  Road nodes: {node_sequence}")
    print(f"  Parking target: {parking_target}")
    
    route_data = generate_parking_route(
        roadmap, 
        node_sequence, 
        parking_target, 
        parking_heading_deg,
        route_name,
        approach_distance=12.0  # Start transitioning 12m before parking spot for smoother curve
    )
    
    if route_data:
        # Save to JSON file
        routes_dir = os.path.join(parent_dir, "config", "routes")
        os.makedirs(routes_dir, exist_ok=True)
        
        filename = f"{route_name}.json"
        filepath = os.path.join(routes_dir, filename)
        
        # Backup existing file if it exists
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            os.rename(filepath, backup_path)
            print(f"  Backed up existing route to: {backup_path}")
        
        with open(filepath, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        print(f"  ✓ Saved: {filename}")
        print(f"    Waypoints: {route_data['num_waypoints']}")
        print(f"    Distance: {route_data['total_distance']:.1f}m")
        
        # Also save a debug preview
        debug_dir = os.path.join(parent_dir, "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        preview_path = os.path.join(debug_dir, f"{route_name}_preview.json")
        
        preview_data = {
            "waypoints": route_data['waypoints'],
            "spawn_location": route_data['spawn_location'],
            "spawn_rotation": route_data['spawn_rotation'],
            "parking_target": route_data['parking_target'],
            "parking_heading": route_data['parking_heading']
        }
        
        with open(preview_path, 'w') as f:
            json.dump(preview_data, f, indent=2)
        
        print(f"  ✓ Saved preview: {preview_path}")
        
        print("\n" + "="*80)
        print("SUCCESS")
        print("="*80)
        print(f"Route saved to: {filepath}")
        print(f"Preview saved to: {preview_path}")
        print("\nYou can now test the route by loading it in your simulation.")
        print("="*80)
        
        return 0
    else:
        print("\n" + "="*80)
        print("FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
