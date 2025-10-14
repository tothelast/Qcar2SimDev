#!/usr/bin/env python3
"""
Fix route coordinates to match actual QLabs Cityscape roads.

This script:
1. Loads SDCSRoadMap to get actual road network
2. Generates a proper route on real roads using correct coordinate scaling
3. Determines correct spawn location
4. Outputs configuration for config.py

COORDINATE SCALING:
- SDCSRoadMap applies internal scaling: scale * (X_mm - offset)
- QLabs uses additional 10x scaling (NO flip):
  * QLabs_X = SDCSRoadMap_X × 10
  * QLabs_Y = SDCSRoadMap_Y × 10

Example from Quanser repository:
- Node 10 from SDCSRoadMap: [-1.28205, -0.45991]
- Spawn in QLabs: [-12.8205, -4.5991] = Node 10 × 10
"""

import sys
import os
import numpy as np

# Add python directory to path for HAL library
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python'))

from hal.products.mats import SDCSRoadMap


def analyze_coordinate_system():
    """Analyze SDCSRoadMap coordinate system and scaling."""
    print("="*80)
    print("COORDINATE SYSTEM ANALYSIS")
    print("="*80)

    # SDCSRoadMap applies its own scaling internally
    # From mats.py:
    #   scale = 0.002035
    #   xOffset = 1134
    #   yOffset = 2363
    #   X_scaled = scale * (X_mm - xOffset)
    #   Y_scaled = scale * (yOffset - Y_mm)

    scale = 0.002035
    xOffset = 1134
    yOffset = 2363

    # Raw node positions (before scaling)
    raw_nodes = {
        0: [1134, 2299, -np.pi/2],
        1: [1266, 2323, np.pi/2],
        11: [1134, 1300, -np.pi/2],
        12: [1134, 1454, -np.pi/2],
        13: [1266, 1454, np.pi/2],
    }

    print("\nNode Positions (Raw mm → Scaled QLabs):")
    print("-" * 80)

    for node_id, (x_mm, y_mm, heading) in raw_nodes.items():
        x_scaled = scale * (x_mm - xOffset)
        y_scaled = scale * (yOffset - y_mm)
        print(f"Node {node_id:2d}: [{x_mm:4d}, {y_mm:4d}] mm → [{x_scaled:7.3f}, {y_scaled:7.3f}] QLabs | Heading: {np.degrees(heading):6.1f}°")

    print("\n" + "="*80)
    print("QLABS COORDINATE SCALING")
    print("="*80)
    print("Formula: QLabs_X = SDCSRoadMap_X × 10")
    print("         QLabs_Y = SDCSRoadMap_Y × 10")
    print("\nVerification with Node 10 (from Quanser example):")
    print(f"  SDCSRoadMap: [-1.28205, -0.45991]")
    print(f"  QLabs:       [-12.8205, -4.5991] = Node 10 × 10 ✓")
    print("="*80)

    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    return roadmap


def create_roundabout_route(roadmap):
    """Create roundabout route: Node 13 → 19 → 17 → 20 → 22."""
    print("\n" + "="*80)
    print("CREATING ROUTE: Node 13 → 19 → 17 → 20 → 22 (Roundabout Navigation)")
    print("="*80)

    # Route through roundabout
    # Node 13: Start position
    # Node 19: Approach roundabout
    # Node 17: Navigate through roundabout
    # Node 20: Exit roundabout
    # Node 22: Final destination

    node_sequence = [13, 19, 17, 20, 22]
    print(f"\nNode sequence: {node_sequence}")

    print("\nNode positions (from SDCSRoadMap):")
    for node_id in node_sequence:
        pose = roadmap.get_node_pose(node_id).squeeze()
        x_qlabs = pose[0] * 10
        y_qlabs = pose[1] * 10
        print(f"  Node {node_id:2d}: SDCSRoadMap [{pose[0]:8.5f}, {pose[1]:8.5f}] → QLabs [{x_qlabs:8.3f}, {y_qlabs:8.3f}] | Heading: {np.degrees(pose[2]):6.1f}°")
    
    # Generate path
    try:
        waypoints_scaled = roadmap.generate_path(node_sequence)
        # waypoints_scaled shape: (2, N) where row 0 = X coords, row 1 = Y coords
        print(f"\nGenerated {waypoints_scaled.shape[1]} waypoints from SDCSRoadMap")

        # Transpose to get [[x1, y1], [x2, y2], ...]
        waypoints_sdcs = waypoints_scaled.T

        # Apply QLabs coordinate scaling:
        # QLabs_X = SDCSRoadMap_X × 10
        # QLabs_Y = SDCSRoadMap_Y × 10
        waypoints_qlabs = []
        for wp in waypoints_sdcs:
            x_qlabs = wp[0] * 10.0
            y_qlabs = wp[1] * 10.0
            waypoints_qlabs.append([x_qlabs, y_qlabs, 0.0])

        # Print first and last few waypoints
        print("\nFirst 5 waypoints (QLabs coordinates):")
        for i, wp in enumerate(waypoints_qlabs[:5]):
            print(f"  [{i}]: [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}]")

        if len(waypoints_qlabs) > 10:
            print(f"\n... ({len(waypoints_qlabs) - 10} waypoints omitted) ...")

            print("\nLast 5 waypoints (QLabs coordinates):")
            for i, wp in enumerate(waypoints_qlabs[-5:], start=len(waypoints_qlabs)-5):
                print(f"  [{i}]: [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}]")

        # Calculate total distance
        total_dist = 0
        for i in range(len(waypoints_qlabs) - 1):
            dx = waypoints_qlabs[i+1][0] - waypoints_qlabs[i][0]
            dy = waypoints_qlabs[i+1][1] - waypoints_qlabs[i][1]
            total_dist += np.sqrt(dx**2 + dy**2)

        print(f"\nTotal route distance: {total_dist:.1f} meters")

        # Downsample to ~1m spacing to match SimLingo training
        # SimLingo uses CARLA GlobalRoutePlanner with hop_resolution=1.0
        # which creates waypoints ~1m apart for target point selection
        target_spacing = 1.0
        downsampled = downsample_waypoints(waypoints_qlabs, target_spacing=target_spacing)
        print(f"Downsampled to {len(downsampled)} waypoints (~{target_spacing}m spacing)")

        # Determine spawn location (start of route)
        spawn_location = downsampled[0]
        start_node_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()
        spawn_heading = start_node_pose[2]  # Heading of starting node
        
        print("\n" + "="*80)
        print("SPAWN CONFIGURATION")
        print("="*80)
        print(f"Location: [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}, 0.005]")
        print(f"Heading: {spawn_heading:.4f} rad = {np.degrees(spawn_heading):.1f}°")
        print(f"Rotation: [0.0, 0.0, {spawn_heading:.4f}]")
        
        return downsampled, spawn_location, spawn_heading
        
    except Exception as e:
        print(f"\nERROR generating path: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def downsample_waypoints(waypoints, target_spacing=2.0):
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


def print_config_output(waypoints, spawn_location, spawn_heading):
    """Print configuration for config.py."""
    print("\n" + "="*80)
    print("CONFIGURATION FOR config.py")
    print("="*80)
    
    print("\n# Spawn Configuration")
    print(f"self.qcar2_spawn_location = [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}, 0.005]")
    print(f"self.qcar2_spawn_rotation = [0.0, 0.0, {spawn_heading:.4f}]  # {np.degrees(spawn_heading):.1f}°")
    
    print("\n# Route Waypoints")
    print("self.route_waypoints = [")
    for i, wp in enumerate(waypoints):
        comment = ""
        if i == 0:
            comment = "  # Start (spawn location)"
        elif i == len(waypoints) - 1:
            comment = "  # End"
        elif i % 5 == 0:
            comment = f"  # Waypoint {i}"
        print(f"    [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}],{comment}")
    print("]")
    
    print("\n# Lookahead Distance")
    total_dist = 0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        total_dist += np.sqrt(dx**2 + dy**2)
    
    recommended_lookahead = min(5.0, total_dist / 3)
    print(f"self.target_point_lookahead = {recommended_lookahead:.1f}  # meters")
    print(f"# (Route length: {total_dist:.1f}m, lookahead should be < {total_dist/2:.1f}m)")


def main():
    print("\n" + "="*80)
    print("FIX ROUTE COORDINATES - QLabs Cityscape")
    print("="*80)
    print("\nThis script generates a proper route on actual QLabs roads")
    print("using the SDCSRoadMap road network.")
    
    # Analyze coordinate system
    roadmap = analyze_coordinate_system()
    
    # Create roundabout route (Node 13 → 19 → 17 → 20 → 22)
    waypoints, spawn_location, spawn_heading = create_roundabout_route(roadmap)
    
    if waypoints is not None:
        # Print configuration
        print_config_output(waypoints, spawn_location, spawn_heading)
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Copy the configuration above into src/config.py")
        print("2. Update the spawn location and rotation")
        print("3. Update the route waypoints")
        print("4. Update the lookahead distance")
        print("5. Run: python src/main.py")
        print("6. Verify vehicle stays on road and follows route")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ERROR: Failed to generate route")
        print("="*80)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

