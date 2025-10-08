#!/usr/bin/env python3
"""
Create a simple, short test route for visual verification.

This script creates a straight-line route starting from our spawn location
[0, -1.3] heading 90° (facing +Y/North) and going straight for ~40 meters.

This allows us to visually verify that the waypoints are correctly placed
on the roads in QLabs Cityscape Lite.
"""

import numpy as np


def create_straight_test_route():
    """
    Create a simple straight-line test route.
    
    Starting from spawn [0, -1.3], go straight north (+Y direction) for ~40m.
    Waypoints spaced every 2 meters for smooth control.
    """
    # Spawn location
    spawn_x = 0.0
    spawn_y = -1.3
    
    # Create waypoints going straight north
    # From Y=-1.3 to Y=40 (total distance ~41 meters)
    waypoints = []
    
    # Spacing: 2 meters
    spacing = 2.0
    end_y = 40.0
    
    y = spawn_y
    while y <= end_y:
        waypoints.append([spawn_x, y, 0.0])
        y += spacing
    
    # Add final waypoint
    if waypoints[-1][1] < end_y:
        waypoints.append([spawn_x, end_y, 0.0])
    
    return waypoints


def print_route_for_config(waypoints):
    """Print waypoints in the format needed for config.py"""
    print("="*80)
    print("Simple Straight Test Route")
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
        print(f"    [{wp[0]:7.3f}, {wp[1]:7.3f}, {wp[2]:.1f}],{comment}")
    print("]")


if __name__ == "__main__":
    waypoints = create_straight_test_route()
    print_route_for_config(waypoints)

