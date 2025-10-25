#!/usr/bin/env python3
"""
Visualize the kink_street route with parking spot to help with adjustment.
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))


def plot_route():
    """Plot the current kink_street route with parking information."""
    
    # Load route
    route_path = os.path.join(parent_dir, "config", "routes", "kink_street.json")
    
    if not os.path.exists(route_path):
        print(f"Route file not found: {route_path}")
        return
    
    with open(route_path, 'r') as f:
        route_data = json.load(f)
    
    waypoints = np.array(route_data['waypoints'])
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot route
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=2, label='Route', alpha=0.7)
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b.', markersize=4, alpha=0.5)
    
    # Mark start and end
    plt.plot(waypoints[0, 0], waypoints[0, 1], 'go', markersize=15, label='Start', zorder=5)
    plt.plot(waypoints[-1, 0], waypoints[-1, 1], 'ro', markersize=15, label='End/Parking', zorder=5)
    
    # Plot spot4 parked vehicle
    spot4_pos = np.array([-13.0, -7.5])
    spot4_heading = -40.0  # degrees
    
    # Draw car rectangle
    car_length = 2.0
    car_width = 1.0
    
    # Car corners in local frame
    corners_local = np.array([
        [-car_length/2, -car_width/2],
        [car_length/2, -car_width/2],
        [car_length/2, car_width/2],
        [-car_length/2, car_width/2],
        [-car_length/2, -car_width/2]
    ])
    
    # Rotate and translate
    heading_rad = np.radians(spot4_heading)
    rotation = np.array([
        [np.cos(heading_rad), -np.sin(heading_rad)],
        [np.sin(heading_rad), np.cos(heading_rad)]
    ])
    
    corners_world = (rotation @ corners_local.T).T + spot4_pos
    
    plt.plot(corners_world[:, 0], corners_world[:, 1], 'k-', linewidth=2, label='Spot4 Vehicle')
    plt.fill(corners_world[:, 0], corners_world[:, 1], color='gray', alpha=0.5)
    
    # Draw heading arrow
    arrow_length = 1.5
    arrow_end = spot4_pos + arrow_length * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    plt.arrow(spot4_pos[0], spot4_pos[1], 
              arrow_end[0] - spot4_pos[0], arrow_end[1] - spot4_pos[1],
              head_width=0.3, head_length=0.3, fc='black', ec='black')
    
    # Annotate parking target
    if 'parking_target' in route_data:
        target = route_data['parking_target']
        plt.plot(target[0], target[1], 'r*', markersize=20, label='Parking Target', zorder=6)
        
        # Draw heading arrow for target
        target_heading_rad = np.radians(route_data['parking_heading'])
        target_arrow_end = np.array(target[:2]) + arrow_length * np.array([
            np.cos(target_heading_rad), 
            np.sin(target_heading_rad)
        ])
        plt.arrow(target[0], target[1], 
                  target_arrow_end[0] - target[0], target_arrow_end[1] - target[1],
                  head_width=0.3, head_length=0.3, fc='red', ec='red', linestyle='--')
    
    # Add waypoint numbers for last 10 waypoints
    for i in range(max(0, len(waypoints) - 10), len(waypoints)):
        plt.text(waypoints[i, 0] + 0.3, waypoints[i, 1] + 0.3, str(i), 
                fontsize=8, color='blue', alpha=0.7)
    
    # Formatting
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title('Kink Street Route - Parking Approach', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    
    # Add info text
    info_text = f"Total waypoints: {len(waypoints)}\n"
    info_text += f"Total distance: {route_data.get('total_distance', 0):.1f}m\n"
    if 'parking_target' in route_data:
        target = route_data['parking_target']
        info_text += f"Parking: [{target[0]:.2f}, {target[1]:.2f}] @ {route_data['parking_heading']}°"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(parent_dir, "debug_output", "kink_street_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    # Show figure
    plt.show()


def main():
    print("="*80)
    print("ROUTE VISUALIZATION")
    print("="*80)
    
    try:
        plot_route()
        print("\n✓ Visualization complete")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
