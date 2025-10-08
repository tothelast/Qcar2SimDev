#!/usr/bin/env python3
"""
Visualize the planned route waypoints.
Use this to verify the route looks correct before running the vehicle.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SimlingoQCar2Config


def visualize_route():
    """Visualize the planned route from config."""
    # Load config
    config = SimlingoQCar2Config()
    
    # Get route waypoints
    route_waypoints = np.array(config.route_waypoints)
    spawn_location = np.array(config.qcar2_spawn_location)
    spawn_rotation = np.array(config.qcar2_spawn_rotation)
    
    # Calculate route statistics
    total_distance = 0
    for i in range(len(route_waypoints) - 1):
        dx = route_waypoints[i+1, 0] - route_waypoints[i, 0]
        dy = route_waypoints[i+1, 1] - route_waypoints[i, 1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title('Planned Route Waypoints', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters, east)', fontsize=12)
    ax.set_ylabel('Y (meters, north)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
    
    # Plot route
    ax.plot(route_waypoints[:, 0], route_waypoints[:, 1], 'b-', linewidth=3,
            label='Planned Route', alpha=0.7)
    ax.plot(route_waypoints[:, 0], route_waypoints[:, 1], 'bo', markersize=8,
            alpha=0.6)
    
    # Add waypoint numbers
    for i, wp in enumerate(route_waypoints):
        if i % 3 == 0 or i == 0 or i == len(route_waypoints) - 1:
            ax.annotate(f'{i}', (wp[0], wp[1]), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='blue', alpha=0.7)
    
    # Plot spawn location
    ax.plot(spawn_location[0], spawn_location[1], 'go', markersize=15,
            label='Spawn', zorder=5)
    
    # Draw spawn heading arrow
    spawn_yaw = spawn_rotation[2]
    arrow_length = 1.0
    dx = arrow_length * np.cos(spawn_yaw)
    dy = arrow_length * np.sin(spawn_yaw)
    ax.arrow(spawn_location[0], spawn_location[1], dx, dy,
             head_width=0.3, head_length=0.2, fc='green', ec='green',
             alpha=0.7, linewidth=2, zorder=5)
    
    # Mark start and end
    ax.plot(route_waypoints[0, 0], route_waypoints[0, 1], 'g^',
            markersize=15, label='Start', zorder=5)
    ax.plot(route_waypoints[-1, 0], route_waypoints[-1, 1], 'rs',
            markersize=15, label='End', zorder=5)
    
    # Add statistics text
    stats_text = f"""Route Statistics:
Total Waypoints: {len(route_waypoints)}
Total Distance: {total_distance:.1f} m
Start: [{route_waypoints[0, 0]:.1f}, {route_waypoints[0, 1]:.1f}]
End: [{route_waypoints[-1, 0]:.1f}, {route_waypoints[-1, 1]:.1f}]
Spawn Heading: {np.degrees(spawn_yaw):.0f}°"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper right', fontsize=12)
    ax.set_aspect('equal')
    
    # Save figure
    save_path = "debug_output/route_visualization.png"
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nRoute visualization saved to {save_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ROUTE SUMMARY")
    print("="*80)
    print(f"Total Waypoints: {len(route_waypoints)}")
    print(f"Total Distance: {total_distance:.1f} meters")
    print(f"Start: [{route_waypoints[0, 0]:.3f}, {route_waypoints[0, 1]:.3f}]")
    print(f"End: [{route_waypoints[-1, 0]:.3f}, {route_waypoints[-1, 1]:.3f}]")
    print(f"Spawn Location: [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}]")
    print(f"Spawn Heading: {np.degrees(spawn_yaw):.1f}° (0°=East, 90°=North)")
    print("="*80)
    
    plt.show()


if __name__ == "__main__":
    visualize_route()

