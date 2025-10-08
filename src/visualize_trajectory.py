#!/usr/bin/env python3
"""
Visualize vehicle trajectory vs expected route.
Run after executing main.py to analyze route following performance.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# Configuration
# ============================================================================
# Success threshold: Maximum acceptable lateral deviation from planned route (meters)
# Points within this threshold are considered "successful"
# Default: 1.0 meter (typical lane width is ~3.5m, so 1.0m allows reasonable deviation)
SUCCESS_THRESHOLD_METERS = 1.0


def load_trajectory_log(filename=None):
    """Load trajectory log from file."""
    if filename is None:
        filename = "debug_output/trajectory_log_latest.json"
    
    if not os.path.exists(filename):
        print(f"ERROR: Trajectory log not found: {filename}")
        print("Please run src/main.py first to generate trajectory data.")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


def calculate_lateral_deviation(trajectory, route_waypoints):
    """
    Calculate perpendicular distance from trajectory to route.
    
    Args:
        trajectory: List of trajectory entries
        route_waypoints: Array of route waypoints [N, 3]
    
    Returns:
        Array of lateral deviations (meters)
    """
    deviations = []
    
    for entry in trajectory:
        pos = np.array(entry['position'][:2])  # [x, y]
        
        # Find closest segment on route
        min_dist = float('inf')
        
        for i in range(len(route_waypoints) - 1):
            p1 = route_waypoints[i, :2]
            p2 = route_waypoints[i+1, :2]
            
            # Calculate perpendicular distance to line segment
            segment = p2 - p1
            segment_len = np.linalg.norm(segment)
            
            if segment_len < 1e-6:
                # Degenerate segment
                dist = np.linalg.norm(pos - p1)
            else:
                # Project point onto line
                t = np.dot(pos - p1, segment) / (segment_len ** 2)
                t = np.clip(t, 0, 1)  # Clamp to segment
                projection = p1 + t * segment
                dist = np.linalg.norm(pos - projection)
            
            min_dist = min(min_dist, dist)
        
        deviations.append(min_dist)
    
    return np.array(deviations)


def calculate_success_rate(lateral_deviations, threshold=SUCCESS_THRESHOLD_METERS):
    """
    Calculate success rate based on lateral deviation threshold.

    Args:
        lateral_deviations: Array of lateral deviations (meters)
        threshold: Maximum acceptable deviation (meters)

    Returns:
        Tuple of (success_rate_percentage, successful_points, total_points)
    """
    total_points = len(lateral_deviations)
    successful_points = np.sum(lateral_deviations <= threshold)
    success_rate = (successful_points / total_points * 100) if total_points > 0 else 0.0

    return success_rate, successful_points, total_points


def visualize_trajectory(data, save_path=None):
    """
    Create comprehensive visualization of trajectory vs route.
    
    Args:
        data: Trajectory log data
        save_path: Path to save figure (optional)
    """
    metadata = data['metadata']
    trajectory = data['trajectory']
    route_waypoints = np.array(metadata['route_waypoints'])
    spawn_location = np.array(metadata['spawn_location'])
    spawn_rotation = np.array(metadata['spawn_rotation'])
    
    # Extract trajectory data
    positions = np.array([entry['position'] for entry in trajectory])
    headings = np.array([entry['heading_deg'] for entry in trajectory])
    speeds = np.array([entry['speed'] for entry in trajectory])
    steerings = np.array([entry['steering'] for entry in trajectory])
    timestamps = np.array([entry['timestamp'] for entry in trajectory])
    collisions = np.array([entry['collision'] for entry in trajectory])
    
    # Calculate metrics
    lateral_deviations = calculate_lateral_deviation(trajectory, route_waypoints)
    total_distance = np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
    success_rate, successful_points, total_points = calculate_success_rate(lateral_deviations)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ========== Plot 1: Full route overview ==========
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_title('Route Following - Expected vs Actual', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters, east)', fontsize=12)
    ax1.set_ylabel('Y (meters, north)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)

    # Plot planned route (blue, thinner, behind)
    ax1.plot(route_waypoints[:, 0], route_waypoints[:, 1], 'b-', linewidth=3,
             label='Planned Route', alpha=0.5, zorder=1)
    ax1.plot(route_waypoints[:, 0], route_waypoints[:, 1], 'bo', markersize=3,
             alpha=0.4, zorder=1)

    # Plot actual trajectory (red, thicker, in front)
    ax1.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=4,
             label='Actual Trajectory', alpha=0.8, zorder=3)

    # Add speed color overlay on trajectory
    points = positions[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdYlGn', linewidth=2, alpha=0.6, zorder=4)
    lc.set_array(speeds)
    lc.set_clim(0, max(speeds.max(), 0.1))
    ax1.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax1, label='Speed (m/s)', pad=0.02)
    
    # Plot spawn location
    ax1.plot(spawn_location[0], spawn_location[1], 'go', markersize=15, 
             label='Spawn', zorder=5)
    
    # Draw spawn heading arrow
    spawn_yaw = spawn_rotation[2]
    arrow_length = 5
    dx = arrow_length * np.cos(spawn_yaw)
    dy = arrow_length * np.sin(spawn_yaw)
    ax1.arrow(spawn_location[0], spawn_location[1], dx, dy, 
              head_width=2, head_length=1, fc='green', ec='green', alpha=0.7, zorder=5)
    
    # Mark collision points
    collision_indices = np.where(collisions)[0]
    if len(collision_indices) > 0:
        collision_positions = positions[collision_indices, :2]
        ax1.scatter(collision_positions[:, 0], collision_positions[:, 1], 
                   c='red', marker='x', s=200, linewidths=3,
                   label=f'Collisions ({len(collision_indices)})', zorder=6)
    
    # Mark start and end
    ax1.plot(positions[0, 0], positions[0, 1], 'g^', markersize=12,
             label='Start', zorder=5)
    ax1.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=12,
             label='End', zorder=5)

    # Position legend at bottom inside plot area to avoid covering trajectory
    ax1.legend(loc='lower center', fontsize=9, ncol=3, framealpha=0.95,
               edgecolor='black', fancybox=True)
    ax1.set_aspect('equal')
    
    # ========== Plot 2: Zoomed start area ==========
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title('Start Area (First 30m)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Find indices for first 30m
    distances_traveled = np.cumsum(np.concatenate([[0], np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)]))
    first_30m_idx = np.where(distances_traveled <= 30)[0]

    if len(first_30m_idx) > 0:
        # Plot route (blue)
        first_segment_route = route_waypoints[:10]
        ax2.plot(first_segment_route[:, 0], first_segment_route[:, 1], 'b-',
                linewidth=3, label='Planned Route', alpha=0.5)
        ax2.plot(first_segment_route[:, 0], first_segment_route[:, 1], 'bo',
                markersize=5, alpha=0.4)

        # Plot actual trajectory (red)
        first_30m_positions = positions[first_30m_idx, :2]
        ax2.plot(first_30m_positions[:, 0], first_30m_positions[:, 1], 'r-',
                linewidth=3, label='Actual Trajectory', alpha=0.8)

        # Mark every 5th point on trajectory
        for i in range(0, len(first_30m_positions), 5):
            ax2.plot(first_30m_positions[i, 0], first_30m_positions[i, 1], 'ro',
                    markersize=4, alpha=0.6)

        # Plot spawn
        ax2.plot(spawn_location[0], spawn_location[1], 'go', markersize=12,
                label='Spawn', zorder=5)
        ax2.arrow(spawn_location[0], spawn_location[1], dx*0.5, dy*0.5,
                 head_width=0.5, head_length=0.3, fc='green', ec='green', alpha=0.7, zorder=5)

        ax2.legend(fontsize=9, loc='best')
        ax2.set_aspect('equal')
    
    # ========== Plot 3: Lateral deviation over distance ==========
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_title('Lateral Deviation from Route', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Distance Traveled (m)', fontsize=10)
    ax3.set_ylabel('Deviation (m)', fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax3.plot(distances_traveled, lateral_deviations, 'b-', linewidth=2)
    ax3.axhline(y=SUCCESS_THRESHOLD_METERS, color='green', linestyle=':', linewidth=2,
               label=f'Success Threshold: {SUCCESS_THRESHOLD_METERS:.1f}m')
    ax3.axhline(y=lateral_deviations.mean(), color='r', linestyle='--',
               label=f'Mean: {lateral_deviations.mean():.2f}m')
    ax3.axhline(y=lateral_deviations.max(), color='orange', linestyle='--',
               label=f'Max: {lateral_deviations.max():.2f}m')
    ax3.legend(fontsize=9)
    
    # ========== Plot 4: Speed over time ==========
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_title('Speed Profile', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Speed (m/s)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    ax4.plot(timestamps, speeds, 'g-', linewidth=2)
    ax4.axhline(y=speeds.mean(), color='r', linestyle='--', 
               label=f'Mean: {speeds.mean():.2f} m/s')
    ax4.legend(fontsize=9)
    
    # ========== Plot 5: Steering over time ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_title('Steering Profile', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Steering Angle', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    ax5.plot(timestamps, steerings, 'b-', linewidth=2)
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.axhline(y=steerings.mean(), color='r', linestyle='--', 
               label=f'Mean: {steerings.mean():.3f}')
    ax5.legend(fontsize=9)
    
    # ========== Plot 6: Summary statistics ==========
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    ax6.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=10, y=0.98)

    stats_text = f"""Steps: {metadata['total_steps']}
Time: {metadata['total_time']:.1f} s
Distance: {total_distance:.1f} m
Collisions: {metadata['collision_count']}

Success: {success_rate:.1f}%
  ({successful_points}/{total_points} pts
   ≤{SUCCESS_THRESHOLD_METERS:.1f}m)

Speed:
  Mean: {speeds.mean():.2f} m/s
  Max: {speeds.max():.2f} m/s

Lateral Dev:
  Mean: {lateral_deviations.mean():.2f} m
  Max: {lateral_deviations.max():.2f} m
  Std: {lateral_deviations.std():.2f} m

Steering:
  Mean: {steerings.mean():.3f}
  Range: [{steerings.min():.2f}, {steerings.max():.2f}]"""

    # Position text centered below title
    ax6.text(0.5, 0.88, stats_text, transform=ax6.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    if save_path is None:
        timestamp = metadata['timestamp']
        save_path = f"debug_output/trajectory_comparison_{timestamp}.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Steps: {metadata['total_steps']}")
    print(f"Total Time: {metadata['total_time']:.1f} seconds")
    print(f"Total Distance Traveled: {total_distance:.1f} meters")
    print(f"Collisions: {metadata['collision_count']}")
    print(f"\nSuccess Rate: {success_rate:.1f}% (within {SUCCESS_THRESHOLD_METERS:.1f}m threshold)")
    print(f"  Successful points: {successful_points}/{total_points}")
    print(f"\nSpeed: Mean={speeds.mean():.2f} m/s, Max={speeds.max():.2f} m/s")
    print(f"Lateral Deviation: Mean={lateral_deviations.mean():.2f} m, Max={lateral_deviations.max():.2f} m")
    print(f"Steering: Mean={steerings.mean():.3f}, Range=[{steerings.min():.3f}, {steerings.max():.3f}]")
    print("="*80)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize vehicle trajectory')
    parser.add_argument('--log', type=str, default=None,
                       help='Path to trajectory log file (default: latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (default: auto-generated)')
    args = parser.parse_args()
    
    # Load trajectory log
    data = load_trajectory_log(args.log)
    if data is None:
        return 1
    
    # Create visualization
    visualize_trajectory(data, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

