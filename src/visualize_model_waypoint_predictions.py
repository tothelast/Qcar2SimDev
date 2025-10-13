#!/usr/bin/env python3
"""
Visualize diagnostic results to identify model prediction issues.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def ego_to_world(ego_point, vehicle_pos, vehicle_heading):
    """Transform point from ego frame to world frame."""
    cos_h = np.cos(vehicle_heading)
    sin_h = np.sin(vehicle_heading)
    
    world_x = cos_h * ego_point[0] - sin_h * ego_point[1]
    world_y = sin_h * ego_point[0] + cos_h * ego_point[1]
    
    return np.array([world_x + vehicle_pos[0], world_y + vehicle_pos[1]])


def world_to_ego(world_point, vehicle_pos, vehicle_heading):
    """Transform point from world frame to ego frame."""
    translated = world_point[:2] - vehicle_pos[:2]
    
    cos_h = np.cos(-vehicle_heading)
    sin_h = np.sin(-vehicle_heading)
    
    ego_x = cos_h * translated[0] - sin_h * translated[1]
    ego_y = sin_h * translated[0] + cos_h * translated[1]
    
    return np.array([ego_x, ego_y])


def visualize_sample_predictions(log_file, output_file='debug_output/diagnostic_visualization.png'):
    """Visualize sample predictions in ego frame to diagnose issues."""
    
    # Load data
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    trajectory = data['trajectory']
    route_waypoints = np.array(data['metadata']['route_waypoints'])
    
    # Select sample timesteps
    sample_indices = [
        len(trajectory) // 4,      # 25%
        len(trajectory) // 2,      # 50%
        3 * len(trajectory) // 4,  # 75%
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Predictions in Ego Frame (Vehicle-Centric View)', fontsize=16, fontweight='bold')
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]
        entry = trajectory[sample_idx]
        
        vehicle_pos = np.array(entry['position'])
        vehicle_heading = entry['heading_rad']
        target_world = np.array(entry['target_waypoint'])
        
        # Transform target to ego frame
        target_ego = world_to_ego(target_world, vehicle_pos, vehicle_heading)
        
        # Get predicted waypoints (already in ego frame)
        if entry['predicted_route_waypoints'] is not None:
            pred_wps_ego = np.array(entry['predicted_route_waypoints'])
        else:
            pred_wps_ego = None
        
        # Transform nearby route waypoints to ego frame
        route_ego = []
        for rw in route_waypoints:
            rw_ego = world_to_ego(rw, vehicle_pos, vehicle_heading)
            # Only show waypoints within reasonable range
            if -20 < rw_ego[0] < 40 and -20 < rw_ego[1] < 20:
                route_ego.append(rw_ego)
        route_ego = np.array(route_ego)
        
        # Plot setup
        ax.set_xlim(-5, 25)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Draw vehicle (at origin in ego frame)
        vehicle = Circle((0, 0), 0.5, color='red', alpha=0.7, zorder=10)
        ax.add_patch(vehicle)
        
        # Draw vehicle heading (forward direction)
        ax.arrow(0, 0, 2, 0, head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.7, zorder=10)
        ax.text(2.5, 0, 'Vehicle\nForward', fontsize=8, ha='left', va='center')
        
        # Plot planned route (in ego frame)
        if len(route_ego) > 0:
            ax.plot(route_ego[:, 0], route_ego[:, 1], 'bo-', markersize=4, linewidth=2, 
                   alpha=0.6, label='Planned Route', zorder=3)
        
        # Plot target point
        ax.plot(target_ego[0], target_ego[1], 'g*', markersize=15, 
               label=f'Target ({target_ego[0]:.1f}, {target_ego[1]:.1f})m', zorder=5)
        
        # Plot predicted waypoints
        if pred_wps_ego is not None:
            ax.plot(pred_wps_ego[:, 0], pred_wps_ego[:, 1], 'mo-', markersize=6, linewidth=2,
                   alpha=0.7, label='Model Prediction', zorder=4)
            
            # Draw arrow from vehicle to first predicted waypoint
            if len(pred_wps_ego) > 0:
                ax.arrow(0, 0, pred_wps_ego[0, 0], pred_wps_ego[0, 1],
                        head_width=0.4, head_length=0.2, fc='magenta', ec='magenta',
                        alpha=0.5, zorder=4, linestyle='--')
            
            # Calculate angle between first waypoint and target
            if len(pred_wps_ego) > 0:
                wp_dir = pred_wps_ego[0] / (np.linalg.norm(pred_wps_ego[0]) + 1e-6)
                target_dir = target_ego / (np.linalg.norm(target_ego) + 1e-6)
                angle = np.arccos(np.clip(np.dot(wp_dir, target_dir), -1, 1))
                
                ax.text(0.05, 0.95, f'Angle to target: {np.degrees(angle):.1f}°',
                       transform=ax.transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('X (forward, meters)', fontsize=10)
        ax.set_ylabel('Y (left, meters)', fontsize=10)
        ax.set_title(f'Step {entry["step"]} ({entry["timestamp"]:.1f}s)\nSpeed: {entry["speed"]:.2f} m/s', 
                    fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnostic visualization saved to: {output_file}")
    plt.close()


def plot_prediction_statistics(log_file, output_file='debug_output/prediction_statistics.png'):
    """Plot statistics about predictions over time."""
    
    # Load data
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    trajectory = data['trajectory']
    route_waypoints = np.array(data['metadata']['route_waypoints'])
    
    # Collect statistics
    steps = []
    timestamps = []
    target_distances = []
    target_angles = []
    pred_lengths = []
    route_deviations = []
    first_wp_distances = []
    
    for entry in trajectory:
        if entry['predicted_route_waypoints'] is None:
            continue
        
        steps.append(entry['step'])
        timestamps.append(entry['timestamp'])
        
        vehicle_pos = np.array(entry['position'])
        vehicle_heading = entry['heading_rad']
        target_world = np.array(entry['target_waypoint'])
        pred_wps_ego = np.array(entry['predicted_route_waypoints'])
        
        # Target point analysis
        target_ego = world_to_ego(target_world, vehicle_pos, vehicle_heading)
        target_distances.append(np.linalg.norm(target_ego))
        target_angles.append(np.degrees(np.arctan2(target_ego[1], target_ego[0])))
        
        # Prediction length
        dists = np.linalg.norm(np.diff(pred_wps_ego, axis=0), axis=1)
        pred_lengths.append(dists.sum())
        
        # First waypoint distance
        first_wp_distances.append(np.linalg.norm(pred_wps_ego[0]))
        
        # Route deviation
        pred_wps_world = []
        for wp_ego in pred_wps_ego:
            wp_world = ego_to_world(wp_ego, vehicle_pos, vehicle_heading)
            pred_wps_world.append(wp_world)
        pred_wps_world = np.array(pred_wps_world)
        
        # Calculate mean deviation from route
        deviations = []
        for wp in pred_wps_world:
            min_dist = float('inf')
            for j in range(len(route_waypoints) - 1):
                p1 = route_waypoints[j, :2]
                p2 = route_waypoints[j+1, :2]
                segment = p2 - p1
                segment_len = np.linalg.norm(segment)
                
                if segment_len < 1e-6:
                    dist = np.linalg.norm(wp - p1)
                else:
                    t = max(0, min(1, np.dot(wp - p1, segment) / (segment_len ** 2)))
                    projection = p1 + t * segment
                    dist = np.linalg.norm(wp - projection)
                
                min_dist = min(min_dist, dist)
            deviations.append(min_dist)
        route_deviations.append(np.mean(deviations))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Prediction Statistics Over Time', fontsize=16, fontweight='bold')
    
    # Plot 1: Target point distance and angle
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(timestamps, target_distances, 'b-', linewidth=1.5, label='Distance to Target')
    ax1_twin.plot(timestamps, target_angles, 'r-', linewidth=1.5, alpha=0.7, label='Angle to Target')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)', color='b')
    ax1_twin.set_ylabel('Angle (degrees)', color='r')
    ax1.set_title('Target Point Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Prediction trajectory length
    ax2 = axes[0, 1]
    ax2.plot(timestamps, pred_lengths, 'g-', linewidth=1.5)
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Expected ~20-40m')
    ax2.axhline(y=40, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Trajectory Length (m)')
    ax2.set_title('Predicted Trajectory Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Route deviation
    ax3 = axes[1, 0]
    ax3.plot(timestamps, route_deviations, 'm-', linewidth=1.5)
    ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='Good alignment (<3m)')
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Poor alignment (>5m)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mean Deviation (m)')
    ax3.set_title('Prediction Deviation from Planned Route')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: First waypoint distance
    ax4 = axes[1, 1]
    ax4.plot(timestamps, first_wp_distances, 'c-', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Distance to First Predicted Waypoint')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Statistics visualization saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model prediction diagnostics')
    parser.add_argument('--log', type=str, default='debug_output/trajectory_log_latest.json',
                        help='Path to trajectory log file')
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING DIAGNOSTIC VISUALIZATIONS")
    print("="*80)
    
    visualize_sample_predictions(args.log)
    plot_prediction_statistics(args.log)
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()

