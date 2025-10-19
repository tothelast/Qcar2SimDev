#!/usr/bin/env python3
"""
Dedicated visualization for analyzing model waypoint predictions.
Focuses on understanding route and speed waypoint behavior in ego frame.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import argparse

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Visual design constants
COLORS = {
    'planned_route': '#0066CC',      # Bold blue
    'route_waypoints': '#00FF00',    # Bright green
    'speed_waypoints': '#FF8C00',    # Orange
    'target': '#FFD700',             # Gold
    'vehicle': '#FF0000',            # Red
    'grid': '#E0E0E0',               # Light gray
    'good': '#90EE90',               # Light green
    'warning': '#FFE4B5',            # Moccasin
    'bad': '#FFB6C1',                # Light pink
    'text_bg': '#F5F5F5',            # Very light gray
}

SIZES = {
    'route_wp_marker': 8,
    'speed_wp_marker': 10,
    'target_marker': 22,
    'route_line': 3.0,
    'speed_line': 3.5,
    'planned_route_line': 3.5,
    'vehicle_size': 0.8,
    'title_font': 13,
    'label_font': 11,
    'metric_font': 10,
    'legend_font': 9,
    'annotation_font': 8,
}


def load_trajectory_log(filename=None):
    """Load trajectory log from file."""
    if filename is None:
        filename = "debug_output/trajectory_log_latest.json"
    
    if not os.path.exists(filename):
        print(f"ERROR: Trajectory log not found: {filename}")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


def world_to_ego(world_point, vehicle_pos, vehicle_heading):
    """Transform point from world frame to ego frame."""
    translated = world_point[:2] - vehicle_pos[:2]
    
    cos_h = np.cos(-vehicle_heading)
    sin_h = np.sin(-vehicle_heading)
    
    ego_x = cos_h * translated[0] - sin_h * translated[1]
    ego_y = sin_h * translated[0] + cos_h * translated[1]
    
    return np.array([ego_x, ego_y])


def calculate_waypoint_metrics(route_wps, speed_wps, current_speed=0):
    """Calculate metrics from waypoints."""
    metrics = {}

    # Route waypoint metrics
    if route_wps is not None and len(route_wps) > 0:
        # Path length
        route_dists = np.linalg.norm(np.diff(route_wps, axis=0), axis=1)
        metrics['route_length'] = np.sum(route_dists)
        metrics['route_mean_spacing'] = np.mean(route_dists)

        # Direction (angle of first waypoint)
        metrics['route_angle'] = np.degrees(np.arctan2(route_wps[0, 1], route_wps[0, 0]))

        # Curvature (change in direction)
        if len(route_wps) >= 3:
            angles = []
            for i in range(len(route_wps) - 2):
                v1 = route_wps[i+1] - route_wps[i]
                v2 = route_wps[i+2] - route_wps[i+1]
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                angles.append(angle)
            metrics['route_curvature'] = np.degrees(np.mean(np.abs(angles)))

    # Speed waypoint metrics
    if speed_wps is not None and len(speed_wps) >= 4:
        # Predicted speed (using same calculation as controller)
        desired_speed = np.linalg.norm(speed_wps[0] - speed_wps[3]) * 2.0
        metrics['predicted_speed'] = desired_speed
        metrics['speed_error'] = abs(desired_speed - current_speed)

        # Path length
        speed_dists = np.linalg.norm(np.diff(speed_wps, axis=0), axis=1)
        metrics['speed_length'] = np.sum(speed_dists)
        metrics['speed_mean_spacing'] = np.mean(speed_dists)

        # Direction
        metrics['speed_angle'] = np.degrees(np.arctan2(speed_wps[0, 1], speed_wps[0, 0]))

    # Calculate angle difference
    if 'route_angle' in metrics and 'speed_angle' in metrics:
        metrics['angle_diff'] = abs(metrics['route_angle'] - metrics['speed_angle'])

    return metrics


def assess_prediction_quality(metrics):
    """Assess quality of predictions and return status."""
    issues = []
    status = 'good'

    # Check angle difference
    if 'angle_diff' in metrics:
        if metrics['angle_diff'] > 30:
            issues.append('Large angle mismatch')
            status = 'bad'
        elif metrics['angle_diff'] > 15:
            issues.append('Moderate angle mismatch')
            if status == 'good':
                status = 'warning'

    # Check speed error
    if 'speed_error' in metrics:
        if metrics['speed_error'] > 2.0:
            issues.append('Large speed error')
            status = 'bad'
        elif metrics['speed_error'] > 1.0:
            issues.append('Moderate speed error')
            if status == 'good':
                status = 'warning'

    # Check route angle (should generally point forward)
    if 'route_angle' in metrics:
        if abs(metrics['route_angle']) > 45:
            issues.append('Sharp turn predicted')
            if status == 'good':
                status = 'warning'

    return status, issues


def add_coordinate_indicator(ax):
    """Add coordinate system indicator to plot - REMOVED to reduce clutter."""
    # Coordinate system is now clear from axis labels
    pass


def add_status_indicator(ax, status, issues):
    """Add visual status indicator to plot - REMOVED per user request."""
    # Status indicator removed to reduce clutter
    pass


def plot_ego_frame_analysis(ax, entry, route_waypoints, title_suffix=""):
    """Plot single timestep in ego frame with detailed annotations."""
    vehicle_pos = np.array(entry['position'])
    vehicle_heading = entry['heading_rad']
    current_speed = entry['speed']

    # Get predicted waypoints (already in ego frame)
    route_wps = np.array(entry['predicted_route_waypoints']) if entry['predicted_route_waypoints'] else None
    speed_wps = np.array(entry['predicted_speed_waypoints']) if entry['predicted_speed_waypoints'] else None

    # Transform target to ego frame
    target_world = np.array(entry['target_waypoint'])
    target_ego = world_to_ego(target_world, vehicle_pos, vehicle_heading)

    # Transform nearby route waypoints to ego
    route_ego = []
    for rw in route_waypoints:
        rw_ego = world_to_ego(rw, vehicle_pos, vehicle_heading)
        if -10 < rw_ego[0] < 40 and -20 < rw_ego[1] < 20:
            route_ego.append(rw_ego)
    route_ego = np.array(route_ego) if route_ego else None

    # Calculate metrics
    metrics = calculate_waypoint_metrics(route_wps, speed_wps, current_speed)
    status, issues = assess_prediction_quality(metrics)

    # Setup plot with better styling
    ax.set_xlim(-5, 30)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=0.8, color=COLORS['grid'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)
    
    # Draw vehicle at origin with better visibility
    vehicle = Circle((0, 0), SIZES['vehicle_size'], color=COLORS['vehicle'],
                    edgecolor='darkred', linewidth=2, alpha=0.9, zorder=10)
    ax.add_patch(vehicle)
    ax.arrow(0, 0, 3.0, 0, head_width=0.7, head_length=0.5,
            fc=COLORS['vehicle'], ec='darkred', linewidth=2, alpha=0.9, zorder=10)

    # Plot planned route with better visibility
    if route_ego is not None and len(route_ego) > 0:
        ax.plot(route_ego[:, 0], route_ego[:, 1], '-', color=COLORS['planned_route'],
               linewidth=SIZES['planned_route_line'], alpha=0.5, label='Planned Route', zorder=2)
        ax.plot(route_ego[:, 0], route_ego[:, 1], 'o', color=COLORS['planned_route'],
               markersize=4, alpha=0.4, zorder=2)

    # Plot target point with high visibility
    ax.plot(target_ego[0], target_ego[1], '*', color=COLORS['target'],
           markersize=SIZES['target_marker'], markeredgecolor='darkorange',
           markeredgewidth=2, label='Target', zorder=6)
    target_dist = np.linalg.norm(target_ego)
    target_angle = np.degrees(np.arctan2(target_ego[1], target_ego[0]))

    # Plot route waypoints with bright green
    if route_wps is not None and len(route_wps) > 0:
        ax.plot(route_wps[:, 0], route_wps[:, 1], '-', color=COLORS['route_waypoints'],
               linewidth=SIZES['route_line'], alpha=0.8, label='Route (20)', zorder=4)
        ax.plot(route_wps[:, 0], route_wps[:, 1], 'o', color=COLORS['route_waypoints'],
               markersize=SIZES['route_wp_marker'], alpha=0.9, zorder=4,
               markeredgecolor='darkgreen', markeredgewidth=1)

    # Plot speed waypoints with orange
    if speed_wps is not None and len(speed_wps) > 0:
        ax.plot(speed_wps[:, 0], speed_wps[:, 1], '--', color=COLORS['speed_waypoints'],
               linewidth=SIZES['speed_line'], alpha=0.9, label='Speed (10)', zorder=5)
        ax.plot(speed_wps[:, 0], speed_wps[:, 1], 'o', color=COLORS['speed_waypoints'],
               markersize=SIZES['speed_wp_marker'], alpha=0.95, zorder=5,
               markeredgecolor='darkorange', markeredgewidth=1.5)

        # Highlight waypoints used for speed calculation (0 and 3) - simplified
        if len(speed_wps) >= 4:
            ax.plot([speed_wps[0, 0], speed_wps[3, 0]], [speed_wps[0, 1], speed_wps[3, 1]],
                   'r-', linewidth=2.5, alpha=0.6, zorder=5, label='Calc')
    
    # Add compact title with key metrics
    title_text = f'Ego Frame - Step {entry["step"]} {title_suffix}'
    if 'predicted_speed' in metrics:
        title_text += f' | Speed: {current_speed:.1f}→{metrics["predicted_speed"]:.1f} m/s'
    if 'angle_diff' in metrics:
        title_text += f' | Δθ={metrics["angle_diff"]:.1f}°'

    # Styling
    ax.set_xlabel('X (forward, meters)', fontsize=SIZES['label_font'], fontweight='bold')
    ax.set_ylabel('Y (left, meters)', fontsize=SIZES['label_font'], fontweight='bold')
    ax.set_title(title_text, fontsize=SIZES['metric_font'], fontweight='bold', pad=8)

    # Compact legend - only essential items, positioned to avoid overlap
    ax.legend(loc='lower right', fontsize=SIZES['annotation_font'], framealpha=0.9,
             edgecolor='gray', ncol=2)


def plot_temporal_evolution(ax, trajectory, route_waypoints):
    """Plot how predictions evolve over time."""
    steps = []
    timestamps = []
    predicted_speeds = []
    actual_speeds = []
    route_angles = []
    speed_angles = []

    for entry in trajectory:
        if entry['predicted_route_waypoints'] is None or entry['predicted_speed_waypoints'] is None:
            continue

        route_wps = np.array(entry['predicted_route_waypoints'])
        speed_wps = np.array(entry['predicted_speed_waypoints'])

        metrics = calculate_waypoint_metrics(route_wps, speed_wps, entry['speed'])

        steps.append(entry['step'])
        timestamps.append(entry['timestamp'])
        predicted_speeds.append(metrics.get('predicted_speed', 0))
        actual_speeds.append(entry['speed'])
        route_angles.append(metrics.get('route_angle', 0))
        speed_angles.append(metrics.get('speed_angle', 0))

    # Create subplots
    ax1 = ax
    ax2 = ax1.twinx()

    # Plot predicted speed vs actual speed with better visibility
    ax1.plot(timestamps, predicted_speeds, '-', color=COLORS['speed_waypoints'],
            linewidth=3, label='Predicted Speed', alpha=0.8, zorder=3)
    ax1.plot(timestamps, actual_speeds, '-', color='green',
            linewidth=3, label='Actual Speed', alpha=0.8, zorder=3)

    # Add shaded region for "good" speed tracking (within 1 m/s)
    ax1.fill_between(timestamps,
                     np.array(actual_speeds) - 1.0,
                     np.array(actual_speeds) + 1.0,
                     color='green', alpha=0.1, label='±1 m/s range', zorder=1)

    ax1.set_xlabel('Time (s)', fontsize=SIZES['label_font'], fontweight='bold')
    ax1.set_ylabel('Speed (m/s)', fontsize=SIZES['label_font'], fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelsize=SIZES['annotation_font'])
    ax1.tick_params(axis='x', labelsize=SIZES['annotation_font'])
    ax1.grid(True, alpha=0.4, linestyle=':', color=COLORS['grid'])
    ax1.set_facecolor('#FAFAFA')

    # Plot angle difference
    angle_diff = [abs(r - s) for r, s in zip(route_angles, speed_angles)]
    ax2.plot(timestamps, angle_diff, '--', color='red', linewidth=2.5,
            label='Angle Mismatch', alpha=0.7, zorder=2)

    # Add threshold lines
    ax2.axhline(y=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label='Warning (15°)')
    ax2.axhline(y=30, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='Critical (30°)')

    ax2.set_ylabel('Angle Difference (degrees)', fontsize=SIZES['label_font'],
                  fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=SIZES['annotation_font'])

    # No title on plot to avoid overlap - will use axis labels only
    ax1.set_title('')

    # Combine legends - compact, 2 columns
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=SIZES['annotation_font'], framealpha=0.9, edgecolor='gray', ncol=2)


def plot_waypoint_spacing_analysis(ax, trajectory):
    """Analyze waypoint spacing patterns."""
    route_spacings = []
    speed_spacings = []

    for entry in trajectory[::3]:  # Sample every 3rd for better statistics
        if entry['predicted_route_waypoints'] is None or entry['predicted_speed_waypoints'] is None:
            continue

        route_wps = np.array(entry['predicted_route_waypoints'])
        speed_wps = np.array(entry['predicted_speed_waypoints'])

        # Calculate spacing for each waypoint pair
        route_dists = np.linalg.norm(np.diff(route_wps, axis=0), axis=1)
        speed_dists = np.linalg.norm(np.diff(speed_wps, axis=0), axis=1)

        route_spacings.append(route_dists)
        speed_spacings.append(speed_dists)

    # Create enhanced box plots
    if route_spacings and speed_spacings:
        route_flat = np.concatenate(route_spacings)
        speed_flat = np.concatenate(speed_spacings)

        # Create box plots with better styling
        bp = ax.boxplot([route_flat, speed_flat],
                        labels=['Route Waypoints\n(20 points)', 'Speed Waypoints\n(10 points)'],
                        patch_artist=True, widths=0.5,
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(linewidth=3, color='darkred'))

        # Color the boxes with better colors
        bp['boxes'][0].set_facecolor(COLORS['route_waypoints'])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][0].set_edgecolor('darkgreen')

        bp['boxes'][1].set_facecolor(COLORS['speed_waypoints'])
        bp['boxes'][1].set_alpha(0.6)
        bp['boxes'][1].set_edgecolor('darkorange')

        ax.set_ylabel('Waypoint Spacing (meters)', fontsize=SIZES['label_font'], fontweight='bold')
        # No title to avoid overlap
        ax.set_title('')
        ax.grid(True, alpha=0.4, axis='y', linestyle=':', color=COLORS['grid'])
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(labelsize=SIZES['annotation_font'])

        # Add mean lines with better visibility
        route_mean = route_flat.mean()
        speed_mean = speed_flat.mean()

        ax.axhline(y=route_mean, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2.5,
                  label=f'Route mean: {route_mean:.2f}m')
        ax.axhline(y=speed_mean, color='darkorange', linestyle='--', alpha=0.7, linewidth=2.5,
                  label=f'Speed mean: {speed_mean:.2f}m')

        # Add reference lines for ideal spacing
        ax.axhline(y=1.5, color='gray', linestyle=':', alpha=0.4, linewidth=1.5,
                  label='Ideal: ~1.5m')

        # Compact legend
        ax.legend(fontsize=SIZES['annotation_font'], loc='upper left', framealpha=0.9,
                 edgecolor='gray', ncol=2)


def visualize_waypoint_analysis(data, save_path=None):
    """Create comprehensive waypoint analysis visualization."""
    metadata = data['metadata']
    trajectory = data['trajectory']
    route_waypoints = np.array(metadata['route_waypoints'])

    # Select sample timesteps for detailed analysis
    total_steps = len(trajectory)
    sample_indices = [
        0,                          # Start
        total_steps // 4,           # 25%
        total_steps // 2,           # 50%
        3 * total_steps // 4,       # 75%
    ]

    # Create figure with improved layout - removed spacing panel
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('white')

    # Use GridSpec with more spacing to prevent overlap
    # 3 rows: 2 rows for 4 ego frames + 1 row for temporal evolution panel
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.25,
                         height_ratios=[1, 1, 0.7],
                         top=0.93, bottom=0.06, left=0.07, right=0.95)

    # Plot ego frame analysis for each sample (top 2 rows)
    for idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(trajectory):
            continue

        entry = trajectory[sample_idx]
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        progress = (sample_idx / total_steps) * 100
        plot_ego_frame_analysis(ax, entry, route_waypoints, f"({progress:.0f}%)")

    # Bottom row: temporal evolution only (spans both columns)
    ax_temporal = fig.add_subplot(gs[2, :])
    plot_temporal_evolution(ax_temporal, trajectory, route_waypoints)

    # Add overall title - simplified to avoid overlap
    title = f'Waypoint Analysis: {metadata["timestamp"]} ({total_steps} steps)'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.97)

    # Save figure with high quality
    if save_path is None:
        timestamp = metadata['timestamp']
        save_path = f"debug_output/waypoint_analysis_{timestamp}.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n✅ Waypoint analysis visualization saved to: {save_path}")
    print(f"   Resolution: 150 DPI | Size: {fig.get_size_inches()[0]:.1f}\" × {fig.get_size_inches()[1]:.1f}\"")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze model waypoint predictions')
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
    visualize_waypoint_analysis(data, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

