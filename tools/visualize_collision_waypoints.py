#!/usr/bin/env python3
"""
Visualize model waypoint predictions during the segment before a collision.

Loads a trajectory log (with predicted waypoints) from a test run,
identifies the pre-collision segment, and creates multi-panel ego-frame
BEV plots at selected timesteps before collision.

Usage:
    python tools/visualize_collision_waypoints.py \
        --log results/runs/obstacle_var2_run_1/trajectory_log.json
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Reuse visual constants from visualize_waypoints.py
COLORS = {
    'planned_route': '#0066CC',
    'route_waypoints': '#00FF00',
    'speed_waypoints': '#FF8C00',
    'target': '#FFD700',
    'vehicle': '#FF0000',
    'obstacle': '#CC0000',
    'trajectory': '#8888FF',
    'grid': '#E0E0E0',
    'collision_zone': '#FFCCCC',
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


def load_trajectory_log(filename):
    """Load trajectory log from file."""
    if not os.path.exists(filename):
        print(f"ERROR: Trajectory log not found: {filename}")
        return None
    with open(filename, 'r') as f:
        return json.load(f)


def world_to_ego(world_point, vehicle_pos, vehicle_heading):
    """Transform point from world frame to ego frame."""
    translated = world_point[:2] - vehicle_pos[:2]
    cos_h = np.cos(-vehicle_heading)
    sin_h = np.sin(-vehicle_heading)
    ego_x = cos_h * translated[0] - sin_h * translated[1]
    ego_y = sin_h * translated[0] + cos_h * translated[1]
    return np.array([ego_x, ego_y])


def find_collision_segment(trajectory, pre_steps=15, obstacle_location=None,
                           obstacle_threshold=5.0):
    """Find the first collision with the obstacle and return the pre-collision segment.

    If obstacle_location is provided, only collisions within obstacle_threshold
    metres of the obstacle are considered (ignoring curb collisions).
    """
    first_collision_idx = None
    for i, entry in enumerate(trajectory):
        if not entry.get('collision', False):
            continue

        if obstacle_location is not None:
            pos = np.array(entry['position'][:2])
            obs = np.array(obstacle_location[:2])
            dist = np.linalg.norm(pos - obs)
            if dist > obstacle_threshold:
                continue  # curb collision, skip

        first_collision_idx = i
        break

    if first_collision_idx is None:
        print("WARNING: No obstacle collision found in trajectory log.")
        return None, None

    start_idx = max(0, first_collision_idx - pre_steps)
    return start_idx, first_collision_idx


def find_closest_approach_segment(trajectory, obstacle_location, pre_steps=15):
    """Find the segment around the closest approach to the obstacle."""
    obs = np.array(obstacle_location[:2])
    dists = [np.linalg.norm(np.array(e['position'][:2]) - obs) for e in trajectory]
    closest_idx = int(np.argmin(dists))
    min_dist = dists[closest_idx]

    start_idx = max(0, closest_idx - pre_steps)
    end_idx = min(len(trajectory) - 1, closest_idx)

    return start_idx, end_idx, min_dist


def plot_ego_frame_panel(ax, entry, route_waypoints_world, obstacle_world, title_suffix=""):
    """Plot a single timestep in ego frame with detailed waypoint annotations."""
    vehicle_pos = np.array(entry['position'])
    vehicle_heading = entry.get('heading_rad', np.deg2rad(entry['heading_deg']))
    current_speed = entry['speed']

    route_wps = np.array(entry['predicted_route_waypoints']) if entry.get('predicted_route_waypoints') else None
    speed_wps = np.array(entry['predicted_speed_waypoints']) if entry.get('predicted_speed_waypoints') else None

    # Compute predicted speed from speed waypoints (same as controller)
    # Controller: desired_speed = norm(wp[2] - wp[0]) / (2 * dt * data_save_freq)
    # With dt=0.25, data_save_freq=1 → time_delta = 0.5 → multiply by 2.0
    predicted_speed = None
    if speed_wps is not None and len(speed_wps) >= 3:
        predicted_speed = np.linalg.norm(speed_wps[2] - speed_wps[0]) * 2.0

    # Transform obstacle to ego frame
    obstacle_ego = None
    if obstacle_world is not None:
        obstacle_ego = world_to_ego(np.array(obstacle_world), vehicle_pos, vehicle_heading)

    # Transform target waypoint to ego frame
    target_ego = None
    if entry.get('target_waypoint') is not None:
        target_ego = world_to_ego(np.array(entry['target_waypoint']), vehicle_pos, vehicle_heading)

    # Transform nearby route waypoints to ego
    route_ego = []
    if route_waypoints_world is not None:
        for rw in route_waypoints_world:
            rw_ego = world_to_ego(np.array(rw), vehicle_pos, vehicle_heading)
            if -10 < rw_ego[0] < 40 and -20 < rw_ego[1] < 20:
                route_ego.append(rw_ego)
    route_ego = np.array(route_ego) if route_ego else None

    # Setup plot
    ax.set_xlim(-5, 30)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=0.8, color=COLORS['grid'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)

    # Vehicle at origin
    vehicle = Circle((0, 0), SIZES['vehicle_size'], color=COLORS['vehicle'],
                      edgecolor='darkred', linewidth=2, alpha=0.9, zorder=10)
    ax.add_patch(vehicle)
    ax.arrow(0, 0, 3.0, 0, head_width=0.7, head_length=0.5,
             fc=COLORS['vehicle'], ec='darkred', linewidth=2, alpha=0.9, zorder=10)

    # Planned route (ground truth)
    if route_ego is not None and len(route_ego) > 0:
        ax.plot(route_ego[:, 0], route_ego[:, 1], '-', color=COLORS['planned_route'],
                linewidth=SIZES['planned_route_line'], alpha=0.5, zorder=2)
        ax.plot(route_ego[:, 0], route_ego[:, 1], 'o', color=COLORS['planned_route'],
                markersize=4, alpha=0.4, zorder=2)

    # Target waypoint
    if target_ego is not None:
        ax.plot(target_ego[0], target_ego[1], '*', color=COLORS['target'],
                markersize=SIZES['target_marker'], markeredgecolor='darkorange',
                markeredgewidth=2, zorder=6)

    # Predicted route waypoints
    if route_wps is not None and len(route_wps) > 0:
        ax.plot(route_wps[:, 0], route_wps[:, 1], '-', color=COLORS['route_waypoints'],
                linewidth=SIZES['route_line'], alpha=0.8, zorder=4)
        ax.plot(route_wps[:, 0], route_wps[:, 1], 'o', color=COLORS['route_waypoints'],
                markersize=SIZES['route_wp_marker'], alpha=0.9, zorder=4,
                markeredgecolor='darkgreen', markeredgewidth=1)

    # Predicted speed waypoints
    if speed_wps is not None and len(speed_wps) > 0:
        spread = np.linalg.norm(speed_wps[0] - speed_wps[-1]) if len(speed_wps) >= 2 else 0.0
        if spread < 0.5:
            # Waypoints are clustered near origin (low speed) — draw above vehicle
            centroid = speed_wps.mean(axis=0)
            ax.plot(centroid[0], centroid[1], 'o', color=COLORS['speed_waypoints'],
                    markersize=14, alpha=0.95, zorder=11,
                    markeredgecolor='darkorange', markeredgewidth=2)
        else:
            ax.plot(speed_wps[:, 0], speed_wps[:, 1], '--', color=COLORS['speed_waypoints'],
                    linewidth=SIZES['speed_line'], alpha=0.9, zorder=5)
            ax.plot(speed_wps[:, 0], speed_wps[:, 1], 'o', color=COLORS['speed_waypoints'],
                    markersize=SIZES['speed_wp_marker'], alpha=0.95, zorder=5,
                    markeredgecolor='darkorange', markeredgewidth=1.5)

    # Obstacle
    if obstacle_ego is not None:
        ax.plot(obstacle_ego[0], obstacle_ego[1], 'X', color=COLORS['obstacle'],
                markersize=20, markeredgecolor='black', markeredgewidth=2.5, zorder=8)
        obstacle_circle = Circle(obstacle_ego, 1.5, color=COLORS['collision_zone'],
                                  alpha=0.3, zorder=1)
        ax.add_patch(obstacle_circle)

    # Build title with actual and predicted speed
    collision_str = " [COLLISION]" if entry.get('collision', False) else ""
    title_text = f'Step {entry["step"]}'
    if predicted_speed is not None:
        title_text += f' | Speed: {current_speed:.1f} m/s (pred {predicted_speed:.1f})'
    else:
        title_text += f' | Speed: {current_speed:.1f} m/s'
    title_text += f'{collision_str}{title_suffix}'

    ax.set_title(title_text, fontsize=SIZES['metric_font'], fontweight='bold', pad=8)
    ax.set_xlabel('X (forward, m)', fontsize=SIZES['annotation_font'], fontweight='bold')
    ax.set_ylabel('Y (left, m)', fontsize=SIZES['annotation_font'], fontweight='bold')
    ax.tick_params(labelsize=7)


def _render_panels(trajectory, panel_indices, anchor_idx, route_waypoints_world,
                    obstacle_world, title, suffix_fn, save_path, num_panels=6):
    """Shared rendering logic for ego-frame panel grids."""
    panel_count = min(num_panels, len(panel_indices))
    panel_indices = panel_indices[:panel_count]

    n_cols = min(3, panel_count)
    n_rows = (panel_count + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows + 1.2))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.30,
                          top=0.88, bottom=0.08, left=0.06, right=0.96)

    for i, idx in enumerate(panel_indices):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        entry = trajectory[idx]
        suffix = suffix_fn(idx)
        plot_ego_frame_panel(ax, entry, route_waypoints_world, obstacle_world,
                             title_suffix=suffix)

    # Shared legend below the suptitle, outside of all panels
    legend_handles = [
        Line2D([0], [0], color=COLORS['planned_route'], linewidth=2.5,
               marker='o', markersize=4, alpha=0.5, label='Planned Route'),
        Line2D([0], [0], color=COLORS['route_waypoints'], linewidth=2.5,
               marker='o', markersize=6, markeredgecolor='darkgreen',
               alpha=0.8, label='Predicted Route WPs (20)'),
        Line2D([0], [0], color=COLORS['speed_waypoints'], linewidth=2.5,
               linestyle='--', marker='o', markersize=6,
               markeredgecolor='darkorange', alpha=0.9, label='Predicted Speed WPs (10)'),
        Line2D([0], [0], color=COLORS['target'], marker='*', linestyle='None',
               markersize=14, markeredgecolor='darkorange',
               markeredgewidth=1.5, label='Target WP'),
        Line2D([0], [0], color=COLORS['obstacle'], marker='X', linestyle='None',
               markersize=12, markeredgecolor='black',
               markeredgewidth=1.5, label='Obstacle'),
        Line2D([0], [0], color=COLORS['vehicle'], marker='o', linestyle='None',
               markersize=10, markeredgecolor='darkred',
               markeredgewidth=1.5, label='Ego Vehicle'),
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.95), ncol=4,
               fontsize=SIZES['legend_font'] + 1, frameon=True,
               framealpha=0.95, edgecolor='gray', fancybox=True,
               columnspacing=1.5, handlelength=2.5)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {save_path}")
    plt.close(fig)


def visualize_obstacle_scenario(data, save_path=None, pre_steps=15, num_panels=6):
    """Visualize ego-frame waypoints near obstacle - collision or closest approach."""
    metadata = data['metadata']
    trajectory = data['trajectory']

    route_waypoints_world = metadata.get('route_waypoints')
    obstacle_world = metadata.get('obstacle_location')
    scenario = metadata.get('scenario_name', 'unknown')
    run_num = metadata.get('run_number', '?')

    # Try collision first
    start_idx, collision_idx = find_collision_segment(
        trajectory, pre_steps=pre_steps, obstacle_location=obstacle_world)

    if collision_idx is not None:
        # Collision mode — check if the vehicle stops after collision
        stopped_idx = None
        STOPPED_THRESHOLD = 0.05
        for si in range(collision_idx, len(trajectory)):
            if trajectory[si]['speed'] < STOPPED_THRESHOLD:
                stopped_idx = si
                break

        if stopped_idx is not None:
            # Vehicle stopped after collision — extend panels to the stop point
            end_idx = stopped_idx
            print(f"Collision at step {trajectory[collision_idx]['step']} (index {collision_idx})")
            print(f"Vehicle stopped at step {trajectory[stopped_idx]['step']} (index {stopped_idx})")
            print(f"Segment: steps {trajectory[start_idx]['step']}-{trajectory[end_idx]['step']}")
        else:
            end_idx = collision_idx
            print(f"Collision at step {trajectory[collision_idx]['step']} (index {collision_idx})")
            print(f"Pre-collision segment: steps {trajectory[start_idx]['step']}-{trajectory[end_idx]['step']}")

        anchor_idx = end_idx
        segment_len = end_idx - start_idx + 1
        panel_indices = list(np.linspace(start_idx, end_idx, min(num_panels, segment_len), dtype=int))
        title = f'Collision Waypoint Analysis: {scenario} (run {run_num})'
        default_name = f'results/pre_collision_waypoints_{scenario}.png'

        def suffix_fn(idx):
            if idx >= collision_idx:
                return ""
            d = collision_idx - idx
            return f" ({d} steps to collision)"
    else:
        # Closest-approach mode (no obstacle collision)
        if obstacle_world is None:
            print("No obstacle location in metadata. Nothing to visualize.")
            return
        start_idx, end_idx, min_dist = find_closest_approach_segment(
            trajectory, obstacle_world, pre_steps=pre_steps)
        anchor_idx = end_idx
        print(f"No obstacle collision. Closest approach: step {trajectory[end_idx]['step']} "
              f"(index {end_idx}), min dist={min_dist:.2f}m")
        print(f"Approach segment: steps {trajectory[start_idx]['step']}-{trajectory[end_idx]['step']}")
        segment_len = end_idx - start_idx + 1
        panel_indices = list(np.linspace(start_idx, end_idx, min(num_panels, segment_len), dtype=int))
        title = f'Near-Obstacle Waypoint Analysis (no collision): {scenario} (run {run_num})'
        default_name = f'results/near_obstacle_waypoints_{scenario}.png'

        def suffix_fn(idx):
            d = end_idx - idx
            return f" ({d} steps to closest)" if d > 0 else " [CLOSEST]"

    # Check predicted waypoints
    if trajectory[panel_indices[0]].get('predicted_route_waypoints') is None:
        print("ERROR: Trajectory log does not contain predicted waypoints.")
        print("Re-run the test with the enhanced logging enabled.")
        return

    if save_path is None:
        save_path = default_name

    _render_panels(trajectory, panel_indices, anchor_idx, route_waypoints_world,
                   obstacle_world, title, suffix_fn, save_path, num_panels=num_panels)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize waypoint predictions before a collision')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to trajectory log JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization (default: auto-generated)')
    parser.add_argument('--pre-steps', type=int, default=15,
                        help='Number of steps before collision to visualize (default: 15)')
    parser.add_argument('--panels', type=int, default=6,
                        help='Number of ego-frame panels (default: 6)')
    args = parser.parse_args()

    data = load_trajectory_log(args.log)
    if data is None:
        return 1

    visualize_obstacle_scenario(data, save_path=args.output,
                                pre_steps=args.pre_steps, num_panels=args.panels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
