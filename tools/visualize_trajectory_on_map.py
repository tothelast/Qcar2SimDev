#!/usr/bin/env python3
"""
Visualize trajectory on map with route centerline, ego positions, and scene actors.

This script:
1. Finds the most recent trajectory log file automatically
2. Extracts scene information from the log
3. Loads ground truth data (route waypoints, scene actors)
4. Generates a visualization with:
   - Map background
   - Route centerline (green)
   - Ego positions sampled every 10 frames (blue dots with speed labels)
   - Scene actors (red dots)
5. Saves the plot to debug_output/trajectory_plots/
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory and python directory to path for core imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python'))

from hal.products.mats import SDCSRoadMap
from core.scene_loader import SceneLoader

# Distance-based sampling: sample positions at least MIN_DISTANCE_METERS apart
# This creates uniform visual spacing regardless of vehicle speed
MIN_DISTANCE_METERS = 3.0 # Sample every 2 meters of travel


def get_all_edges():
    """Get all road edges in the network."""
    return [
        # Basic edges (nodes 0-10)
        [0, 2], [1, 7], [1, 8], [2, 4], [3, 1], [4, 6], [5, 3],
        [6, 0], [6, 8], [7, 5], [8, 10], [9, 0], [9, 7], [10, 1], [10, 2],
        # Full map edges (nodes 11-23)
        [1, 13], [4, 14], [6, 13], [7, 14], [8, 23], [9, 13], [11, 12],
        [12, 0], [12, 7], [12, 8], [13, 19], [14, 16], [14, 20], [15, 5],
        [15, 6], [16, 17], [16, 18], [17, 15], [17, 16], [17, 20], [18, 11],
        [19, 17], [20, 22], [21, 16], [22, 9], [22, 10], [23, 21],
    ]


def draw_map_background(ax, roadmap):
    """Draw the full road network as background."""
    edges = get_all_edges()
    for edge in edges:
        n1, n2 = edge[0], edge[1]
        try:
            path = roadmap.generate_path([n1, n2])
            x_coords = path[0, :] * 10.0
            y_coords = path[1, :] * 10.0
            ax.plot(x_coords, y_coords, color='lightgray', linewidth=1.5, alpha=0.5, zorder=1)
        except Exception:
            pass


def find_latest_trajectory_log():
    """Find the most recent trajectory log file."""
    log_dir = Path("debug_output")
    if not log_dir.exists():
        return None
    
    # Find all trajectory log files (excluding latest symlink-like file)
    log_files = list(log_dir.glob("trajectory_log_*.json"))
    log_files = [f for f in log_files if "latest" not in f.name]
    
    if not log_files:
        return None
    
    # Sort by modification time and return the most recent
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[0]


def load_trajectory_log(filepath):
    """Load trajectory log from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_scene_actors(scene_name):
    """Load scene actors from config files."""
    if not scene_name:
        return []
    
    scene_loader = SceneLoader()
    scene_def = scene_loader.load_scene(scene_name)
    
    if not scene_def:
        print(f"Warning: Could not load scene '{scene_name}'")
        return []
    
    actors = []
    
    # Collect all actor positions
    for actor in scene_def.actors:
        actor_info = {
            'name': actor.name,
            'type': actor.type,
            'position': None
        }
        
        # Extract position based on actor type
        if 'location' in actor.data:
            actor_info['position'] = actor.data['location'][:2]  # x, y only
        elif 'curb_1' in actor.data:
            # For pedestrians, use midpoint between curbs
            c1 = actor.data['curb_1']
            c2 = actor.data['curb_2']
            actor_info['position'] = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2]
        
        if actor_info['position']:
            actors.append(actor_info)
    
    return actors


def visualize_trajectory_on_map():
    """Main visualization function."""
    print("=" * 80)
    print("TRAJECTORY ON MAP VISUALIZATION")
    print("=" * 80)
    
    # Find and load trajectory log
    log_path = find_latest_trajectory_log()
    if not log_path:
        print("ERROR: No trajectory log files found in debug_output/")
        return 1
    
    print(f"Loading trajectory log: {log_path.name}")
    data = load_trajectory_log(log_path)
    
    metadata = data['metadata']
    trajectory = data['trajectory']
    
    # Extract info from metadata
    timestamp = metadata.get('timestamp', 'unknown')
    scene_name = metadata.get('scene_name', None)
    route_name = metadata.get('route_name', 'unknown')
    route_waypoints = np.array(metadata['route_waypoints'])
    
    print(f"  Timestamp: {timestamp}")
    print(f"  Scene: {scene_name if scene_name else 'None'}")
    print(f"  Route: {route_name}")
    print(f"  Total frames: {len(trajectory)}")
    
    # Load scene actors
    actors = load_scene_actors(scene_name)
    print(f"  Loaded {len(actors)} scene actors")

    ego_positions = []
    ego_speeds = []
    ego_desired_speeds = []
    last_sampled_pos = None

    # Check if desired_speed is available in the log
    has_desired_speed = 'desired_speed' in trajectory[0] if trajectory else False
    if not has_desired_speed:
        print("  Note: 'desired_speed' not found in trajectory log. Run inference again to log it.")

    for entry in trajectory:
        pos = np.array(entry['position'][:2])  # x, y only
        speed = entry['speed']
        desired_speed = entry.get('desired_speed', None)

        if last_sampled_pos is None:
            # Always include the first position
            ego_positions.append(pos)
            ego_speeds.append(speed)
            ego_desired_speeds.append(desired_speed)
            last_sampled_pos = pos
        else:
            distance = np.linalg.norm(pos - last_sampled_pos)
            if distance >= MIN_DISTANCE_METERS:
                ego_positions.append(pos)
                ego_speeds.append(speed)
                ego_desired_speeds.append(desired_speed)
                last_sampled_pos = pos

    # Always include the final position if not already included
    final_pos = np.array(trajectory[-1]['position'][:2])
    if last_sampled_pos is not None and np.linalg.norm(final_pos - last_sampled_pos) > 0.1:
        ego_positions.append(final_pos)
        ego_speeds.append(trajectory[-1]['speed'])
        ego_desired_speeds.append(trajectory[-1].get('desired_speed', None))

    ego_positions = np.array(ego_positions)
    print(f"  Sampled {len(ego_positions)} ego positions (distance-based, every {MIN_DISTANCE_METERS}m)")
    
    # Initialize roadmap for background
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw map background
    draw_map_background(ax, roadmap)
    
    # Plot route centerline (green line)
    ax.plot(route_waypoints[:, 0], route_waypoints[:, 1],
            'g-', linewidth=3, alpha=0.8, label='Route Centerline', zorder=2)
    
    # Plot ego positions (blue dots) and speed labels together
    # Labels are positioned perpendicular to the trajectory direction (to the left of travel)
    LABEL_OFFSET_METERS = 1.2  # Distance from dot to label in meters

    for i in range(len(ego_positions)):
        x = float(ego_positions[i, 0])
        y = float(ego_positions[i, 1])
        speed = ego_speeds[i]
        desired_speed = ego_desired_speeds[i]

        # Plot the dot
        if i == 0:
            ax.plot(x, y, 'o', color='blue', markersize=6, alpha=0.9,
                    markeredgecolor='darkblue', markeredgewidth=1,
                    label='Ego Positions', zorder=4)
        else:
            ax.plot(x, y, 'o', color='blue', markersize=6, alpha=0.9,
                    markeredgecolor='darkblue', markeredgewidth=1, zorder=4)

        # Check if this is a zero-speed position followed by another zero-speed position
        is_zero_speed = abs(speed) < 0.05
        next_is_zero = (i + 1 < len(ego_speeds) and abs(ego_speeds[i + 1]) < 0.05)
        if is_zero_speed and next_is_zero:
            continue

        # Build label: "actual | desired" or just "actual" if desired_speed not available
        if desired_speed is not None:
            label_text = f'{speed:.1f}|{desired_speed:.1f}'
        else:
            label_text = f'{speed:.1f}'

        # Calculate perpendicular direction (to the left of travel direction)
        # Use neighboring points to determine trajectory direction
        if i == 0 and len(ego_positions) > 1:
            # First point: use direction to next point
            direction = ego_positions[1] - ego_positions[0]
        elif i == len(ego_positions) - 1 and len(ego_positions) > 1:
            # Last point: use direction from previous point
            direction = ego_positions[-1] - ego_positions[-2]
        elif len(ego_positions) > 2:
            # Middle points: use direction from previous to next (smoother)
            direction = ego_positions[i + 1] - ego_positions[i - 1]
        else:
            # Fallback: offset to the left
            direction = np.array([0.0, 1.0])

        # Normalize direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0.001:
            direction = direction / dir_norm
        else:
            direction = np.array([1.0, 0.0])

        # Perpendicular vector (rotate 90 degrees counterclockwise = left side of travel)
        perp = np.array([-direction[1], direction[0]])

        # Calculate label position
        label_x = x + perp[0] * LABEL_OFFSET_METERS
        label_y = y + perp[1] * LABEL_OFFSET_METERS

        # Add label at perpendicular offset position
        ax.text(label_x, label_y, label_text,
                fontsize=6, color='darkblue', alpha=0.8,
                ha='center', va='center', zorder=5)
    
    # Plot scene actors (red dots)
    if actors:
        actor_x = [a['position'][0] for a in actors]
        actor_y = [a['position'][1] for a in actors]
        ax.scatter(actor_x, actor_y,
                   c='red', s=100, marker='o', alpha=0.9,
                   label='Scene Actors', zorder=3,
                   edgecolors='darkred', linewidths=2)

        # Add actor labels
        for i, actor in enumerate(actors):
            ax.annotate(actor['name'],
                        xy=(actor_x[i], actor_y[i]),
                        xytext=(-10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        color='darkred',
                        fontweight='bold',
                        alpha=0.9,
                        zorder=5)
    
    # Mark start and end positions
    if len(ego_positions) > 0:
        ax.plot(ego_positions[0, 0], ego_positions[0, 1],
                'g^', markersize=15, label='Start', zorder=6,
                markeredgecolor='darkgreen', markeredgewidth=2)
    if len(ego_positions) > 1:
        ax.plot(ego_positions[-1, 0], ego_positions[-1, 1],
                'ro', markersize=6, label='End', zorder=6,
                markeredgecolor='darkred', markeredgewidth=1)
    
    # Set axis properties
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    
    # Set title
    title = f'Trajectory Visualization: {timestamp}'
    if scene_name:
        title += f'\nScene: {scene_name}'
    title += f' | Route: {route_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Save the plot
    output_dir = Path("debug_output/trajectory_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trajectory_map_{timestamp}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(visualize_trajectory_on_map())

