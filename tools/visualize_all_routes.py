#!/usr/bin/env python3
"""
Visualize the planned route waypoints.
Use this to verify the route looks correct before running the vehicle.
"""

import sys
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory and python directory to path for core imports
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python'))

from hal.products.mats import SDCSRoadMap


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

    # Draw all road edges as light gray lines
    for edge in edges:
        n1, n2 = edge[0], edge[1]
        try:
            path = roadmap.generate_path([n1, n2])
            x_coords = path[0, :] * 10.0
            y_coords = path[1, :] * 10.0
            ax.plot(x_coords, y_coords, color='lightgray', linewidth=1.5, alpha=0.5, zorder=1)
        except:
            pass


def load_all_routes():
    """Load all routes from the routes/ directory."""
    routes_dir = "routes"
    if not os.path.exists(routes_dir):
        print(f"Routes directory not found: {routes_dir}")
        return []

    route_files = glob.glob(os.path.join(routes_dir, "*.json"))
    routes = []

    for route_file in sorted(route_files):
        try:
            with open(route_file, 'r') as f:
                route_data = json.load(f)
                routes.append(route_data)
        except Exception as e:
            print(f"Error loading {route_file}: {e}")

    return routes


def visualize_route():
    """Visualize all routes from the routes/ directory in separate subplots."""
    print("Starting visualization...")
    # Load all routes
    routes = load_all_routes()
    print(f"Loaded {len(routes)} routes")

    if not routes:
        print("ERROR: No routes found in routes/ directory!")
        print("Please run 'python tools/generate_routes.py' to generate routes first.")
        return

    # Initialize roadmap for background
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    # Fixed axis limits to show entire map - extended to prevent cropping
    x_min, x_max = -25, 30
    y_min, y_max = -15, 52  # Extended bottom and top to show full map

    # Determine grid layout (5 columns)
    n_routes = len(routes)
    n_cols = 5
    n_rows = (n_routes + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots - optimized for 5 columns
    # Each subplot gets good visibility - increased height to prevent cropping
    fig_width = 30  # Wide enough for 5 columns
    fig_height = 7 * n_rows  # Taller to prevent bottom cropping
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier iteration
    if n_routes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each route in its own subplot
    for idx, route_data in enumerate(routes):
        ax = axes[idx]
        route_waypoints = np.array(route_data['waypoints'])
        spawn_location = np.array(route_data['spawn_location'])
        route_name = route_data['name']
        num_waypoints = route_data.get('num_waypoints', len(route_waypoints))
        total_distance = route_data.get('total_distance', 0)

        # Draw map background first
        draw_map_background(ax, roadmap)

        # Plot route line and waypoints on top - thicker and more visible
        ax.plot(route_waypoints[:, 0], route_waypoints[:, 1],
                'b-', linewidth=5, alpha=0.9, label='Route', zorder=3)
        ax.plot(route_waypoints[:, 0], route_waypoints[:, 1],
                'b.', markersize=8, alpha=0.7, zorder=3)

        # Mark start (circle) and end (square) - larger markers
        ax.plot(route_waypoints[0, 0], route_waypoints[0, 1],
                'go', markersize=18, label='Start', zorder=5,
                markeredgecolor='darkgreen', markeredgewidth=2.5)
        ax.plot(route_waypoints[-1, 0], route_waypoints[-1, 1],
                'rs', markersize=18, label='End', zorder=5,
                markeredgecolor='darkred', markeredgewidth=2.5)

        # Add statistics text box with route name
        stats_text = f'{route_name}\nWP: {num_waypoints} | Dist: {total_distance:.1f}m'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                zorder=6)

        # Set consistent axis limits to show full map
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Grid and minimal labels
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, zorder=0)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=8)

        # Add legend - smaller and more compact
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9, markerscale=0.7)

    # Hide unused subplots
    for idx in range(n_routes, len(axes)):
        axes[idx].axis('off')

    # Adjust spacing between subplots for better layout - increased margins to prevent cropping
    plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.97, bottom=0.08, left=0.05, right=0.98)

    # Save figure - don't use bbox_inches='tight' as it can crop content
    save_path = "debug_output/all_routes_visualization.png"
    os.makedirs("debug_output", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nAll routes visualization saved to {save_path}")

    # Print summary
    print("\n" + "="*80)
    print("ROUTES SUMMARY")
    print("="*80)
    for route_data in routes:
        route_waypoints = np.array(route_data['waypoints'])
        total_distance = route_data.get('total_distance', 0)
        print(f"\n{route_data['name']}:")
        print(f"  Waypoints: {len(route_waypoints)}")
        print(f"  Distance: {total_distance:.1f}m")
        print(f"  Start: [{route_waypoints[0, 0]:.1f}, {route_waypoints[0, 1]:.1f}]")
        print(f"  End: [{route_waypoints[-1, 0]:.1f}, {route_waypoints[-1, 1]:.1f}]")
    print("="*80)

    # Show interactive plot
    plt.show()
    
    plt.show()


if __name__ == "__main__":
    visualize_route()

