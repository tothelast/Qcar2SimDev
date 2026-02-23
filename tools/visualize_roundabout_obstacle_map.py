#!/usr/bin/env python3
"""
Generate a publication-quality map of the roundabout navigation route with
all five static obstacle placement variants.

Output is saved to report/figures/roundabout_obstacle_map.png (300 DPI).

Usage:
    python tools/visualize_roundabout_obstacle_map.py
    python tools/visualize_roundabout_obstacle_map.py --show
    python tools/visualize_roundabout_obstacle_map.py --output path/to/output.png
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from pathlib import Path

# Add python directory to path for SDCSRoadMap
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from hal.products.mats import SDCSRoadMap

# ---------------------------------------------------------------------------
# Style (matches results/generate_report_figures.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

# Colors from generate_report_figures.py
ROUTE_COLOR = '#2196F3'      # Blue (SimLingo color)
ROUTE_ARROW_COLOR = '#1565C0'
ROAD_BG_COLOR = '#D0D0D0'

OUTCOME_COLORS = {
    'pass':      '#4CAF50',  # Green  (PASS_COLOR)
    'low_speed': '#FF9800',  # Orange (ACC_COLOR)
    'fail':      '#F44336',  # Red    (FAIL_COLOR)
}

OUTCOME_LABELS = {
    'pass':      'Clean stop (pass)',
    'low_speed': 'Low-speed contact (fail)',
    'fail':      'Collision (fail)',
}

# ---------------------------------------------------------------------------
# Data constants (from config JSON files)
# ---------------------------------------------------------------------------
ROUTE_FILE = Path(__file__).parent.parent / 'config' / 'routes' / 'roundabout_navigation.json'

# Obstacle definitions (config/actors/static/obstacle_car_var{1-5}.json)
# Label offsets are (dx, dy) from car center, tuned to avoid overlaps.
OBSTACLES = [
    {
        'var': 1,
        'label': 'V1 (WP 25)\nEarly Roundabout',
        'location': [21.01, 33.90],
        'rotation_deg': 55.3,
        'waypoint_idx': 25,
        'outcome': 'pass',
        'label_offset': (2.5, -3.5),
    },
    {
        'var': 2,
        'label': 'V2 (WP 35)\nMid Roundabout',
        'location': [18.85, 44.23],
        'rotation_deg': 157.6,
        'waypoint_idx': 35,
        'outcome': 'low_speed',
        'label_offset': (3.0, 3.0),
    },
    {
        'var': 3,
        'label': 'V3 (WP 50)\nRoundabout Exit',
        'location': [6.07, 44.97],
        'rotation_deg': 180.0,
        'waypoint_idx': 50,
        'outcome': 'pass',
        'label_offset': (0.0, 3.5),
    },
    {
        'var': 4,
        'label': 'V4 (WP 65)\nStraight Section',
        'location': [-10.60, 44.97],
        'rotation_deg': 181.5,
        'waypoint_idx': 65,
        'outcome': 'low_speed',
        'label_offset': (0.0, 3.5),
    },
    {
        'var': 5,
        'label': 'V5 (WP 75)\nLate Route',
        'location': [-18.73, 40.37],
        'rotation_deg': 244.6,
        'waypoint_idx': 75,
        'outcome': 'fail',
        'label_offset': (-3.5, 2.5),
    },
]

# Ego start (from route JSON spawn_location / spawn_rotation)
EGO_START = [2.686, 18.498]
EGO_HEADING_DEG = 90.0  # facing north (spawn_rotation z = pi/2)

# Car rectangle size for visualization (scaled up ~3x for visibility)
CAR_VIS_LENGTH = 1.2  # meters (visual, not to scale)
CAR_VIS_WIDTH = 0.6


# ---------------------------------------------------------------------------
# Road network (from tools/visualize_map.py)
# ---------------------------------------------------------------------------
def get_all_edges():
    """Get all 42 road edges in the network."""
    return [
        [0, 2], [1, 7], [1, 8], [2, 4], [3, 1], [4, 6], [5, 3],
        [6, 0], [6, 8], [7, 5], [8, 10], [9, 0], [9, 7], [10, 1], [10, 2],
        [1, 13], [4, 14], [6, 13], [7, 14], [8, 23], [9, 13], [11, 12],
        [12, 0], [12, 7], [12, 8], [13, 19], [14, 16], [14, 20], [15, 5],
        [15, 6], [16, 17], [16, 18], [17, 15], [17, 16], [17, 20], [18, 11],
        [19, 17], [20, 22], [21, 16], [22, 9], [22, 10], [23, 21],
    ]


def draw_road_network(ax, roadmap):
    """Draw the full road network centerlines as light gray background."""
    for edge in get_all_edges():
        n1, n2 = edge
        try:
            path = roadmap.generate_path([n1, n2])
            x = path[0, :] * 10.0
            y = path[1, :] * 10.0
            ax.plot(x, y, color=ROAD_BG_COLOR, linewidth=1.2, alpha=0.5, zorder=1)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
def load_route_waypoints():
    """Load roundabout_navigation route waypoints from JSON."""
    with open(ROUTE_FILE) as f:
        data = json.load(f)
    return np.array(data['waypoints'])[:, :2]  # x, y only


def draw_route(ax, waypoints):
    """Draw the navigation route with direction arrows."""
    ax.plot(waypoints[:, 0], waypoints[:, 1],
            color=ROUTE_COLOR, linewidth=2.5, alpha=0.9, zorder=3,
            solid_capstyle='round')

    # Direction arrows at ~6 evenly spaced positions
    n = len(waypoints)
    arrow_indices = np.linspace(5, n - 5, 6, dtype=int)
    for idx in arrow_indices:
        x0, y0 = waypoints[idx]
        x1, y1 = waypoints[min(idx + 3, n - 1)]
        dx, dy = x1 - x0, y1 - y0
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 0:
            dx, dy = dx / norm, dy / norm
            ax.annotate('', xy=(x0 + dx * 1.5, y0 + dy * 1.5),
                        xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=ROUTE_ARROW_COLOR,
                                        lw=1.8, mutation_scale=12),
                        zorder=4)


def draw_route_segment_labels(ax):
    """Add subtle segment labels along the route."""
    segments = [
        {'text': 'Approach', 'pos': (4.5, 23.0), 'rotation': 80},
        {'text': 'Roundabout\ncurve', 'pos': (22.0, 37.0), 'rotation': 0},
        {'text': 'Final\ncurve', 'pos': (-16.0, 33.0), 'rotation': 0},
    ]
    for seg in segments:
        ax.text(seg['pos'][0], seg['pos'][1], seg['text'],
                fontsize=7, fontstyle='italic', color='#777777',
                rotation=seg['rotation'], ha='center', va='center',
                zorder=2)


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
def draw_obstacle_car(ax, x, y, heading_deg, color, label, label_offset):
    """Draw an obstacle car as a rotated rectangle with label."""
    angle_rad = np.radians(heading_deg)

    # Rotated rectangle
    rect = Rectangle(
        (-CAR_VIS_LENGTH / 2, -CAR_VIS_WIDTH / 2),
        CAR_VIS_LENGTH, CAR_VIS_WIDTH,
        linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.85,
        zorder=7,
    )
    t = Affine2D().rotate(angle_rad).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    # Heading indicator (small line from center toward front)
    front_x = x + np.cos(angle_rad) * (CAR_VIS_LENGTH * 0.7)
    front_y = y + np.sin(angle_rad) * (CAR_VIS_LENGTH * 0.7)
    ax.plot([x, front_x], [y, front_y], color='black', linewidth=1.2,
            alpha=0.7, zorder=8)

    # Label with connector line
    lx = x + label_offset[0]
    ly = y + label_offset[1]
    ax.annotate(
        label, xy=(x, y), xytext=(lx, ly),
        fontsize=7.5, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7,
                  edgecolor='gray', linewidth=0.5),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.8,
                        connectionstyle='arc3,rad=0.1'),
        zorder=9,
    )


# ---------------------------------------------------------------------------
# Scene context
# ---------------------------------------------------------------------------
def draw_ego_start(ax):
    """Draw the ego vehicle start position."""
    x, y = EGO_START
    ax.plot(x, y, marker='^', color=ROUTE_COLOR, markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, zorder=6)
    ax.text(x + 1.8, y - 0.3, 'Ego start', fontsize=8, fontweight='bold',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=ROUTE_COLOR, alpha=0.9, linewidth=0.8),
            zorder=8)


def draw_scene_context(ax):
    """Draw scene context elements (currently none active)."""
    pass


def draw_scale_bar(ax):
    """Scale bar (disabled -- axis labels provide scale)."""
    pass


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
def build_legend(ax):
    """Build the figure legend."""
    handles = [
        Line2D([0], [0], color=ROAD_BG_COLOR, linewidth=1.5,
               label='Road network'),
        Line2D([0], [0], color=ROUTE_COLOR, linewidth=2.5,
               label='Navigation route'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=ROUTE_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=1,
               label='Ego start', linestyle='None'),
    ]
    for key, lbl in OUTCOME_LABELS.items():
        handles.append(
            mpatches.Patch(facecolor=OUTCOME_COLORS[key], edgecolor='black',
                           linewidth=0.8, label=f'Obstacle: {lbl}')
        )
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.95, edgecolor='gray', fancybox=True,
              handlelength=2.0, labelspacing=0.6)


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def create_roundabout_obstacle_map(output_path, show=False):
    """Generate the roundabout obstacle map figure."""
    print("Initializing SDCSRoadMap...")
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    print("Loading route waypoints...")
    waypoints = load_route_waypoints()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Layer 1: road network background
    draw_road_network(ax, roadmap)

    # Layer 2: navigation route
    draw_route(ax, waypoints)

    # Layer 3: scene context
    draw_scene_context(ax)

    # Layer 4: obstacle cars
    for obs in OBSTACLES:
        color = OUTCOME_COLORS[obs['outcome']]
        draw_obstacle_car(ax, obs['location'][0], obs['location'][1],
                          obs['rotation_deg'], color,
                          obs['label'], obs['label_offset'])

    # Layer 5: ego start
    draw_ego_start(ax)

    # Layer 6: route segment labels
    draw_route_segment_labels(ax)

    # Axis configuration
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(False)

    # Crop to route region with margin (extra right margin for V1 label)
    margin = 5.0
    ax.set_xlim(waypoints[:, 0].min() - margin, waypoints[:, 0].max() + margin + 2)
    ax.set_ylim(waypoints[:, 1].min() - margin, waypoints[:, 1].max() + margin + 1)

    # Scale bar and legend
    draw_scale_bar(ax)
    build_legend(ax)

    # Save
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate roundabout obstacle map for thesis report')
    parser.add_argument('--output', type=str,
                        default='report/figures/roundabout_obstacle_map.png',
                        help='Output file path')
    parser.add_argument('--show', action='store_true',
                        help='Show interactive plot window')
    args = parser.parse_args()
    create_roundabout_obstacle_map(args.output, show=args.show)


if __name__ == '__main__':
    main()
