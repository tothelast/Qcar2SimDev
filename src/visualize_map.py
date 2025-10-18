#!/usr/bin/env python3
"""
Visualize the complete Cityscape Lite road network with actual road polylines.

This script generates a comprehensive map showing:
- All 24 nodes in the road network
- All 42 road segments with actual curved geometry (polylines)
- Green/yellow centerline polylines for each road edge
- Red direction arrows at nodes
- Current pedestrian crossing positions
- Vehicle route path

Output is saved to debug_output/cityscape_directional_map.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from hal.products.mats import SDCSRoadMap


def get_node_positions():
    """Get all 24 node positions in QLabs coordinates."""
    scale = 0.002035
    xOffset = 1134
    yOffset = 2363
    pi = np.pi
    halfPi = pi/2

    nodePoses_raw = [
        [1134, 2299, -halfPi], [1266, 2323, halfPi], [1688, 2896, 0], [1688, 2763, pi],
        [2242, 2323, halfPi], [2109, 2323, -halfPi], [1632, 1822, pi], [1741, 1955, 0],
        [766, 1822, pi], [766, 1955, 0], [504, 2589, -42*pi/180], [1134, 1300, -halfPi],
        [1134, 1454, -halfPi], [1266, 1454, halfPi], [2242, 905, halfPi], [2109, 1454, -halfPi],
        [1580, 540, -80.6*pi/180], [1854.4, 814.5, -9.4*pi/180], [1440, 856, -138*pi/180],
        [1523, 958, 42*pi/180], [1134, 153, pi], [1134, 286, 0], [159, 905, -halfPi], [291, 905, halfPi],
    ]

    nodes = {}
    for i, pose in enumerate(nodePoses_raw):
        x_qlabs = scale * (pose[0] - xOffset) * 10
        y_qlabs = scale * (yOffset - pose[1]) * 10
        nodes[i] = (x_qlabs, y_qlabs)
    
    return nodes


def get_all_edges():
    """Get all 42 road edges in the network."""
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


def get_pedestrian_positions():
    """Get current pedestrian crossing positions."""
    return {
        'Ped 1': {'curb1': (-2.0, 23.5), 'curb2': (8.0, 23.5)},
        'Ped 2': {'curb1': (-19.841, 26.0), 'curb2': (-19.841, 36.0)},
        'Ped 3': {'curb1': (0.0, 42.0), 'curb2': (0.0, 48.0)},
        'Ped 4': {'curb1': (-5.0, 5.0), 'curb2': (5.0, 5.0)},
        'Ped 5': {'curb1': (17.0, 12.0), 'curb2': (23.0, 12.0)},
    }


def calculate_offset_polyline(x_center, y_center, offset_distance):
    """
    Calculate an offset polyline parallel to the centerline.

    Args:
        x_center: X coordinates of centerline
        y_center: Y coordinates of centerline
        offset_distance: Distance to offset (positive = left, negative = right)

    Returns:
        x_offset, y_offset: Coordinates of offset polyline
    """
    n_points = len(x_center)
    x_offset = np.zeros(n_points)
    y_offset = np.zeros(n_points)

    for i in range(n_points):
        # Calculate tangent direction at this point
        if i == 0:
            # First point: use direction to next point
            dx = x_center[i+1] - x_center[i]
            dy = y_center[i+1] - y_center[i]
        elif i == n_points - 1:
            # Last point: use direction from previous point
            dx = x_center[i] - x_center[i-1]
            dy = y_center[i] - y_center[i-1]
        else:
            # Middle points: use average of directions
            dx = x_center[i+1] - x_center[i-1]
            dy = y_center[i+1] - y_center[i-1]

        # Normalize tangent
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length

        # Perpendicular direction (rotate 90 degrees left)
        perp_x = -dy
        perp_y = dx

        # Apply offset
        x_offset[i] = x_center[i] + perp_x * offset_distance
        y_offset[i] = y_center[i] + perp_y * offset_distance

    return x_offset, y_offset


def generate_road_polylines():
    """Generate polylines for all road edges using SDCSRoadMap."""
    print("Initializing SDCSRoadMap...")
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    edges = get_all_edges()
    road_polylines = {}

    print(f"\nGenerating polylines for {len(edges)} edges...")
    for i, edge in enumerate(edges):
        n1, n2 = edge[0], edge[1]
        try:
            # Generate path for this edge
            path = roadmap.generate_path([n1, n2])
            # path is shape (2, N) where row 0 = X, row 1 = Y
            # Convert to QLabs coordinates
            x_center = path[0, :] * 10.0
            y_center = path[1, :] * 10.0

            road_polylines[f"{n1}→{n2}"] = {
                'x_center': x_center,
                'y_center': y_center,
                'from': n1,
                'to': n2,
                'points': len(x_center)
            }
            print(f"  Edge {i+1:2d}: {n1:2d}→{n2:2d} - {len(x_center):4d} points")
        except Exception as e:
            print(f"  Edge {i+1:2d}: {n1:2d}→{n2:2d} - ERROR: {e}")

    print(f"\n✓ Generated {len(road_polylines)} road polylines")
    return road_polylines


def downsample_path(x_coords, y_coords, target_spacing=1.0):
    """
    Downsample a path to achieve target spacing between points.

    Args:
        x_coords: X coordinates of path
        y_coords: Y coordinates of path
        target_spacing: Target distance between points in meters

    Returns:
        x_downsampled, y_downsampled: Downsampled coordinates
    """
    if len(x_coords) <= 2:
        return x_coords, y_coords

    downsampled_x = [x_coords[0]]
    downsampled_y = [y_coords[0]]
    accumulated_distance = 0.0

    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        segment_distance = np.sqrt(dx**2 + dy**2)
        accumulated_distance += segment_distance

        if accumulated_distance >= target_spacing:
            downsampled_x.append(x_coords[i])
            downsampled_y.append(y_coords[i])
            accumulated_distance = 0.0

    # Always include the last point
    if downsampled_x[-1] != x_coords[-1]:
        downsampled_x.append(x_coords[-1])
        downsampled_y.append(y_coords[-1])

    return np.array(downsampled_x), np.array(downsampled_y)


def find_parallel_edge_pairs(road_polylines, max_distance=1.0):
    """
    Find pairs of edges that run parallel to each other (spatially close).

    Args:
        road_polylines: Dictionary of road polylines
        max_distance: Maximum distance to consider edges as parallel (meters)

    Returns:
        List of tuples (edge1_name, edge2_name) for parallel edges
    """
    edge_names = list(road_polylines.keys())
    parallel_pairs = []

    for i, edge1_name in enumerate(edge_names):
        edge1 = road_polylines[edge1_name]

        for j, edge2_name in enumerate(edge_names):
            if i >= j:  # Skip self and already processed pairs
                continue

            edge2 = road_polylines[edge2_name]

            # Sample a few points from each edge to check if they're parallel
            # Use the middle point of each edge
            mid1_idx = len(edge1['x_center']) // 2
            mid2_idx = len(edge2['x_center']) // 2

            x1, y1 = edge1['x_center'][mid1_idx], edge1['y_center'][mid1_idx]
            x2, y2 = edge2['x_center'][mid2_idx], edge2['y_center'][mid2_idx]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # If edges are close, they might be parallel
            if distance < max_distance:
                parallel_pairs.append((edge1_name, edge2_name))

    return parallel_pairs


def resample_path_by_distance(x_coords, y_coords, num_points):
    """
    Resample a path to have exactly num_points, evenly distributed by arc length.

    Args:
        x_coords, y_coords: Original path coordinates
        num_points: Number of points in resampled path

    Returns:
        x_resampled, y_resampled: Resampled coordinates
    """
    # Calculate cumulative arc length
    distances = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        distances[i] = distances[i-1] + np.sqrt(dx**2 + dy**2)

    total_length = distances[-1]

    # Create evenly spaced arc length samples
    target_distances = np.linspace(0, total_length, num_points)

    # Interpolate x and y at these arc lengths
    x_resampled = np.interp(target_distances, distances, x_coords)
    y_resampled = np.interp(target_distances, distances, y_coords)

    return x_resampled, y_resampled


def calculate_midline_between_paths(path1_x, path1_y, path2_x, path2_y, target_spacing=1.0, max_distance=10.0):
    """
    Calculate the midline between two parallel paths.

    Args:
        path1_x, path1_y: Coordinates of first path
        path2_x, path2_y: Coordinates of second path
        target_spacing: Target spacing for downsampled points
        max_distance: Maximum distance to consider paths as parallel (meters)

    Returns:
        x, y: Coordinates of the midline
    """
    # Calculate the total length of each path
    def path_length(x, y):
        length = 0
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            length += np.sqrt(dx**2 + dy**2)
        return length

    len1 = path_length(path1_x, path1_y)
    len2 = path_length(path2_x, path2_y)

    # Use the average length and target spacing to determine number of points
    avg_length = (len1 + len2) / 2.0
    num_points = max(10, int(avg_length / target_spacing))

    # Resample both paths to have the same number of evenly-spaced points
    path1_x_resampled, path1_y_resampled = resample_path_by_distance(path1_x, path1_y, num_points)
    path2_x_resampled, path2_y_resampled = resample_path_by_distance(path2_x, path2_y, num_points)

    # Calculate midline by averaging ALL corresponding points
    # Don't filter by distance - just calculate the midpoint
    yellow_x = (path1_x_resampled + path2_x_resampled) / 2.0
    yellow_y = (path1_y_resampled + path2_y_resampled) / 2.0

    return yellow_x, yellow_y


def get_edge_radius_map():
    """
    Get the radius value for each edge based on the SDCSRoadMap configuration.
    This information comes from python/hal/products/mats.py

    Returns:
        Dictionary mapping (from_node, to_node) to radius type
    """
    # From mats.py lines 23-27 (right-hand traffic):
    # innerLaneRadius = 305.5 * scale  (≈ 0.622m)
    # outerLaneRadius = 438 * scale    (≈ 0.891m)
    # trafficCircleRadius = 333 * scale
    # oneWayStreetRadius = 350 * scale
    # kinkStreetRadius = 375 * scale

    scale = 0.002035
    innerLaneRadius = 305.5 * scale
    outerLaneRadius = 438 * scale
    trafficCircleRadius = 333 * scale
    oneWayStreetRadius = 350 * scale
    kinkStreetRadius = 375 * scale

    # Edge configurations from mats.py lines 148-194 (right-hand traffic, not useSmallMap)
    edge_radius_map = {}

    # Main edges
    edge_radius_map[(0, 2)] = outerLaneRadius
    edge_radius_map[(1, 7)] = innerLaneRadius
    edge_radius_map[(1, 8)] = outerLaneRadius
    edge_radius_map[(2, 4)] = outerLaneRadius
    edge_radius_map[(3, 1)] = innerLaneRadius
    edge_radius_map[(4, 6)] = outerLaneRadius
    edge_radius_map[(5, 3)] = innerLaneRadius
    edge_radius_map[(6, 0)] = outerLaneRadius
    edge_radius_map[(6, 8)] = 0
    edge_radius_map[(7, 5)] = innerLaneRadius
    edge_radius_map[(8, 10)] = oneWayStreetRadius
    edge_radius_map[(9, 0)] = innerLaneRadius
    edge_radius_map[(9, 7)] = 0
    edge_radius_map[(10, 1)] = innerLaneRadius
    edge_radius_map[(10, 2)] = innerLaneRadius

    # Extended edges
    edge_radius_map[(1, 13)] = 0
    edge_radius_map[(4, 14)] = 0
    edge_radius_map[(6, 13)] = innerLaneRadius
    edge_radius_map[(7, 14)] = outerLaneRadius
    edge_radius_map[(8, 23)] = innerLaneRadius
    edge_radius_map[(9, 13)] = outerLaneRadius
    edge_radius_map[(11, 12)] = 0
    edge_radius_map[(12, 0)] = 0
    edge_radius_map[(12, 7)] = outerLaneRadius
    edge_radius_map[(12, 8)] = innerLaneRadius
    edge_radius_map[(13, 19)] = innerLaneRadius
    edge_radius_map[(14, 16)] = trafficCircleRadius
    edge_radius_map[(14, 20)] = trafficCircleRadius
    edge_radius_map[(15, 5)] = outerLaneRadius
    edge_radius_map[(15, 6)] = innerLaneRadius
    edge_radius_map[(16, 17)] = trafficCircleRadius
    edge_radius_map[(16, 18)] = innerLaneRadius
    edge_radius_map[(17, 15)] = innerLaneRadius
    edge_radius_map[(17, 16)] = trafficCircleRadius
    edge_radius_map[(17, 20)] = trafficCircleRadius
    edge_radius_map[(18, 11)] = kinkStreetRadius
    edge_radius_map[(19, 17)] = innerLaneRadius
    edge_radius_map[(20, 22)] = outerLaneRadius
    edge_radius_map[(21, 16)] = innerLaneRadius
    edge_radius_map[(22, 9)] = outerLaneRadius
    edge_radius_map[(22, 10)] = outerLaneRadius
    edge_radius_map[(23, 21)] = innerLaneRadius

    return edge_radius_map, innerLaneRadius, outerLaneRadius


def calculate_lane_divider():
    """
    Calculate yellow lane dividers between parallel edges.
    Uses edge radius information from SDCSRoadMap to identify parallel roads.

    Edges with innerLaneRadius and outerLaneRadius that are spatially close
    form parallel roads that need yellow lane dividers between them.

    Returns:
        List of yellow line segments
    """
    print("\nCalculating lane dividers using edge radius information...")

    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    # Get edge radius mapping
    edge_radius_map, innerLaneRadius, outerLaneRadius = get_edge_radius_map()

    # Group edges by radius
    inner_edges = []
    outer_edges = []

    for (from_node, to_node), radius in edge_radius_map.items():
        if abs(radius - innerLaneRadius) < 0.001:
            inner_edges.append((from_node, to_node))
        elif abs(radius - outerLaneRadius) < 0.001:
            outer_edges.append((from_node, to_node))

    print(f"  Found {len(inner_edges)} inner lane edges")
    print(f"  Found {len(outer_edges)} outer lane edges")

    # Generate paths for all inner and outer edges
    inner_paths = {}
    outer_paths = {}

    for from_node, to_node in inner_edges:
        path = roadmap.generate_path([from_node, to_node])
        x = path[0, :] * 10.0
        y = path[1, :] * 10.0
        x_down, y_down = downsample_path(x, y, target_spacing=1.0)
        inner_paths[(from_node, to_node)] = (x_down, y_down)

    for from_node, to_node in outer_edges:
        path = roadmap.generate_path([from_node, to_node])
        x = path[0, :] * 10.0
        y = path[1, :] * 10.0
        x_down, y_down = downsample_path(x, y, target_spacing=1.0)
        outer_paths[(from_node, to_node)] = (x_down, y_down)

    # Find parallel edge pairs (inner + outer that are spatially close)
    # Only keep pairs that are truly parallel (similar direction and close distance)
    lane_dividers = []
    processed_pairs = set()

    for inner_edge, (inner_x, inner_y) in inner_paths.items():
        for outer_edge, (outer_x, outer_y) in outer_paths.items():
            # Skip if already processed
            pair_key = tuple(sorted([inner_edge, outer_edge]))
            if pair_key in processed_pairs:
                continue

            # Check if these edges are spatially close (parallel)
            # Sample points and check average distance
            n_samples = min(5, len(inner_x), len(outer_x))
            if n_samples < 2:
                continue

            idx_inner = np.linspace(0, len(inner_x)-1, n_samples).astype(int)
            idx_outer = np.linspace(0, len(outer_x)-1, n_samples).astype(int)

            total_dist = 0
            for i in idx_inner:
                min_dist = float('inf')
                for j in idx_outer:
                    dist = np.sqrt((outer_x[j] - inner_x[i])**2 + (outer_y[j] - inner_y[i])**2)
                    if dist < min_dist:
                        min_dist = dist
                total_dist += min_dist

            avg_dist = total_dist / n_samples

            # If edges are 2.5-3.2 meters apart on average, they're parallel
            # This stricter range filters out edges that are just nearby but not truly parallel
            if 2.5 < avg_dist < 3.2:
                # Calculate midline
                yellow_x = []
                yellow_y = []

                for i in range(len(inner_x)):
                    # Find nearest waypoint on outer edge
                    min_dist = float('inf')
                    nearest_idx = 0

                    for j in range(len(outer_x)):
                        dist = np.sqrt((outer_x[j] - inner_x[i])**2 + (outer_y[j] - inner_y[i])**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = j

                    # Calculate midpoint
                    mid_x = (inner_x[i] + outer_x[nearest_idx]) / 2.0
                    mid_y = (inner_y[i] + outer_y[nearest_idx]) / 2.0
                    yellow_x.append(mid_x)
                    yellow_y.append(mid_y)

                if len(yellow_x) > 0:
                    lane_dividers.append({
                        'x': np.array(yellow_x),
                        'y': np.array(yellow_y),
                        'description': f"{inner_edge[0]}→{inner_edge[1]} ↔ {outer_edge[0]}→{outer_edge[1]}"
                    })
                    print(f"  {inner_edge[0]}→{inner_edge[1]} ↔ {outer_edge[0]}→{outer_edge[1]}: {len(yellow_x)} waypoints (avg dist: {avg_dist:.2f}m)")
                    processed_pairs.add(pair_key)

    print(f"\n✓ Generated {len(lane_dividers)} lane divider segments")
    return lane_dividers


def create_directional_map(road_polylines, lane_dividers, output_path):
    """Create and save the directional map visualization."""
    nodes = get_node_positions()
    pedestrians = get_pedestrian_positions()

    _, ax = plt.subplots(figsize=(26, 22))

    # Plot all roads with green centerlines
    print("\nPlotting road centerlines...")
    for edge_data in road_polylines.values():
        # Downsample to ~1m spacing to show actual waypoints
        x_down, y_down = downsample_path(
            edge_data['x_center'],
            edge_data['y_center'],
            target_spacing=1.0
        )

        # Plot green centerline (the actual road path)
        ax.plot(x_down, y_down, '-',
                color='green', linewidth=2.5, alpha=0.8, zorder=1)

        # Plot individual waypoints as small dots
        ax.plot(x_down, y_down, 'o',
                color='darkgreen', markersize=3, alpha=0.6, zorder=1)

    # Plot yellow lane dividers (between bidirectional roads)
    print("Plotting lane dividers...")
    for divider in lane_dividers:
        # Plot yellow line
        ax.plot(divider['x'], divider['y'], '-',
                color='yellow', linewidth=2.0, alpha=0.9, zorder=2)

        # Plot individual waypoints on yellow line
        ax.plot(divider['x'], divider['y'], 'o',
                color='orange', markersize=3, alpha=0.7, zorder=2)

    # Add direction arrows at nodes
    print("Adding direction arrows...")
    arrow_length = 1.5  # meters
    for edge_data in road_polylines.values():
        n1 = edge_data['from']
        node_x, node_y = nodes[n1]

        # Get direction from node to first point of edge
        x_center = edge_data['x_center']
        y_center = edge_data['y_center']

        # Use first few points to determine direction
        if len(x_center) >= 2:
            dx = x_center[1] - x_center[0]
            dy = y_center[1] - y_center[0]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normalize and scale
                dx = (dx / length) * arrow_length
                dy = (dy / length) * arrow_length

                # Draw red arrow
                ax.arrow(node_x, node_y, dx, dy,
                        head_width=0.4, head_length=0.3,
                        fc='red', ec='red', linewidth=2.5,
                        alpha=0.9, zorder=4, length_includes_head=True)

    # Legend for colors
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=3, label='Road Centerlines'),
        Line2D([0], [0], color='yellow', linewidth=3, label='Lane Dividers'),
        Line2D([0], [0], color='red', linewidth=3, label='Direction Arrows', marker='>'),
    ]
    
    # Plot nodes
    for node_id, (x, y) in nodes.items():
        ax.plot(x, y, 'o', color='black', markersize=13, zorder=3, 
                markeredgecolor='white', markeredgewidth=2)
        ax.text(x+1.2, y+1.2, f'{node_id}', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                         edgecolor='black', alpha=0.95), zorder=4)
    
    # Highlight route nodes
    route_nodes = [13, 19, 17, 20, 22]
    for node_id in route_nodes:
        x, y = nodes[node_id]
        if node_id == 13:
            ax.plot(x, y, 'o', color='green', markersize=20, zorder=5, 
                    markeredgecolor='white', markeredgewidth=3)
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor='green', markersize=12, 
                                         label='START (Node 13)'))
        elif node_id == 22:
            ax.plot(x, y, 's', color='darkred', markersize=20, zorder=5,
                    markeredgecolor='white', markeredgewidth=3)
            legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor='darkred', markersize=12, 
                                         label='END (Node 22)'))
    
    # Plot pedestrians
    ped_colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow']
    for i, (ped_name, ped_data) in enumerate(pedestrians.items()):
        c1 = ped_data['curb1']
        c2 = ped_data['curb2']
        mid_x = (c1[0] + c2[0]) / 2
        mid_y = (c1[1] + c2[1]) / 2
        color = ped_colors[i]
        
        # Draw thick crossing line
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], '-', color=color, linewidth=8, alpha=0.95, zorder=6)
        ax.plot(c1[0], c1[1], 'o', color=color, markersize=18, zorder=7, 
                markeredgecolor='black', markeredgewidth=3)
        ax.plot(c2[0], c2[1], 's', color=color, markersize=18, zorder=7,
                markeredgecolor='black', markeredgewidth=3)
        
        # Arrow
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            ax.arrow(c1[0], c1[1], dx*0.5, dy*0.5, 
                    head_width=2.0, head_length=1.5, fc=color, ec='black', 
                    linewidth=3, alpha=0.95, zorder=7)
        
        # Label
        dist = np.sqrt(dx**2 + dy**2)
        ax.text(mid_x, mid_y + 3.5, f'{ped_name}\n{dist:.1f}m', 
                fontsize=14, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=color, 
                         edgecolor='black', linewidth=3, alpha=0.98), zorder=8)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=18, fontweight='bold')
    ax.set_title('Cityscape Lite - Complete Road Network\nGreen = Road Centerlines | Yellow = Lane Dividers | Red = Direction',
                 fontsize=20, fontweight='bold', pad=20)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, framealpha=0.95)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    print(f"\n✓ Map saved to {output_path}")
    plt.close()


def main():
    """Main function to generate and save the map visualization."""
    # Set up output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'debug_output'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'cityscape_directional_map.png'

    print("="*80)
    print("CITYSCAPE LITE MAP VISUALIZATION")
    print("="*80)

    # Generate road polylines
    road_polylines = generate_road_polylines()

    # Calculate lane dividers
    lane_dividers = calculate_lane_divider()

    # Create and save the map
    create_directional_map(road_polylines, lane_dividers, output_path)

    print("\n" + "="*80)
    print("Map visualization complete!")
    print(f"Output: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()

