"""
Simple ACC Baseline Test - QCar2 follows route and stops for obstacles.

Uses simple lane-based obstacle detection:
- LiDAR returns (angle, distance) in vehicle frame
- Convert to (x, y) where y=forward, x=left/right
- Obstacle detected if: y in [MIN_DETECT, STOP_DIST] and |x| < LANE_HALF_WIDTH
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from python.qvl.qlabs import QuanserInteractiveLabs
from python.qvl.qcar2 import QLabsQCar2
from core.scene_loader import SceneLoader
from core.scene_spawner import SceneSpawner

# === PARAMETERS ===
CRUISE_SPEED = 2.0       # m/s
CONTROL_HZ = 10          # Hz
MAX_STEERING = np.pi / 6 # ~30 degrees
LOOKAHEAD = 4.5          # meters for pure pursuit

# ACC parameters - wider detection, filtered by lane
LIDAR_HALF_WIDTH = 5.0   # meters - wide detection cone to catch obstacles
STOP_DISTANCE = 8.0      # meters - detection range
MIN_DETECT_DIST = 1.0    # meters - skip close points (car body)
MIN_POINTS = 3           # minimum points to detect the obstacle. 1 point does not work becuase of noice from the lidar

# Lane parameters - narrower to filter out the points from the curbs
ROAD_LANE_HALF_WIDTH = 1.0  # meters 


def calculate_lane_boundaries(waypoints, lane_half_width):
    """Calculate left and right lane boundaries offset from centerline."""
    n = len(waypoints)
    left_x, left_y = np.zeros(n), np.zeros(n)
    right_x, right_y = np.zeros(n), np.zeros(n)

    for i in range(n):
        # Calculate tangent direction
        if i == 0:
            dx = waypoints[i+1, 0] - waypoints[i, 0]
            dy = waypoints[i+1, 1] - waypoints[i, 1]
        elif i == n - 1:
            dx = waypoints[i, 0] - waypoints[i-1, 0]
            dy = waypoints[i, 1] - waypoints[i-1, 1]
        else:
            dx = waypoints[i+1, 0] - waypoints[i-1, 0]
            dy = waypoints[i+1, 1] - waypoints[i-1, 1]

        # Normalize and get perpendicular
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx / length, dy / length
        perp_x, perp_y = -dy, dx  # Perpendicular (90° left)

        # Offset points
        left_x[i] = waypoints[i, 0] + perp_x * lane_half_width
        left_y[i] = waypoints[i, 1] + perp_y * lane_half_width
        right_x[i] = waypoints[i, 0] - perp_x * lane_half_width
        right_y[i] = waypoints[i, 1] - perp_y * lane_half_width

    return left_x, left_y, right_x, right_y


def save_stop_visualization(position, heading, waypoints, detected_points_world=None, raw_scan_world=None, filename="lidar_stop.png"):
    """Save minimal visualization: route, car, detection cone, and detected obstacle points."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot route centerline
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=2, label='Route')

    # Plot lane boundaries (dashed lines) - using actual road lane width
    left_x, left_y, right_x, right_y = calculate_lane_boundaries(waypoints, ROAD_LANE_HALF_WIDTH)
    ax.plot(left_x, left_y, 'b--', linewidth=1, alpha=0.6, label='Lane Boundaries')
    ax.plot(right_x, right_y, 'b--', linewidth=1, alpha=0.6)

    # Plot car position (triangle pointing in heading direction)
    car_size = 0.8
    dx, dy = car_size * np.cos(heading), car_size * np.sin(heading)
    ax.arrow(position[0], position[1], dx, dy, head_width=0.4, head_length=0.3, fc='green', ec='green')
    ax.plot(position[0], position[1], 'go', markersize=10, label='Car')

    # Transform helper
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    def to_world(x_veh, y_veh):
        x_w = position[0] + x_veh * sin_h + y_veh * cos_h
        y_w = position[1] - x_veh * cos_h + y_veh * sin_h
        return x_w, y_w

    # Plot raw LiDAR scan points - faint background showing what the sensor actually sees
    if raw_scan_world is not None and len(raw_scan_world) > 0:
        ax.scatter(raw_scan_world[:, 0], raw_scan_world[:, 1],
                   c='gray', s=5, alpha=0.15, label='Raw LiDAR Scan')

    # Plot active detection zone (rectangle in world frame)
    corners_veh = [
        (-LIDAR_HALF_WIDTH, MIN_DETECT_DIST),
        (LIDAR_HALF_WIDTH, MIN_DETECT_DIST),
        (LIDAR_HALF_WIDTH, STOP_DISTANCE),
        (-LIDAR_HALF_WIDTH, STOP_DISTANCE),
        (-LIDAR_HALF_WIDTH, MIN_DETECT_DIST),
    ]

    cone_x, cone_y = [], []
    for x_v, y_v in corners_veh:
        x_w, y_w = to_world(x_v, y_v)
        cone_x.append(x_w)
        cone_y.append(y_w)

    ax.fill(cone_x, cone_y, alpha=0.3, color='orange', label='Detection Zone')
    ax.plot(cone_x, cone_y, 'orange', linewidth=2)

    # Plot detected obstacle points
    if detected_points_world is not None and len(detected_points_world) > 0:
        ax.scatter(detected_points_world[:, 0], detected_points_world[:, 1],
                   c='red', s=100, marker='x', linewidths=2, label='Detected Obstacle')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.set_title(f'Stop at ({position[0]:.1f}, {position[1]:.1f})')
    ax.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def load_route(route_name: str):
    """Load route waypoints from JSON file."""
    path = Path(__file__).parent.parent / f"config/routes/{route_name}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data["waypoints"], dtype=np.float32), data["spawn_location"], data["spawn_rotation"]


def is_point_in_lane(point_x, point_y, waypoints, lane_half_width):
    """Check if a world point is within the lane boundaries along the route."""
    # Find nearest waypoint
    dists = np.sqrt((waypoints[:, 0] - point_x)**2 + (waypoints[:, 1] - point_y)**2)
    nearest_idx = np.argmin(dists)

    # Get perpendicular distance to the route at this point
    n = len(waypoints)
    if nearest_idx == 0:
        dx = waypoints[1, 0] - waypoints[0, 0]
        dy = waypoints[1, 1] - waypoints[0, 1]
    elif nearest_idx == n - 1:
        dx = waypoints[n-1, 0] - waypoints[n-2, 0]
        dy = waypoints[n-1, 1] - waypoints[n-2, 1]
    else:
        dx = waypoints[nearest_idx+1, 0] - waypoints[nearest_idx-1, 0]
        dy = waypoints[nearest_idx+1, 1] - waypoints[nearest_idx-1, 1]

    # Normalize tangent
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx, dy = dx / length, dy / length

    # Vector from waypoint to point
    vx = point_x - waypoints[nearest_idx, 0]
    vy = point_y - waypoints[nearest_idx, 1]

    # Perpendicular distance (cross product gives signed distance)
    perp_dist = abs(vx * (-dy) + vy * dx)

    return perp_dist <= lane_half_width


def get_obstacle_distance(qcar, position, heading, waypoints):
    """
    Detect obstacles in lane ahead using LiDAR.
    Only returns obstacle if it's within the actual lane boundaries.
    Returns (distance, detected_points_world) where detected_points_world is None if no obstacle.
    """
    success, angles, distances = qcar.get_lidar(samplePoints=400)
    if not success:
        return float('inf'), None, None

    # Convert to vehicle frame: x=left/right, y=forward
    x = np.sin(angles) * distances
    y = np.cos(angles) * distances

    # Transform ALL raw LiDAR points to world frame (for visualization)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    all_x_w = position[0] + x * sin_h + y * cos_h
    all_y_w = position[1] - x * cos_h + y * sin_h
    raw_scan_world = np.column_stack([all_x_w, all_y_w])

    # Find points in detection cone ahead (wide cone)
    in_cone = (y > MIN_DETECT_DIST) & (y < STOP_DISTANCE) & (np.abs(x) < LIDAR_HALF_WIDTH)

    if np.sum(in_cone) < MIN_POINTS:
        return float('inf'), None, raw_scan_world

    # Get detected points and transform to world frame
    x_det, y_det = x[in_cone], y[in_cone]
    x_w = position[0] + x_det * sin_h + y_det * cos_h
    y_w = position[1] - x_det * cos_h + y_det * sin_h

    # Filter: only keep points that are within the actual lane boundaries
    in_lane_mask = np.array([is_point_in_lane(x_w[i], y_w[i], waypoints, ROAD_LANE_HALF_WIDTH)
                             for i in range(len(x_w))])

    if np.sum(in_lane_mask) < MIN_POINTS:
        return float('inf'), None, raw_scan_world

    # Return distance to closest in-lane obstacle
    y_in_lane = y_det[in_lane_mask]
    detected_world = np.column_stack([x_w[in_lane_mask], y_w[in_lane_mask]])

    return float(np.min(y_in_lane)), detected_world, raw_scan_world


def compute_steering(pos, heading, waypoints):
    """Pure pursuit steering to follow waypoints."""
    # Find lookahead point
    dists = np.linalg.norm(waypoints[:, :2] - pos[:2], axis=1)
    idx = np.argmin(dists)

    cumul = 0.0
    target = waypoints[-1]
    for i in range(idx, len(waypoints) - 1):
        cumul += np.linalg.norm(waypoints[i+1, :2] - waypoints[i, :2])
        if cumul >= LOOKAHEAD:
            target = waypoints[i+1]
            break

    # Convert to ego frame
    rot = np.array([[np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)]])
    ego = rot.T @ (target[:2] - pos[:2])

    angle = np.arctan2(ego[1], ego[0])
    return float(np.clip(angle / MAX_STEERING, -1.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="ACC Baseline Test")
    parser.add_argument('--scene', default='03_roundabout_navigation_obstacle', help='Scene name')
    args = parser.parse_args()

    # Load scene and route
    scene = SceneLoader().load_scene(args.scene)
    if not scene:
        print(f"Failed to load scene '{args.scene}'")
        return

    waypoints, spawn_loc, spawn_rot = load_route(scene.ego_route)
    print(f"Scene: {args.scene}, Route: {scene.ego_route} ({len(waypoints)} waypoints)")

    # Connect and spawn
    qlabs = QuanserInteractiveLabs()
    qlabs.open("localhost")

    qcar = QLabsQCar2(qlabs)
    qcar.destroy_all_actors_of_class()

    spawner = SceneSpawner(scene)
    if spawner.connect():
        spawner.spawn_all_actors()

    qcar.spawn_id(0, spawn_loc, spawn_rot, waitForConfirmation=True)
    time.sleep(0.5)

    # Get initial state
    success, loc, rot, _, _ = qcar.set_velocity_and_request_state(0, 0, False, False, False, False, False)
    position, rotation = np.array(loc), np.array(rot)
    print(f"Starting at ({position[0]:.1f}, {position[1]:.1f})")

    dt = 1.0 / CONTROL_HZ
    stopped = False

    try:
        while True:
            t0 = time.time()
            heading = rotation[2]

            # Check for obstacles in lane ahead (filtered by lane boundaries)
            obstacle_dist, detected_pts, raw_scan = get_obstacle_distance(qcar, position, heading, waypoints)

            if obstacle_dist < STOP_DISTANCE and not stopped:
                print(f"OBSTACLE at {obstacle_dist:.1f}m - stopping")
                save_stop_visualization(position, heading, waypoints, detected_pts, raw_scan)
                stopped = True

            # Check route completion
            if np.linalg.norm(waypoints[-1, :2] - position[:2]) < 2.0:
                print("Route complete!")
                break

            # Compute control
            speed = 0.0 if stopped else CRUISE_SPEED
            steering = compute_steering(position, heading, waypoints)
            turn = -steering * MAX_STEERING

            # Send control
            success, loc, rot, _, _ = qcar.set_velocity_and_request_state(speed, turn, False, False, False, False, False)
            if not success:
                break
            position, rotation = np.array(loc), np.array(rot)

            # Rate limit
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        qcar.set_velocity_and_request_state(0, 0, False, False, False, False, False)
        spawner.cleanup()


if __name__ == "__main__":
    main()
