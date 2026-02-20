"""
ACC Baseline Testing Framework - Comparable to SimLingo Test Framework.
Runs the LiDAR-based ACC baseline through the same test matrix and metrics.

Test Matrix:
- Baseline (no obstacle): 5 runs
- Obstacle Var 1-5: 2 runs each
- Total: 15 test runs

Usage:
    python results/test_acc_baseline_roundabout.py
    python results/test_acc_baseline_roundabout.py --scenario obstacle_var1
    python results/test_acc_baseline_roundabout.py --quick

Prerequisites:
    - QLabs must be running with SDCS RoadMap loaded
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from python.qvl.qlabs import QuanserInteractiveLabs
from python.qvl.qcar2 import QLabsQCar2
from core.scene_loader import SceneLoader, SceneDefinition
from core.scene_spawner import SceneSpawner

from results.metrics import (
    SafetyMetrics, TestResult, NumpyEncoder,
    compute_safety_metrics, compute_route_coverage, compute_lateral_deviation,
    compute_distance_traveled, detect_stuck, determine_pass_status,
    compute_total_route_distance, generate_report,
    STOPPED_SPEED_THRESHOLD, STOPPED_WINDOW_SIZE,
    MIN_DISTANCE_FROM_START, MAX_DISTANCE_TO_OBSTACLE,
)


# =============================================================================
# ACC Controller Parameters
# =============================================================================

CRUISE_SPEED = 2.0       # m/s
CONTROL_HZ = 10          # Hz
MAX_STEERING = np.pi / 6 # ~30 degrees
LOOKAHEAD = 4.5          # meters for pure pursuit

# LiDAR detection parameters
LIDAR_HALF_WIDTH = 5.0   # meters - wide detection cone
STOP_DISTANCE = 8.0      # meters - detection range
MIN_DETECT_DIST = 1.0    # meters - skip close points (car body)
MIN_POINTS = 3           # minimum points for obstacle detection

# Lane filtering parameters
ROAD_LANE_HALF_WIDTH = 1.0  # meters

# Route start position (from roundabout_navigation.json)
ROUTE_START_POS = [2.69, 18.50]

# Obstacle locations from config files
OBSTACLE_LOCATIONS = {
    "obstacle_car_var1": [21.01, 33.90, 0.005],
    "obstacle_car_var2": [18.85, 44.23, 0.005],
    "obstacle_car_var3": [6.07, 44.97, 0.005],
    "obstacle_car_var4": [-10.60, 44.97, 0.005],
    "obstacle_car_var5": [-18.73, 40.37, 0.005],
}


# =============================================================================
# Test Scenarios (same as SimLingo framework)
# =============================================================================

@dataclass
class TestScenario:
    """Configuration for a single test scenario."""
    name: str
    description: str
    ego_route: str = "roundabout_navigation"
    obstacle_actor: Optional[str] = None
    obstacle_location: Optional[List[float]] = None
    num_runs: int = 1
    max_steps: int = 1500  # Higher limit since ACC runs at 10Hz
    timeout_seconds: float = 150.0


TEST_SCENARIOS = [
    TestScenario(
        name="baseline",
        description="Roundabout navigation without obstacles",
        ego_route="roundabout_navigation",
        obstacle_actor=None,
        num_runs=5,
    ),
    TestScenario(
        name="obstacle_var1",
        description="Obstacle at early roundabout position",
        ego_route="roundabout_navigation",
        obstacle_actor="obstacle_car_var1",
        obstacle_location=OBSTACLE_LOCATIONS["obstacle_car_var1"],
        num_runs=2,
    ),
    TestScenario(
        name="obstacle_var2",
        description="Obstacle at mid roundabout position",
        ego_route="roundabout_navigation",
        obstacle_actor="obstacle_car_var2",
        obstacle_location=OBSTACLE_LOCATIONS["obstacle_car_var2"],
        num_runs=2,
    ),
    TestScenario(
        name="obstacle_var3",
        description="Obstacle at roundabout exit",
        ego_route="roundabout_navigation",
        obstacle_actor="obstacle_car_var3",
        obstacle_location=OBSTACLE_LOCATIONS["obstacle_car_var3"],
        num_runs=2,
    ),
    TestScenario(
        name="obstacle_var4",
        description="Obstacle on straight section",
        ego_route="roundabout_navigation",
        obstacle_actor="obstacle_car_var4",
        obstacle_location=OBSTACLE_LOCATIONS["obstacle_car_var4"],
        num_runs=2,
    ),
    TestScenario(
        name="obstacle_var5",
        description="Obstacle at late route position",
        ego_route="roundabout_navigation",
        obstacle_actor="obstacle_car_var5",
        obstacle_location=OBSTACLE_LOCATIONS["obstacle_car_var5"],
        num_runs=2,
    ),
]


# =============================================================================
# ACC Controller Functions (from test_acc_baseline.py)
# =============================================================================

def is_point_in_lane(point_x, point_y, waypoints, lane_half_width):
    """Check if a world point is within the lane boundaries along the route."""
    dists = np.sqrt((waypoints[:, 0] - point_x)**2 + (waypoints[:, 1] - point_y)**2)
    nearest_idx = np.argmin(dists)

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

    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx, dy = dx / length, dy / length

    vx = point_x - waypoints[nearest_idx, 0]
    vy = point_y - waypoints[nearest_idx, 1]
    perp_dist = abs(vx * (-dy) + vy * dx)

    return perp_dist <= lane_half_width


def get_obstacle_distance(qcar, position, heading, waypoints):
    """
    Detect obstacles in lane ahead using LiDAR.
    Returns (distance, detected_points_world, raw_scan_world).
    """
    success, angles, distances = qcar.get_lidar(samplePoints=400)
    if not success:
        return float('inf'), None, None

    x = np.sin(angles) * distances
    y = np.cos(angles) * distances

    cos_h, sin_h = np.cos(heading), np.sin(heading)
    all_x_w = position[0] + x * sin_h + y * cos_h
    all_y_w = position[1] - x * cos_h + y * sin_h
    raw_scan_world = np.column_stack([all_x_w, all_y_w])

    in_cone = (y > MIN_DETECT_DIST) & (y < STOP_DISTANCE) & (np.abs(x) < LIDAR_HALF_WIDTH)

    if np.sum(in_cone) < MIN_POINTS:
        return float('inf'), None, raw_scan_world

    x_det, y_det = x[in_cone], y[in_cone]
    x_w = position[0] + x_det * sin_h + y_det * cos_h
    y_w = position[1] - x_det * cos_h + y_det * sin_h

    in_lane_mask = np.array([is_point_in_lane(x_w[i], y_w[i], waypoints, ROAD_LANE_HALF_WIDTH)
                             for i in range(len(x_w))])

    if np.sum(in_lane_mask) < MIN_POINTS:
        return float('inf'), None, raw_scan_world

    y_in_lane = y_det[in_lane_mask]
    detected_world = np.column_stack([x_w[in_lane_mask], y_w[in_lane_mask]])

    return float(np.min(y_in_lane)), detected_world, raw_scan_world


def compute_steering(pos, heading, waypoints):
    """Pure pursuit steering to follow waypoints."""
    dists = np.linalg.norm(waypoints[:, :2] - pos[:2], axis=1)
    idx = np.argmin(dists)

    cumul = 0.0
    target = waypoints[-1]
    for i in range(idx, len(waypoints) - 1):
        cumul += np.linalg.norm(waypoints[i+1, :2] - waypoints[i, :2])
        if cumul >= LOOKAHEAD:
            target = waypoints[i+1]
            break

    rot = np.array([[np.cos(heading), -np.sin(heading)],
                    [np.sin(heading), np.cos(heading)]])
    ego = rot.T @ (target[:2] - pos[:2])

    angle = np.arctan2(ego[1], ego[0])
    return float(np.clip(angle / MAX_STEERING, -1.0, 1.0))


# =============================================================================
# Route Loading
# =============================================================================

def load_route(route_name: str):
    """Load route waypoints from JSON file."""
    path = Path(__file__).parent.parent / f"config/routes/{route_name}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data["waypoints"], dtype=np.float32), data["spawn_location"], data["spawn_rotation"]


# =============================================================================
# ACC Test Runner
# =============================================================================

class ACCTestRunner:
    """Test runner for ACC baseline evaluation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results: List[TestResult] = []

    def run_scenario(self, scenario: TestScenario, run_number: int) -> TestResult:
        """Run a single test scenario with ACC controller."""
        print(f"\n{'='*60}")
        print(f"RUNNING ACC: {scenario.name} - Run {run_number}/{scenario.num_runs}")
        print(f"Description: {scenario.description}")
        print(f"{'='*60}")

        run_dir = self.output_dir / 'acc_runs' / f"{scenario.name}_run_{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load route
        waypoints, spawn_loc, spawn_rot = load_route(scenario.ego_route)

        # Connect to QLabs
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        qcar = QLabsQCar2(qlabs)
        qcar.destroy_all_actors_of_class()

        # Spawn scene actors if needed
        spawner = None
        if scenario.obstacle_actor:
            loader = SceneLoader()
            actor_def = loader.load_actor(scenario.obstacle_actor)
            if actor_def:
                scene_data = {
                    'name': f"test_{scenario.obstacle_actor}",
                    'description': f"Test with {scenario.obstacle_actor}",
                    'ego_route': scenario.ego_route,
                }
                scene_def = SceneDefinition(scene_data, scene_path="", actors=[actor_def])
                spawner = SceneSpawner(scene_def)
                if spawner.connect():
                    spawner.spawn_all_actors()

        # Spawn QCar2
        qcar.spawn_id(0, spawn_loc, spawn_rot, waitForConfirmation=True)
        time.sleep(0.5)

        # Get initial state
        success, loc, rot, _, _ = qcar.set_velocity_and_request_state(0, 0, False, False, False, False, False)
        position, rotation = np.array(loc), np.array(rot)

        dt = 1.0 / CONTROL_HZ
        stopped = False
        speed = 0.0
        trajectory_log = []
        collision_count = 0
        route_complete = False
        timeout = False
        start_time = time.time()
        step = 0

        try:
            while step < scenario.max_steps:
                t0 = time.time()

                # Check timeout
                elapsed_total = time.time() - start_time
                if elapsed_total > scenario.timeout_seconds:
                    print(f"TIMEOUT: Exceeded {scenario.timeout_seconds}s")
                    timeout = True
                    break

                heading = rotation[2]
                prev_position = position.copy()

                # Check for obstacles in lane ahead
                obstacle_dist, detected_pts, raw_scan = get_obstacle_distance(qcar, position, heading, waypoints)

                if obstacle_dist < STOP_DISTANCE and not stopped:
                    print(f"OBSTACLE at {obstacle_dist:.1f}m - stopping")
                    stopped = True

                # Check route completion
                if np.linalg.norm(waypoints[-1, :2] - position[:2]) < 2.0:
                    print("Route complete!")
                    route_complete = True
                    trajectory_log.append({
                        'step': int(step),
                        'timestamp': float(time.time() - start_time),
                        'position': position.tolist(),
                        'heading_deg': float(heading * 180 / np.pi),
                        'speed': float(speed),
                        'desired_speed': float(0.0),
                        'steering': 0.0,
                        'collision': False,
                    })
                    break

                # Compute control
                desired_speed = 0.0 if stopped else CRUISE_SPEED
                steering = compute_steering(position, heading, waypoints)
                turn = -steering * MAX_STEERING

                # Send control and get new state
                success, loc, rot, _, _ = qcar.set_velocity_and_request_state(
                    desired_speed, turn, False, False, False, False, False
                )
                if not success:
                    break
                position, rotation = np.array(loc), np.array(rot)

                # Compute speed from position change (new pos vs old pos)
                actual_dt = time.time() - t0
                if actual_dt > 0:
                    speed = float(np.linalg.norm(position[:2] - prev_position[:2]) / actual_dt)
                else:
                    speed = 0.0

                # Log trajectory
                trajectory_log.append({
                    'step': int(step),
                    'timestamp': float(time.time() - start_time),
                    'position': position.tolist(),
                    'heading_deg': float(rotation[2] * 180 / np.pi),
                    'speed': float(speed),
                    'desired_speed': float(desired_speed),
                    'steering': float(steering),
                    'collision': False,
                })

                # Early termination: sustained stop near obstacle
                if scenario.obstacle_location is not None and len(trajectory_log) >= STOPPED_WINDOW_SIZE:
                    recent_speeds = [e['speed'] for e in trajectory_log[-STOPPED_WINDOW_SIZE:]]
                    avg_recent_speed = sum(recent_speeds) / len(recent_speeds)

                    if avg_recent_speed < STOPPED_SPEED_THRESHOLD:
                        current_pos = position[:2]
                        obstacle_pos = np.array(scenario.obstacle_location[:2])
                        start_pos = np.array(ROUTE_START_POS)

                        dist_to_obs = np.linalg.norm(current_pos - obstacle_pos)
                        dist_from_start = np.linalg.norm(current_pos - start_pos)

                        if dist_to_obs < MAX_DISTANCE_TO_OBSTACLE and dist_from_start > MIN_DISTANCE_FROM_START:
                            print(f"Vehicle stopped near obstacle (dist: {dist_to_obs:.1f}m) - terminating")
                            break

                # Print status
                if step % 50 == 0:
                    dists = np.linalg.norm(waypoints[:, :2] - position[:2], axis=1)
                    progress = np.argmin(dists) / len(waypoints)
                    print(f"Step {step:4d} | Speed: {speed:5.2f} m/s | Progress: {progress*100:5.1f}%")

                step += 1

                # Rate limit
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"ERROR during run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop vehicle and cleanup
            qcar.set_velocity_and_request_state(0, 0, False, False, False, False, False)
            if spawner:
                spawner.cleanup()

        # Compute metrics
        safety = compute_safety_metrics(
            trajectory_log, scenario.obstacle_location, collision_count
        )
        route_coverage = compute_route_coverage(trajectory_log, waypoints)
        avg_dev, max_dev = compute_lateral_deviation(trajectory_log, waypoints)
        distance_traveled = compute_distance_traveled(trajectory_log)
        total_route_dist = compute_total_route_distance(waypoints)
        total_time = trajectory_log[-1]['timestamp'] if trajectory_log else 0.0
        stuck = detect_stuck(trajectory_log)

        result = TestResult(
            scenario_name=scenario.name,
            run_number=run_number,
            safety=safety,
            route_complete=route_complete,
            route_coverage_percent=route_coverage,
            distance_traveled=distance_traveled,
            total_route_distance=total_route_dist,
            avg_lateral_deviation=avg_dev,
            max_lateral_deviation=max_dev,
            total_steps=len(trajectory_log),
            total_time=total_time,
            stuck_detected=stuck,
            timeout=timeout,
            pass_status=False,
            trajectory_log_path=str(run_dir / "trajectory_log.json"),
        )

        is_obstacle = scenario.obstacle_actor is not None
        result.pass_status = determine_pass_status(result, is_obstacle)

        # Save trajectory log
        traj_data = {
            'metadata': {
                'controller': 'ACC_baseline',
                'scenario_name': scenario.name,
                'run_number': run_number,
                'timestamp': datetime.now().isoformat(),
                'pass_status': result.pass_status,
                'collision_detected': result.safety.collision_detected,
                'stopping_distance': result.safety.stopping_distance,
                'route_coverage_percent': result.route_coverage_percent,
            },
            'trajectory': trajectory_log,
        }
        traj_path = run_dir / "trajectory_log.json"
        with open(traj_path, 'w') as f:
            json.dump(traj_data, f, indent=2, cls=NumpyEncoder)

        # Print summary
        status = "PASS" if result.pass_status else "FAIL"
        print(f"\n{'─'*40}")
        print(f"Result: {status}")
        print(f"  Route Coverage: {result.route_coverage_percent:.1f}%")
        print(f"  Distance Traveled: {result.distance_traveled:.2f}m")
        print(f"  Avg Lateral Deviation: {result.avg_lateral_deviation:.3f}m")
        print(f"  Collision: {result.safety.collision_detected}")
        if result.safety.stopping_distance >= 0:
            print(f"  Stopped: {result.safety.stopped_before_obstacle}, "
                  f"Distance: {result.safety.stopping_distance:.2f}m")
        print(f"{'─'*40}")

        return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACC Baseline Testing Framework (comparable to SimLingo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python results/test_acc_baseline_roundabout.py
    python results/test_acc_baseline_roundabout.py --scenario obstacle_var1
    python results/test_acc_baseline_roundabout.py --quick

Available scenarios:
    baseline, obstacle_var1, obstacle_var2, obstacle_var3, obstacle_var4, obstacle_var5
        """
    )

    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (baseline only, 1 run)')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['baseline', 'obstacle_var1', 'obstacle_var2',
                                 'obstacle_var3', 'obstacle_var4', 'obstacle_var5'],
                        help='Run only a specific scenario')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline tests')
    parser.add_argument('--runs', type=int, default=None,
                        help='Override number of runs per scenario')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'acc_runs').mkdir(exist_ok=True)

    print("="*80)
    print("ACC BASELINE TESTING FRAMEWORK")
    print("="*80)

    # Determine scenarios
    if args.quick:
        scenarios = [TestScenario(
            name="baseline",
            description="Quick test - baseline only",
            ego_route="roundabout_navigation",
            num_runs=1,
        )]
    elif args.scenario:
        scenarios = [s for s in TEST_SCENARIOS if s.name == args.scenario]
        if args.runs:
            scenarios[0].num_runs = args.runs
    elif args.skip_baseline:
        scenarios = [s for s in TEST_SCENARIOS if s.name != "baseline"]
        if args.runs:
            for s in scenarios:
                s.num_runs = args.runs
    else:
        scenarios = TEST_SCENARIOS
        if args.runs:
            for s in scenarios:
                s.num_runs = args.runs

    total_runs = sum(s.num_runs for s in scenarios)
    print(f"Total runs: {total_runs}")

    runner = ACCTestRunner(output_dir=output_dir)

    results = []
    for scenario in scenarios:
        for run_num in range(1, scenario.num_runs + 1):
            result = runner.run_scenario(scenario, run_num)
            results.append(result)
            time.sleep(2.0)

    # Generate report
    generate_report(results, output_dir, controller_name="ACC Baseline", prefix="acc_test")

    pass_rate = sum(1 for r in results if r.pass_status) / len(results) if results else 0
    print(f"\n{'='*80}")
    print(f"ACC TESTING COMPLETE - Pass Rate: {pass_rate*100:.1f}%")
    print(f"{'='*80}")

    return 0 if pass_rate >= 0.8 else 1


if __name__ == '__main__':
    sys.exit(main())
