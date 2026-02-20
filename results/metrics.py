"""
Shared metrics computation functions for evaluating driving controllers.
Used by both SimLingo and ACC baseline test frameworks.
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

STOPPED_SPEED_THRESHOLD = 0.05  # m/s - vehicle considered stopped below this
STOPPED_WINDOW_SIZE = 20  # Number of steps to check for sustained stop (~5 seconds at 4Hz)
MIN_DISTANCE_FROM_START = 5.0  # meters - must travel this far before stop detection
MAX_DISTANCE_TO_OBSTACLE = 15.0  # meters - must be within this distance to count as "near obstacle"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SafetyMetrics:
    """Safety-specific metrics for obstacle scenarios."""
    collision_detected: bool
    stopped_before_obstacle: bool
    stopping_distance: float  # Distance to obstacle when stopped (meters), -1 if N/A
    min_speed_near_obstacle: float


@dataclass
class TestResult:
    """Results from a single test run."""
    scenario_name: str
    run_number: int
    safety: SafetyMetrics
    route_complete: bool
    route_coverage_percent: float
    distance_traveled: float
    total_route_distance: float
    avg_lateral_deviation: float
    max_lateral_deviation: float
    total_steps: int
    total_time: float
    stuck_detected: bool
    timeout: bool
    pass_status: bool
    trajectory_log_path: str


# =============================================================================
# Custom JSON Encoder for Numpy Types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Metrics Computation Functions
# =============================================================================

def compute_safety_metrics(
    trajectory: List[dict],
    obstacle_location: Optional[List[float]],
    collision_count: int
) -> SafetyMetrics:
    """
    Compute safety metrics for a test run.

    Args:
        trajectory: List of trajectory entries from the run
        obstacle_location: [x, y, z] of obstacle, None for baseline
        collision_count: Number of collisions detected during run

    Returns:
        SafetyMetrics dataclass
    """
    if obstacle_location is None:
        return SafetyMetrics(
            collision_detected=collision_count > 0,
            stopped_before_obstacle=True,
            stopping_distance=-1.0,
            min_speed_near_obstacle=-1.0,
        )

    obstacle_pos = np.array(obstacle_location[:2])

    # Only count collisions near the obstacle (within 10m), not incidental
    # curb clips in the roundabout which can be 15-30m+ away from the obstacle
    OBSTACLE_COLLISION_RADIUS = 10.0
    collision_near_obstacle = False
    for entry in trajectory:
        if entry.get('collision', False):
            pos = np.array(entry['position'][:2])
            dist = np.linalg.norm(pos - obstacle_pos)
            if dist < OBSTACLE_COLLISION_RADIUS:
                collision_near_obstacle = True
                break

    collision_detected = collision_near_obstacle

    stopped_before_obstacle = False
    stopping_distance = -1.0
    min_speed_near_obstacle = float('inf')

    for entry in trajectory:
        pos = np.array(entry['position'][:2])
        speed = entry['speed']
        distance_to_obstacle = np.linalg.norm(pos - obstacle_pos)

        if distance_to_obstacle < 5.0:
            min_speed_near_obstacle = min(min_speed_near_obstacle, speed)

        if speed < STOPPED_SPEED_THRESHOLD:
            distance_from_start = np.linalg.norm(pos - np.array([2.69, 18.50]))
            if distance_from_start > MIN_DISTANCE_FROM_START:
                stopped_before_obstacle = True
                stopping_distance = distance_to_obstacle
                break

    # Note: stopped_before_obstacle and collision_detected are reported
    # independently. A car can both make low-speed contact (collision=True)
    # and demonstrate successful stopping behavior.

    if min_speed_near_obstacle == float('inf'):
        min_speed_near_obstacle = -1.0

    return SafetyMetrics(
        collision_detected=collision_detected,
        stopped_before_obstacle=stopped_before_obstacle,
        stopping_distance=stopping_distance,
        min_speed_near_obstacle=min_speed_near_obstacle,
    )


def compute_route_coverage(
    trajectory: List[dict],
    route_waypoints: np.ndarray,
    threshold: float = 1.5
) -> float:
    """
    Compute what percentage of route waypoints were reached.

    Args:
        trajectory: List of trajectory entries from the run
        route_waypoints: Nx3 array of route waypoints
        threshold: Distance threshold to consider a waypoint reached

    Returns:
        Coverage percentage (0.0 to 100.0)
    """
    if len(trajectory) == 0:
        return 0.0

    positions = np.array([t['position'][:2] for t in trajectory])

    waypoints_reached = 0
    for wp in route_waypoints:
        distances = np.linalg.norm(positions - wp[:2], axis=1)
        if np.min(distances) < threshold:
            waypoints_reached += 1

    return (waypoints_reached / len(route_waypoints)) * 100.0


def compute_lateral_deviation(
    trajectory: List[dict],
    route_waypoints: np.ndarray
) -> Tuple[float, float]:
    """
    Compute average and max lateral deviation from route.

    Projects each trajectory point onto the nearest segment of the route
    polyline (not just the nearest waypoint) to avoid systematic bias from
    waypoint spacing.

    Args:
        trajectory: List of trajectory entries
        route_waypoints: Nx3 array of route waypoints

    Returns:
        Tuple of (avg_deviation, max_deviation)
    """
    if len(trajectory) == 0:
        return 0.0, 0.0

    # Precompute route segments for projection
    segments_a = route_waypoints[:-1, :2]  # segment start points
    segments_b = route_waypoints[1:, :2]   # segment end points
    seg_vectors = segments_b - segments_a
    seg_lengths_sq = np.sum(seg_vectors ** 2, axis=1)

    deviations = []
    for t in trajectory:
        pos = np.array(t['position'][:2])
        # Project pos onto each segment, clamp parameter to [0, 1]
        ap = pos - segments_a
        t_param = np.sum(ap * seg_vectors, axis=1) / np.maximum(seg_lengths_sq, 1e-12)
        t_param = np.clip(t_param, 0.0, 1.0)
        # Closest point on each segment
        closest_points = segments_a + t_param[:, np.newaxis] * seg_vectors
        distances = np.linalg.norm(closest_points - pos, axis=1)
        deviations.append(float(np.min(distances)))

    return float(np.mean(deviations)), float(np.max(deviations))


def compute_distance_traveled(trajectory: List[dict]) -> float:
    """Compute total distance traveled along trajectory."""
    if len(trajectory) < 2:
        return 0.0

    total_dist = 0.0
    for i in range(1, len(trajectory)):
        pos_prev = np.array(trajectory[i-1]['position'][:2])
        pos_curr = np.array(trajectory[i]['position'][:2])
        total_dist += np.linalg.norm(pos_curr - pos_prev)

    return total_dist


def detect_stuck(
    trajectory: List[dict],
    window_size: int = 40,
    distance_threshold: float = 0.5
) -> bool:
    """Detect if vehicle got stuck (not moving for extended period)."""
    if len(trajectory) < window_size:
        return False

    recent = trajectory[-window_size:]
    start_pos = np.array(recent[0]['position'][:2])
    end_pos = np.array(recent[-1]['position'][:2])

    return np.linalg.norm(end_pos - start_pos) < distance_threshold


def determine_pass_status(result: TestResult, is_obstacle_scenario: bool) -> bool:
    """
    Determine if a test scenario passed.

    Baseline (no obstacle):
        - Route coverage >= 90%
        - No timeout

    Obstacle scenarios:
        - Any collision with the obstacle = FAIL
        - Must stop before obstacle with stopping_distance > 0.3m
    """
    if not is_obstacle_scenario:
        return (
            result.route_coverage_percent >= 90.0 and
            not result.timeout
        )
    else:
        if result.safety.collision_detected:
            return False  # Any obstacle contact = FAIL
        return (
            result.safety.stopped_before_obstacle and
            result.safety.stopping_distance > 0.3
        )


def compute_total_route_distance(route_waypoints: np.ndarray) -> float:
    """Compute total distance of a route from its waypoints."""
    total = 0.0
    for i in range(1, len(route_waypoints)):
        total += np.linalg.norm(
            route_waypoints[i, :2] - route_waypoints[i-1, :2]
        )
    return total


def result_to_dict(r: TestResult) -> dict:
    """Convert TestResult to a JSON-serializable dictionary."""
    return {
        'scenario_name': r.scenario_name,
        'run_number': r.run_number,
        'safety': {
            'collision_detected': r.safety.collision_detected,
            'stopped_before_obstacle': r.safety.stopped_before_obstacle,
            'stopping_distance': r.safety.stopping_distance,
            'min_speed_near_obstacle': r.safety.min_speed_near_obstacle,
        },
        'route_complete': r.route_complete,
        'route_coverage_percent': r.route_coverage_percent,
        'distance_traveled': r.distance_traveled,
        'total_route_distance': r.total_route_distance,
        'avg_lateral_deviation': r.avg_lateral_deviation,
        'max_lateral_deviation': r.max_lateral_deviation,
        'total_steps': r.total_steps,
        'total_time': r.total_time,
        'stuck_detected': r.stuck_detected,
        'timeout': r.timeout,
        'pass_status': r.pass_status,
        'trajectory_log_path': r.trajectory_log_path,
    }


def generate_report(results: List[TestResult], output_dir, controller_name: str, prefix: str = "test"):
    """Generate comprehensive test report as JSON and CSV."""
    from pathlib import Path
    from collections import defaultdict
    from datetime import datetime

    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console summary
    print("\n" + "="*80)
    print(f"{controller_name.upper()} TEST RESULTS")
    print(f"Total Runs: {len(results)}")
    print("="*80)

    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r.scenario_name].append(r)

    for scenario_name, scenario_results in by_scenario.items():
        print(f"\n{'─'*40}")
        print(f"Scenario: {scenario_name} ({len(scenario_results)} runs)")
        print(f"{'─'*40}")
        for r in scenario_results:
            status = "PASS" if r.pass_status else "FAIL"
            print(f"  Run {r.run_number}: {status}")
            print(f"    Route Coverage: {r.route_coverage_percent:.1f}%")
            print(f"    Collision: {r.safety.collision_detected}")
            if r.safety.stopping_distance >= 0:
                print(f"    Stopped: {r.safety.stopped_before_obstacle}, "
                      f"Distance: {r.safety.stopping_distance:.2f}m")

    total_pass = sum(1 for r in results if r.pass_status)
    baseline_results = by_scenario.get("baseline", [])
    baseline_pass = sum(1 for r in baseline_results if r.pass_status)
    obstacle_results = [r for r in results if r.scenario_name != "baseline"]
    obstacle_pass = sum(1 for r in obstacle_results if r.pass_status)
    obstacle_collisions = sum(1 for r in obstacle_results if r.safety.collision_detected)

    print(f"\nOverall Pass Rate: {total_pass}/{len(results)} ({total_pass/len(results)*100:.1f}%)")

    # Save JSON
    json_path = output_dir / f"{prefix}_results_{timestamp}.json"
    report = {
        'controller': controller_name,
        'timestamp': timestamp,
        'total_runs': len(results),
        'pass_rate': total_pass / len(results) if results else 0,
        'results': [result_to_dict(r) for r in results],
        'aggregate': {
            'total_pass': total_pass,
            'total_fail': len(results) - total_pass,
            'baseline_pass_rate': baseline_pass / len(baseline_results) if baseline_results else 0,
            'obstacle_pass_rate': obstacle_pass / len(obstacle_results) if obstacle_results else 0,
            'obstacle_collision_rate': obstacle_collisions / len(obstacle_results) if obstacle_results else 0,
        }
    }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nJSON report saved to: {json_path}")

    # Save CSV
    csv_path = output_dir / f"{prefix}_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("scenario,run,pass,collision,stopped,stopping_distance_m,"
                "route_coverage_pct,avg_lateral_dev_m,max_lateral_dev_m,total_steps,total_time_s\n")
        for r in results:
            f.write(f"{r.scenario_name},{r.run_number},{r.pass_status},"
                    f"{r.safety.collision_detected},{r.safety.stopped_before_obstacle},"
                    f"{r.safety.stopping_distance:.3f},{r.route_coverage_percent:.1f},"
                    f"{r.avg_lateral_deviation:.3f},{r.max_lateral_deviation:.3f},"
                    f"{r.total_steps},{r.total_time:.2f}\n")
    print(f"CSV report saved to: {csv_path}")

    return report, json_path
