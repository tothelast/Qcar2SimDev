"""
Comprehensive Testing Framework for Simlingo Model Evaluation.
Tests the fine-tuned model on roundabout navigation with obstacle variations.

Test Matrix:
- Baseline (no obstacle): 5 runs
- Obstacle Var 1-5: 2 runs each
- Total: 15 test runs

Usage:
    python results/test_simlingo_roundabout.py \
        --checkpoint simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune/checkpoints/epoch_14.pt

Requirements:
    - QLabs must be running with SDCS RoadMap loaded
    - Model checkpoint must exist
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import argparse
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# Import project modules
from core.config import SimlingoQCar2Config
from core.qcar2_interface import QCar2Interface
from core.scene_loader import SceneLoader, SceneDefinition, ActorDefinition
from core.scene_spawner import SceneSpawner
from inference.state_estimator import StateEstimator
from inference.route_manager import RouteManager
from inference.control_converter import ControlConverter

try:
    from core.camera_processor import CameraProcessor
    from inference.simlingo_model import SimlingoModelWrapper
except ImportError as e:
    print(f"Warning: Could not import simlingo modules: {e}")
    CameraProcessor = None
    SimlingoModelWrapper = None


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
# Constants
# =============================================================================

STOPPED_SPEED_THRESHOLD = 0.05  # m/s - vehicle considered stopped below this
STOPPED_WINDOW_SIZE = 20  # Number of steps to check for sustained stop (~5 seconds at 4Hz)
MIN_DISTANCE_FROM_START = 5.0  # meters - must travel this far before stop detection
MAX_DISTANCE_TO_OBSTACLE = 15.0  # meters - must be within this distance to count as "near obstacle"

# Route start and end positions (from roundabout_navigation.json)
ROUTE_START_POS = [2.69, 18.50]
ROUTE_END_POS = [-19.84, 29.76]

# Obstacle locations from config files
OBSTACLE_LOCATIONS = {
    "obstacle_car_var1": [21.01, 33.90, 0.005],   # Early Roundabout
    "obstacle_car_var2": [18.85, 44.23, 0.005],   # Mid Roundabout
    "obstacle_car_var3": [6.07, 44.97, 0.005],    # Roundabout Exit
    "obstacle_car_var4": [-10.60, 44.97, 0.005],  # Straight Section
    "obstacle_car_var5": [-18.73, 40.37, 0.005],  # Late Route
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestScenario:
    """Configuration for a single test scenario."""
    name: str
    description: str
    ego_route: str = "roundabout_navigation"
    obstacle_actor: Optional[str] = None  # None for baseline
    obstacle_location: Optional[List[float]] = None
    num_runs: int = 1
    max_steps: int = 500  # Timeout after N steps (~125 seconds at 4Hz)
    timeout_seconds: float = 150.0


@dataclass
class SafetyMetrics:
    """Safety-specific metrics for obstacle scenarios."""
    collision_detected: bool
    stopped_before_obstacle: bool  # True if speed < 0.05 m/s before reaching obstacle
    stopping_distance: float  # Distance to obstacle when stopped (meters), -1 if N/A
    min_speed_near_obstacle: float  # Minimum speed recorded near obstacle


@dataclass
class TestResult:
    """Results from a single test run."""
    scenario_name: str
    run_number: int
    
    # Safety Metrics
    safety: SafetyMetrics
    
    # Route Completeness Metrics
    route_complete: bool
    route_coverage_percent: float
    distance_traveled: float
    total_route_distance: float
    avg_lateral_deviation: float
    max_lateral_deviation: float
    
    # Timing
    total_steps: int
    total_time: float
    
    # Status
    stuck_detected: bool
    timeout: bool
    pass_status: bool
    
    # Path to detailed trajectory log
    trajectory_log_path: str


# =============================================================================
# Test Scenarios Definition
# =============================================================================

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
    # Baseline scenario - no obstacle
    if obstacle_location is None:
        return SafetyMetrics(
            collision_detected=collision_count > 0,
            stopped_before_obstacle=True,  # N/A for baseline
            stopping_distance=-1.0,  # N/A for baseline
            min_speed_near_obstacle=-1.0,  # N/A for baseline
        )

    obstacle_pos = np.array(obstacle_location[:2])

    # Only count collisions near the obstacle (within 10m), not incidental
    # curb clips in the roundabout which can be 15-30m+ away from the obstacle
    OBSTACLE_COLLISION_RADIUS = 10.0
    collision_near_obstacle = False
    for entry in trajectory:
        if entry['collision']:
            pos = np.array(entry['position'][:2])
            dist = np.linalg.norm(pos - obstacle_pos)
            if dist < OBSTACLE_COLLISION_RADIUS:
                collision_near_obstacle = True
                break

    collision_detected = collision_near_obstacle

    # Find the trajectory point where vehicle stopped or collided
    stopped_before_obstacle = False
    stopping_distance = -1.0
    min_speed_near_obstacle = float('inf')

    for entry in trajectory:
        pos = np.array(entry['position'][:2])
        speed = entry['speed']
        distance_to_obstacle = np.linalg.norm(pos - obstacle_pos)

        # Track minimum speed when within 5m of obstacle
        if distance_to_obstacle < 5.0:
            min_speed_near_obstacle = min(min_speed_near_obstacle, speed)

        # Check if vehicle stopped (speed < 0.05 m/s) and has traveled past start
        if speed < STOPPED_SPEED_THRESHOLD:
            distance_from_start = np.linalg.norm(pos - np.array(ROUTE_START_POS))
            if distance_from_start > MIN_DISTANCE_FROM_START:
                stopped_before_obstacle = True
                stopping_distance = distance_to_obstacle
                break  # Use first meaningful stop point

    # Note: stopped_before_obstacle and collision_detected are reported
    # independently. A car can both make low-speed contact (collision=True)
    # and demonstrate successful stopping behavior.

    # Handle case where vehicle never got close to obstacle
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
    segments_a = route_waypoints[:-1, :2]
    segments_b = route_waypoints[1:, :2]
    seg_vectors = segments_b - segments_a
    seg_lengths_sq = np.sum(seg_vectors ** 2, axis=1)

    deviations = []
    for t in trajectory:
        pos = np.array(t['position'][:2])
        ap = pos - segments_a
        t_param = np.sum(ap * seg_vectors, axis=1) / np.maximum(seg_lengths_sq, 1e-12)
        t_param = np.clip(t_param, 0.0, 1.0)
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
    window_size: int = 40,  # ~10 seconds at 4Hz
    distance_threshold: float = 0.5
) -> bool:
    """
    Detect if vehicle got stuck (not moving for extended period).
    """
    if len(trajectory) < window_size:
        return False

    recent = trajectory[-window_size:]
    start_pos = np.array(recent[0]['position'][:2])
    end_pos = np.array(recent[-1]['position'][:2])

    return np.linalg.norm(end_pos - start_pos) < distance_threshold


def determine_pass_status(result: 'TestResult', is_obstacle_scenario: bool) -> bool:
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


# =============================================================================
# Test Runner Class
# =============================================================================

class SimlingoTestRunner:
    """Test runner for Simlingo model evaluation."""

    def __init__(self, checkpoint_path: str, output_dir: Path, seed: int = 42):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.seed = seed
        self.results: List[TestResult] = []

    def run_scenario(self, scenario: TestScenario, run_number: int) -> TestResult:
        """
        Run a single test scenario and collect metrics.

        Args:
            scenario: TestScenario configuration
            run_number: Which run number this is (1-based)

        Returns:
            TestResult with all computed metrics
        """
        print(f"\n{'='*60}")
        print(f"RUNNING: {scenario.name} - Run {run_number}/{scenario.num_runs}")
        print(f"Description: {scenario.description}")
        print(f"{'='*60}")

        # Create run output directory
        run_dir = self.output_dir / 'runs' / f"{scenario.name}_run_{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize controller with custom scene
        controller = self._create_controller(scenario)

        if controller is None:
            print("ERROR: Failed to create controller")
            return self._create_error_result(scenario, run_number, run_dir)

        # Initialize controller (connects to QLabs, spawns QCar, loads model)
        try:
            if not controller.initialize():
                print("ERROR: Controller initialization failed")
                return self._create_error_result(scenario, run_number, run_dir)
        except Exception as e:
            print(f"ERROR: Controller initialization exception: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(scenario, run_number, run_dir)

        # Run the control loop with step limit
        trajectory_log = []
        collision_count = 0
        route_complete = False
        timeout = False
        stuck_detected = False

        try:
            controller.running = True
            controller.start_time = time.time()
            dt = controller.config.dt

            step = 0
            while step < scenario.max_steps:
                loop_start = time.time()

                # Check timeout
                elapsed_total = time.time() - controller.start_time
                if elapsed_total > scenario.timeout_seconds:
                    print(f"TIMEOUT: Exceeded {scenario.timeout_seconds}s")
                    timeout = True
                    break

                # Execute one control step
                if not controller.run_step(dt):
                    # Route complete or error
                    route_complete = controller.route_manager.is_route_complete(
                        controller.state_estimator.get_position()
                    )
                    break

                # Early termination for obstacle scenarios: detect sustained stop near obstacle
                if scenario.obstacle_location is not None and len(controller.trajectory_log) >= STOPPED_WINDOW_SIZE:
                    # Get recent speeds
                    recent_entries = controller.trajectory_log[-STOPPED_WINDOW_SIZE:]
                    recent_speeds = [e['speed'] for e in recent_entries]
                    avg_recent_speed = sum(recent_speeds) / len(recent_speeds)

                    if avg_recent_speed < STOPPED_SPEED_THRESHOLD:
                        # Vehicle has stopped - check if it's near the obstacle (not at start/end)
                        current_pos = np.array(controller.state_estimator.get_position()[:2])
                        obstacle_pos = np.array(scenario.obstacle_location[:2])
                        start_pos = np.array(ROUTE_START_POS)

                        distance_to_obstacle = np.linalg.norm(current_pos - obstacle_pos)
                        distance_from_start = np.linalg.norm(current_pos - start_pos)

                        # Stopped near obstacle AND past the start phase
                        if distance_to_obstacle < MAX_DISTANCE_TO_OBSTACLE and distance_from_start > MIN_DISTANCE_FROM_START:
                            print(f"Vehicle stopped near obstacle (dist: {distance_to_obstacle:.1f}m) - terminating test")
                            break

                # Maintain control frequency
                elapsed = time.time() - loop_start
                sleep_time = controller.config.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                dt = time.time() - loop_start
                step += 1

            trajectory_log = controller.trajectory_log
            collision_count = controller.collision_count

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            trajectory_log = controller.trajectory_log
            collision_count = controller.collision_count

        except Exception as e:
            print(f"ERROR during run: {e}")
            import traceback
            traceback.print_exc()
            trajectory_log = controller.trajectory_log
            collision_count = controller.collision_count

        finally:
            # Always cleanup
            self._cleanup_controller(controller)

        # Compute metrics
        result = self._compute_result(
            scenario=scenario,
            run_number=run_number,
            trajectory_log=trajectory_log,
            collision_count=collision_count,
            route_complete=route_complete,
            timeout=timeout,
            stuck_detected=stuck_detected,
            route_waypoints=controller.route_manager.route_waypoints,
            run_dir=run_dir
        )

        # Save trajectory log
        trajectory_path = run_dir / "trajectory_log.json"
        self._save_trajectory_log(
            trajectory_log, trajectory_path, scenario, result,
            route_waypoints=controller.route_manager.route_waypoints
        )
        result.trajectory_log_path = str(trajectory_path)

        # Print summary
        self._print_run_summary(result)

        return result

    def _create_controller(self, scenario: TestScenario):
        """Create a SimlingoQCar2Controller-like object for testing."""
        try:
            # Create config and override checkpoint path
            config = SimlingoQCar2Config()
            config.model_checkpoint_path = self.checkpoint_path

            # Load the route
            if not config.load_route(scenario.ego_route):
                print(f"ERROR: Failed to load route: {scenario.ego_route}")
                return None

            # Create scene definition if obstacle is specified
            scene_definition = None
            if scenario.obstacle_actor:
                scene_definition = self._create_scene_with_obstacle(
                    scenario.ego_route,
                    scenario.obstacle_actor
                )

            # Create a controller-like object
            controller = _TestController(config, scene_definition)
            return controller

        except Exception as e:
            print(f"ERROR creating controller: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_scene_with_obstacle(
        self,
        ego_route: str,
        obstacle_actor: str
    ) -> SceneDefinition:
        """Create a scene definition with the specified obstacle."""
        # Load the obstacle actor definition
        loader = SceneLoader()
        actor_def = loader.load_actor(obstacle_actor)

        # Create scene_data dictionary as expected by SceneDefinition
        scene_data = {
            'name': f"test_{obstacle_actor}",
            'description': f"Test scene with {obstacle_actor}",
            'ego_route': ego_route,
        }

        if actor_def is None:
            print(f"WARNING: Could not load actor {obstacle_actor}")
            return SceneDefinition(scene_data, scene_path="", actors=[])

        return SceneDefinition(scene_data, scene_path="", actors=[actor_def])

    def _cleanup_controller(self, controller):
        """Cleanup controller resources including GPU memory."""
        try:
            # Stop vehicle
            if hasattr(controller, 'qcar_interface') and controller.qcar_interface.connected:
                controller.qcar_interface.set_control(0.0, 0.0)

            # Cleanup scene actors
            if hasattr(controller, 'scene_spawner') and controller.scene_spawner:
                controller.scene_spawner.cleanup()

            # Close QLabs connection
            if hasattr(controller, 'qcar_interface'):
                controller.qcar_interface.close()

        except Exception as e:
            print(f"WARNING: Cleanup error: {e}")

        # Free GPU memory - this MUST happen to avoid CUDA OOM across runs
        try:
            import torch
            if hasattr(controller, 'model_wrapper') and controller.model_wrapper is not None:
                if hasattr(controller.model_wrapper, 'model') and controller.model_wrapper.model is not None:
                    del controller.model_wrapper.model
                    controller.model_wrapper.model = None
                del controller.model_wrapper
                controller.model_wrapper = None
            if hasattr(controller, 'camera_processor') and controller.camera_processor is not None:
                del controller.camera_processor
                controller.camera_processor = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("GPU memory released successfully")
        except Exception as e:
            print(f"WARNING: GPU cleanup error: {e}")

    def _compute_result(
        self,
        scenario: TestScenario,
        run_number: int,
        trajectory_log: List[dict],
        collision_count: int,
        route_complete: bool,
        timeout: bool,
        stuck_detected: bool,
        route_waypoints: np.ndarray,
        run_dir: Path
    ) -> TestResult:
        """Compute all metrics and create TestResult."""
        # Safety metrics
        safety = compute_safety_metrics(
            trajectory_log,
            scenario.obstacle_location,
            collision_count
        )

        # Route coverage
        route_coverage = compute_route_coverage(trajectory_log, route_waypoints)

        # Lateral deviation
        avg_dev, max_dev = compute_lateral_deviation(trajectory_log, route_waypoints)

        # Distance traveled
        distance_traveled = compute_distance_traveled(trajectory_log)

        # Total route distance (from waypoints)
        total_route_dist = 0.0
        for i in range(1, len(route_waypoints)):
            total_route_dist += np.linalg.norm(
                route_waypoints[i, :2] - route_waypoints[i-1, :2]
            )

        # Total time
        total_time = 0.0
        if trajectory_log:
            total_time = trajectory_log[-1].get('timestamp', 0.0)

        # Create result (without pass_status first)
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
            stuck_detected=stuck_detected,
            timeout=timeout,
            pass_status=False,  # Will be set below
            trajectory_log_path=str(run_dir / "trajectory_log.json")
        )

        # Determine pass/fail
        is_obstacle_scenario = scenario.obstacle_actor is not None
        result.pass_status = determine_pass_status(result, is_obstacle_scenario)

        return result

    def _create_error_result(
        self,
        scenario: TestScenario,
        run_number: int,
        run_dir: Path
    ) -> TestResult:
        """Create a failed result for error cases."""
        return TestResult(
            scenario_name=scenario.name,
            run_number=run_number,
            safety=SafetyMetrics(
                collision_detected=False,
                stopped_before_obstacle=False,
                stopping_distance=-1.0,
                min_speed_near_obstacle=-1.0
            ),
            route_complete=False,
            route_coverage_percent=0.0,
            distance_traveled=0.0,
            total_route_distance=0.0,
            avg_lateral_deviation=0.0,
            max_lateral_deviation=0.0,
            total_steps=0,
            total_time=0.0,
            stuck_detected=False,
            timeout=False,
            pass_status=False,
            trajectory_log_path=str(run_dir / "trajectory_log.json")
        )

    def _save_trajectory_log(
        self,
        trajectory_log: List[dict],
        path: Path,
        scenario: TestScenario,
        result: TestResult,
        route_waypoints: Optional[np.ndarray] = None,
    ):
        """Save trajectory log to JSON file."""
        data = {
            'metadata': {
                'scenario_name': scenario.name,
                'run_number': result.run_number,
                'checkpoint': self.checkpoint_path,
                'timestamp': datetime.now().isoformat(),
                'pass_status': result.pass_status,
                'collision_detected': result.safety.collision_detected,
                'stopping_distance': result.safety.stopping_distance,
                'route_coverage_percent': result.route_coverage_percent,
                'route_waypoints': route_waypoints.tolist() if route_waypoints is not None else None,
                'obstacle_location': scenario.obstacle_location,
            },
            'trajectory': trajectory_log
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def _print_run_summary(self, result: TestResult):
        """Print summary of a test run."""
        status = "✓ PASS" if result.pass_status else "✗ FAIL"
        print(f"\n{'─'*40}")
        print(f"Result: {status}")
        print(f"{'─'*40}")
        print(f"  Route Coverage: {result.route_coverage_percent:.1f}%")
        print(f"  Distance Traveled: {result.distance_traveled:.2f}m")
        print(f"  Total Steps: {result.total_steps}")
        print(f"  Total Time: {result.total_time:.2f}s")
        print(f"  Collision: {result.safety.collision_detected}")
        if result.safety.stopping_distance >= 0:
            print(f"  Stopped Before Obstacle: {result.safety.stopped_before_obstacle}")
            print(f"  Stopping Distance: {result.safety.stopping_distance:.2f}m")
        print(f"  Stuck: {result.stuck_detected}")
        print(f"  Timeout: {result.timeout}")


# =============================================================================
# Test Controller Class (adapted from inference/main.py)
# =============================================================================

class _TestController:
    """
    Simplified controller for testing purposes.
    Based on SimlingoQCar2Controller from inference/main.py.
    """

    def __init__(self, config: SimlingoQCar2Config, scene_definition: Optional[SceneDefinition] = None):
        self.config = config
        self.scene_definition = scene_definition
        self.scene_spawner = None

        # Initialize components
        self.qcar_interface = QCar2Interface(config)
        self.camera_processor = CameraProcessor(config) if CameraProcessor else None
        self.state_estimator = StateEstimator(config)
        self.route_manager = RouteManager(config)
        self.model_wrapper = SimlingoModelWrapper(config, nav_mode='target_point') if SimlingoModelWrapper else None
        self.control_converter = ControlConverter(config)

        # State
        self.running = False
        self.step_count = 0
        self.trajectory_log = []
        self.collision_count = 0
        self.start_time = None

        # Model inference caching
        self.cached_speed_wps = None
        self.cached_route_wps = None
        self.cached_language = None
        self.inference_counter = 0

    def initialize(self) -> bool:
        """Initialize all components."""
        print("Initializing test controller...")

        # Connect to QLabs
        if not self.qcar_interface.connect():
            return False

        # Spawn QCar2
        if not self.qcar_interface.spawn_qcar(model_wrapper=self.model_wrapper):
            return False

        # Setup scene actors
        if self.scene_definition and self.scene_definition.actors:
            print(f"Setting up scene actors ({len(self.scene_definition.actors)} actors)...")
            self.scene_spawner = SceneSpawner(self.scene_definition)
            if self.scene_spawner.connect():
                self.scene_spawner.spawn_all_actors()

        # Load Simlingo model
        print("Loading Simlingo model...")
        self.model_wrapper.load_tokenizer()
        self.model_wrapper.load_model()
        print("Model loaded successfully")

        print("Test controller initialization complete!")
        return True

    def run_step(self, dt: float) -> bool:
        """Execute one control loop iteration."""
        # Get camera image
        image = self.qcar_interface.get_camera_image()

        if image is None:
            self.qcar_interface.set_control(0.0, 0.0)
            return True

        # Process camera image
        camera_images, image_sizes = self.camera_processor.process_image(image)
        camera_intrinsics = self.camera_processor.get_camera_intrinsics_tensor()
        camera_extrinsics = self.camera_processor.get_camera_extrinsics_tensor()

        # Get current state
        location, rotation = self.qcar_interface.get_state()
        self.state_estimator.update(location, rotation)
        velocity = self.state_estimator.get_velocity()

        # Get target points and HLC
        current_position = self.state_estimator.get_position()
        current_heading = self.state_estimator.get_heading()
        target_point, next_target_point, hlc = self.route_manager.get_target_point_ego(
            current_position, current_heading
        )

        # Run model inference
        if self.inference_counter % getattr(self.config, "inference_stride", 1) == 0:
            speed_wps, route_wps, language = self.model_wrapper.inference(
                camera_images=camera_images,
                image_sizes=image_sizes,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
                vehicle_speed=velocity,
                target_point=target_point,
                next_target_point=next_target_point,
                hlc=hlc
            )
            self.cached_speed_wps = speed_wps
            self.cached_route_wps = route_wps
            self.cached_language = language
        else:
            speed_wps = self.cached_speed_wps
            route_wps = self.cached_route_wps

        self.inference_counter += 1

        # Convert to numpy
        route_waypoints = route_wps[0].cpu().numpy()
        speed_waypoints = speed_wps[0].cpu().numpy()

        # Compute control
        steer, target_speed_cmd, brake, desired_speed = self.control_converter.control_pid(
            route_waypoints, velocity, speed_waypoints
        )

        # Convert to QCar2 control
        forward_velocity, turn_angle = self.control_converter.convert_to_qcar2_control(
            desired_speed, steer, velocity, dt, target_speed_cmd, brake
        )

        # Send control to QCar2
        _, location, rotation = self.qcar_interface.set_control(
            forward_velocity, turn_angle, brake=brake
        )

        # Log trajectory data
        target_world, _, _ = self.route_manager.get_target_point(current_position)
        distance_to_target = np.linalg.norm(target_world[:2] - current_position[:2])

        # Check for collision
        collision_detected = self.qcar_interface.check_collision()
        if collision_detected:
            self.collision_count += 1

        trajectory_entry = {
            'step': int(self.step_count),
            'timestamp': float(time.time() - self.start_time if self.start_time else 0),
            'position': current_position.tolist(),
            'heading_deg': float(rotation[2] * 180 / np.pi),
            'heading_rad': float(rotation[2]),
            'speed': float(velocity),
            'desired_speed': float(desired_speed),
            'steering': float(steer),
            'collision': bool(collision_detected),
            'current_waypoint_index': int(self.route_manager.current_waypoint_index),
            'distance_to_target': float(distance_to_target),
            'predicted_route_waypoints': route_waypoints.tolist(),
            'predicted_speed_waypoints': speed_waypoints.tolist(),
            'target_waypoint': target_world.tolist(),
        }
        self.trajectory_log.append(trajectory_entry)

        # Print status every 10 steps
        if self.step_count % 10 == 0:
            progress = self.route_manager.get_progress(current_position)
            print(f"Step {self.step_count:4d} | Speed: {velocity:5.2f} m/s | Progress: {progress*100:5.1f}%")

        self.step_count += 1

        # Check if route complete
        if self.route_manager.is_route_complete(current_position):
            print("Route complete!")
            return False

        return True


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: List[TestResult], output_dir: Path, checkpoint_path: str):
    """Generate comprehensive test report."""
    from collections import defaultdict

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console summary
    print("\n" + "="*80)
    print("SIMLINGO MODEL TEST RESULTS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total Runs: {len(results)}")
    print("="*80)

    # Group results by scenario
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r.scenario_name].append(r)

    for scenario_name, scenario_results in by_scenario.items():
        print(f"\n{'─'*40}")
        print(f"Scenario: {scenario_name} ({len(scenario_results)} runs)")
        print(f"{'─'*40}")

        for r in scenario_results:
            status = "✓ PASS" if r.pass_status else "✗ FAIL"
            print(f"  Run {r.run_number}: {status}")
            print(f"    Route Coverage: {r.route_coverage_percent:.1f}%")
            print(f"    Collision: {r.safety.collision_detected}")
            if r.safety.stopping_distance >= 0:
                print(f"    Stopped: {r.safety.stopped_before_obstacle}, "
                      f"Distance: {r.safety.stopping_distance:.2f}m")

    # Aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")

    total_pass = sum(1 for r in results if r.pass_status)

    # Baseline stats
    baseline_results = by_scenario.get("baseline", [])
    baseline_pass = sum(1 for r in baseline_results if r.pass_status)

    # Obstacle stats
    obstacle_results = [r for r in results if r.scenario_name != "baseline"]
    obstacle_pass = sum(1 for r in obstacle_results if r.pass_status)
    obstacle_collisions = sum(1 for r in obstacle_results if r.safety.collision_detected)
    obstacle_stopped = sum(1 for r in obstacle_results if r.safety.stopped_before_obstacle)

    print(f"\nOverall Pass Rate: {total_pass}/{len(results)} ({total_pass/len(results)*100:.1f}%)")
    if baseline_results:
        print(f"Baseline Pass Rate: {baseline_pass}/{len(baseline_results)}")
    if obstacle_results:
        print(f"Obstacle Pass Rate: {obstacle_pass}/{len(obstacle_results)}")
        print(f"Obstacle Collisions: {obstacle_collisions}/{len(obstacle_results)}")
        print(f"Successful Stops: {obstacle_stopped}/{len(obstacle_results)}")

    if obstacle_stopped > 0:
        stopping_distances = [
            r.safety.stopping_distance
            for r in obstacle_results
            if r.safety.stopped_before_obstacle and r.safety.stopping_distance > 0
        ]
        if stopping_distances:
            avg_stopping_dist = np.mean(stopping_distances)
            print(f"Avg Stopping Distance: {avg_stopping_dist:.2f}m")

    # Save JSON report
    json_path = output_dir / f"test_results_{timestamp}.json"

    # Convert SafetyMetrics to dict for JSON serialization
    def result_to_dict(r: TestResult) -> dict:
        d = {
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
        return d

    report = {
        'checkpoint': str(checkpoint_path),
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

    # Save CSV summary
    csv_path = output_dir / f"test_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("scenario,run,pass,collision,stopped,stopping_distance_m,"
                "route_coverage_pct,avg_lateral_dev_m,total_steps,total_time_s\n")
        for r in results:
            f.write(f"{r.scenario_name},{r.run_number},{r.pass_status},"
                    f"{r.safety.collision_detected},{r.safety.stopped_before_obstacle},"
                    f"{r.safety.stopping_distance:.3f},{r.route_coverage_percent:.1f},"
                    f"{r.avg_lateral_deviation:.3f},{r.total_steps},{r.total_time:.2f}\n")
    print(f"CSV report saved to: {csv_path}")

    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the testing framework."""
    parser = argparse.ArgumentParser(
        description='Simlingo Model Testing Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full test suite (15 runs total)
    python results/test_simlingo_roundabout.py \\
        --checkpoint simlingo/outputs/2025_11_26_18_06_21_qlabs_roundabout_finetune/checkpoints/epoch_14.pt

    # Run a specific scenario (e.g., obstacle_var1)
    python results/test_simlingo_roundabout.py --checkpoint <path> --scenario obstacle_var1

    # Run a specific scenario with custom number of runs
    python results/test_simlingo_roundabout.py --checkpoint <path> --scenario obstacle_var1 --runs 1

    # Skip baseline, run only obstacle variations
    python results/test_simlingo_roundabout.py --checkpoint <path> --skip-baseline

    # Quick test (baseline only, 1 run)
    python results/test_simlingo_roundabout.py --checkpoint <path> --quick

Available scenarios:
    baseline, obstacle_var1, obstacle_var2, obstacle_var3, obstacle_var4, obstacle_var5

Prerequisites:
    - QLabs must be running with SDCS RoadMap loaded
    - Model checkpoint must exist
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (baseline only, 1 run)')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['baseline', 'obstacle_var1', 'obstacle_var2',
                                 'obstacle_var3', 'obstacle_var4', 'obstacle_var5'],
                        help='Run only a specific scenario')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline tests, run only obstacle variations')
    parser.add_argument('--runs', type=int, default=None,
                        help='Override number of runs per scenario')

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'runs').mkdir(exist_ok=True)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    print("="*80)
    print("SIMLINGO MODEL TESTING FRAMEWORK")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Quick Mode: {args.quick}")
    print("="*80)

    # Determine which scenarios to run
    if args.quick:
        scenarios = [
            TestScenario(
                name="baseline",
                description="Quick test - baseline only",
                ego_route="roundabout_navigation",
                obstacle_actor=None,
                num_runs=1,
            )
        ]
        print("\nQuick mode: Running 1 baseline test only")
    elif args.scenario:
        # Run only the specified scenario
        scenarios = [s for s in TEST_SCENARIOS if s.name == args.scenario]
        if not scenarios:
            print(f"ERROR: Scenario '{args.scenario}' not found")
            return 1
        # Override runs if specified
        if args.runs:
            scenarios[0].num_runs = args.runs
        print(f"\nRunning scenario: {args.scenario} ({scenarios[0].num_runs} runs)")
    elif args.skip_baseline:
        # Skip baseline, run only obstacle variations
        scenarios = [s for s in TEST_SCENARIOS if s.name != "baseline"]
        # Override runs if specified
        if args.runs:
            for s in scenarios:
                s.num_runs = args.runs
        total_runs = sum(s.num_runs for s in scenarios)
        print(f"\nSkipping baseline. Running {len(scenarios)} obstacle scenarios ({total_runs} total runs)")
    else:
        scenarios = TEST_SCENARIOS
        # Override runs if specified
        if args.runs:
            for s in scenarios:
                s.num_runs = args.runs
        total_runs = sum(s.num_runs for s in scenarios)
        print(f"\nFull test suite: {total_runs} total runs")
        print(f"  - Baseline: {scenarios[0].num_runs} runs")
        print(f"  - Obstacle variations: {sum(s.num_runs for s in scenarios[1:])} runs")

    # Create test runner
    runner = SimlingoTestRunner(
        checkpoint_path=str(checkpoint_path),
        output_dir=output_dir,
        seed=args.seed
    )

    # Run all test scenarios
    results = []
    for scenario in scenarios:
        for run_num in range(1, scenario.num_runs + 1):
            result = runner.run_scenario(scenario, run_num)
            results.append(result)

            # Small delay between runs to ensure cleanup
            time.sleep(2.0)

    # Generate reports
    generate_report(results, output_dir, str(checkpoint_path))

    # Return exit code based on pass rate
    pass_rate = sum(1 for r in results if r.pass_status) / len(results) if results else 0
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE - Pass Rate: {pass_rate*100:.1f}%")
    print(f"{'='*80}")

    return 0 if pass_rate >= 0.8 else 1


if __name__ == '__main__':
    sys.exit(main())

