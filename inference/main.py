"""
Main entry point for Simlingo-QCar2 integration.
Run with: python inference/main.py
"""

import sys
import os
import time
import numpy as np
import argparse
import json
from datetime import datetime

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import SimlingoQCar2Config
from core.qcar2_interface import QCar2Interface
from core.camera_processor import CameraProcessor
from core.scene_loader import SceneLoader
from core.scene_spawner import SceneSpawner
from state_estimator import StateEstimator
from route_manager import RouteManager
from simlingo_model import SimlingoModelWrapper
from control_converter import ControlConverter


class SimlingoQCar2Controller:
    """Main controller for Simlingo-QCar2 integration."""
    
    def __init__(self, config_path=None, nav_mode='target_point', scene_name=None):
        """
        Initialize controller.

        Args:
            config_path: Path to custom config file (optional)
            nav_mode: Navigational conditioning mode ('target_point' or 'command')
            scene_name: Name of the scene to load from scenes/ directory (optional)
        """
        # Load configuration
        self.config = SimlingoQCar2Config()

        # Load scene if specified
        self.scene_definition = None
        self.scene_spawner = None

        if scene_name:
            print(f"\nLoading scene: {scene_name}")
            scene_loader = SceneLoader()
            self.scene_definition = scene_loader.load_scene(scene_name)

            if not self.scene_definition:
                raise RuntimeError(f"Failed to load scene: {scene_name}")

            print(f"Scene loaded: {self.scene_definition}")

            # Load route from scene definition
            route_name = self.scene_definition.ego_route
            print(f"Loading route from scene: {route_name}")
        else:
            # Default route if no scene specified
            route_name = 'simple_straight'
            print(f"\nNo scene specified, using default route: {route_name}")

        # Load route from JSON file
        if not self.config.load_route(route_name):
            raise RuntimeError(f"Failed to load route: {route_name}")

        self.nav_mode = nav_mode
        self.route_name = route_name
        self.scene_name = scene_name

        # Initialize components
        self.qcar_interface = QCar2Interface(self.config)
        self.camera_processor = CameraProcessor(self.config)
        self.state_estimator = StateEstimator(self.config)
        self.route_manager = RouteManager(self.config)
        self.model_wrapper = SimlingoModelWrapper(self.config, nav_mode=nav_mode)
        self.control_converter = ControlConverter(self.config)

        # Control loop state
        self.running = False
        self.step_count = 0
        self.stuck_detector = 0
        self.force_move = 0

        # Trajectory logging
        self.trajectory_log = []
        self.collision_count = 0
        self.start_time = None
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        print("=" * 80)
        print("Simlingo-QCar2 Integration")
        print("=" * 80)
        
        # Connect to QLabs
        if not self.qcar_interface.connect():
            return False

        # Spawn QCar2 (pass model_wrapper for HLC support)
        if not self.qcar_interface.spawn_qcar(model_wrapper=self.model_wrapper):
            return False

        # Possess camera for visualization
        self.qcar_interface.possess_camera()

        # Setup scene actors using SceneSpawner
        self._setup_scene_actors()
        
        # Load Simlingo model
        print("\nLoading Simlingo model...")
        self.model_wrapper.load_tokenizer()
        self.model_wrapper.load_model()
        print("Model loaded successfully")
 
        print("\nInitialization complete!")
        print("=" * 80)

        return True

    def _setup_scene_actors(self):
        """Setup scene actors using SceneSpawner on separate QLabs connection."""
        if not self.scene_definition:
            print("\nNo scene defined - skipping actor setup")
            return

        print("\n" + "="*80)
        print("SETTING UP SCENE ACTORS")
        print("="*80)

        # Create scene spawner with separate QLabs connection
        self.scene_spawner = SceneSpawner(self.scene_definition)

        # Connect to QLabs (separate connection for actors)
        if not self.scene_spawner.connect():
            print("WARNING: Failed to connect scene actors to QLabs")
            self.scene_spawner = None
            return

        # Spawn all actors defined in the scene
        if not self.scene_spawner.spawn_all_actors():
            print("WARNING: Some actors failed to spawn")

        # Start control threads for dynamic actors
        self.scene_spawner.start_actor_control()

        print("="*80)
        print("Scene setup complete")
        print("="*80)

    def run_step(self) -> bool:
        """
        Execute one control loop iteration.

        Returns:
            True if step successful, False otherwise
        """
        # Get camera image
        image = self.qcar_interface.get_camera_image()

        # Check if image is valid
        if image is None:
            print("WARNING: Skipping step due to invalid camera image")
            # Send stop command to be safe
            self.qcar_interface.set_control(0.0, 0.0)
            return True  # Continue running, just skip this step

        # Process camera image
        camera_images, image_sizes = self.camera_processor.process_image(image)
        camera_intrinsics = self.camera_processor.get_camera_intrinsics_tensor()
        camera_extrinsics = self.camera_processor.get_camera_extrinsics_tensor()
        
        # Get current state
        location, rotation = self.qcar_interface.get_state()
        self.state_estimator.update(location, rotation)
        
        # Get velocity
        velocity = self.state_estimator.get_velocity()
        
        # Get target points and HLC
        current_position = self.state_estimator.get_position()
        current_heading = self.state_estimator.get_heading()
        target_point, next_target_point, hlc = self.route_manager.get_target_point_ego(
            current_position, current_heading
        )

        # Run model inference
        speed_wps, route_wps, language = self.model_wrapper.inference(
            camera_images=camera_images,
            image_sizes=image_sizes,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            vehicle_speed=velocity,
            target_point=target_point,
            next_target_point=next_target_point,
            hlc=hlc  # Pass HLC for command mode
        )

        
        # Convert to numpy
        route_waypoints = route_wps[0].cpu().numpy()
        speed_waypoints = speed_wps[0].cpu().numpy()

        # Update commentary widget if language output available
        # Handle both list and string types
        if isinstance(language, list):
            commentary_text = language[0] if (language and len(language) > 0 and language[0]) else ""
        elif isinstance(language, str):
            commentary_text = language
        else:
            commentary_text = ""

        if commentary_text and commentary_text.strip():
            self.qcar_interface.update_commentary(commentary_text.strip())

        # Update speed display in commentary window
        self.qcar_interface.update_speed(velocity)

        # Update waypoint display in commentary window
        self.qcar_interface.update_waypoints(route_waypoints, speed_waypoints)

        # Compute control using PID
        steer, throttle, brake, desired_speed = self.control_converter.control_pid(
            route_waypoints, velocity, speed_waypoints
        )

        # Stuck detection
        if velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Initial startup boost (first 50 steps or until moving)
        if self.step_count < 50 and velocity < 0.5:
            throttle = max(0.3, throttle)
            brake = False

        # Stuck recovery (after initial startup period)
        elif self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1

        # Convert to QCar2 control using direct velocity control
        forward_velocity, turn_angle = self.control_converter.convert_to_qcar2_control(
            steer, throttle, brake, desired_speed, velocity, self.config.dt
        )
        
        # Send control to QCar2
        success, location, rotation = self.qcar_interface.set_control(
            forward_velocity, turn_angle
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
            'heading_rad': float(rotation[2]),  # Store heading in radians for coordinate transformation
            'speed': float(velocity),
            'steering': float(steer),
            'throttle': float(throttle),
            'brake': bool(brake),
            'current_waypoint_index': int(self.route_manager.current_waypoint_index),
            'target_waypoint': target_world.tolist(),
            'distance_to_target': float(distance_to_target),
            'collision': bool(collision_detected),
            # Model predicted waypoints (in ego frame)
            'predicted_route_waypoints': route_waypoints.tolist() if route_waypoints is not None else None,
            'predicted_speed_waypoints': speed_waypoints.tolist() if speed_waypoints is not None else None
        }
        self.trajectory_log.append(trajectory_entry)
        
        # Print status
        if self.step_count % 10 == 0:
            progress = self.route_manager.get_progress(current_position)
            current_wp_idx = self.route_manager.current_waypoint_index
            total_wps = len(self.route_manager.route_waypoints)

            print(f"Step {self.step_count:4d} | "
                  f"Speed: {velocity:5.2f} m/s | "
                  f"Steer: {steer:6.3f} | "
                  f"Throttle: {throttle:5.3f} | "
                  f"Brake: {brake} | "
                  f"Progress: {progress*100:5.1f}%")
            print(f"  Pos: [{current_position[0]:6.2f}, {current_position[1]:6.2f}] | "
                  f"Heading: {trajectory_entry['heading_deg']:6.1f}° | "
                  f"Target WP[{current_wp_idx}/{total_wps}]: [{target_world[0]:6.2f}, {target_world[1]:6.2f}] | "
                  f"Dist: {distance_to_target:5.2f}m")

            if language is not None:
                print(f"  Language: {language}")
        
        self.step_count += 1
        
        # Check if route complete
        if self.route_manager.is_route_complete(current_position):
            print("\nRoute complete!")
            return False
        
        return True
    
    def save_trajectory_log(self):
        """Save trajectory log to file."""
        os.makedirs("debug_output", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_output/trajectory_log_{timestamp}.json"

        # Prepare metadata
        metadata = {
            'timestamp': timestamp,
            'route_name': self.route_name,
            'total_steps': self.step_count,
            'total_time': self.trajectory_log[-1]['timestamp'] if self.trajectory_log else 0,
            'collision_count': self.collision_count,
            'route_waypoints': self.route_manager.route_waypoints.tolist(),
            'spawn_location': self.config.qcar2_spawn_location,
            'spawn_rotation': self.config.qcar2_spawn_rotation
        }

        # Save to JSON
        data = {
            'metadata': metadata,
            'trajectory': self.trajectory_log
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nTrajectory log saved to {filename}")

        # Also save as "latest" for easy access
        latest_filename = "debug_output/trajectory_log_latest.json"
        with open(latest_filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Trajectory log also saved to {latest_filename}")

    def run(self):
        """Run main control loop."""
        self.running = True
        self.start_time = time.time()

        print("\nStarting control loop at {} Hz...".format(self.config.control_frequency))
        print("Press Ctrl+C to stop")
        print("-" * 80)

        try:
            while self.running:
                loop_start_time = time.time()

                # Execute one step
                if not self.run_step():
                    break

                # Maintain control frequency
                elapsed = time.time() - loop_start_time
                sleep_time = self.config.dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"WARNING: Control loop running slow ({elapsed:.3f}s > {self.config.dt:.3f}s)")

        except KeyboardInterrupt:
            print("\n\nControl loop interrupted by user")
        
        except Exception as e:
            print(f"\n\nERROR: Control loop failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown and cleanup."""
        print("\n" + "=" * 80)
        print("Shutting down...")

        # Save trajectory log
        if self.trajectory_log:
            self.save_trajectory_log()

        # Stop vehicle
        if self.qcar_interface.connected:
            print("Stopping vehicle...")
            self.qcar_interface.set_control(0.0, 0.0)

        # Cleanup scene actors
        if self.scene_spawner:
            print("Cleaning up scene actors...")
            self.scene_spawner.cleanup()

        # Close QLabs connection
        self.qcar_interface.close()

        print("Shutdown complete")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Simlingo-QCar2 Integration with Scene System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scene System:
  Scenes are predefined configurations that include the ego route and all actors.
  Scenes are organized in:
    - scenes/training/  (7 training scenes)
    - scenes/testing/   (3 testing scenes)

  Use --scene to load a complete scene configuration.
  If no scene is specified, only the ego vehicle will be spawned.

Available Training Scenes:
  01_empty_road              Empty road (simple_straight route)
  02_light_traffic           Light traffic (short_route + 1 circular car)
  03_roundabout_north        Roundabout with north pedestrian
  04_kink_navigation         Kink street with south pedestrian
  05_roundabout_exit_east    Traffic circle with roundabout car + east pedestrian
  06_urban_parking           One-way street with parked vehicle
  07_mixed_traffic           Long route with circular car + east pedestrian

Available Testing Scenes:
  08_full_circuit            Full circuit with 2 cars + east pedestrian
  09_roundabout_exit_dual    Roundabout exit with 2 pedestrians
  10_heavy_traffic           Complex route with 2 cars + pedestrian + stop sign

Examples:
  # Empty scene (just ego vehicle, no actors)
  python inference/main.py

  # Load a training scene
  python inference/main.py --scene 01_empty_road
  python inference/main.py --scene light_traffic

  # Load a testing scene
  python inference/main.py --scene full_circuit
  python inference/main.py --scene heavy_traffic

  # Use command-based navigation instead of target points
  python inference/main.py --scene roundabout_north --nav-mode command
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')

    # Scene selection (NEW - replaces individual actor flags)
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene name to load from scenes/ directory (e.g., "empty_road", "01_empty_road", "light_traffic")')

    # Navigation mode
    parser.add_argument('--nav-mode', type=str, default='target_point',
                        choices=['target_point', 'command'],
                        help='Navigational conditioning mode: target_point (uses <TARGET_POINT> tokens) or command (uses HLC text)')

    args = parser.parse_args()

    # Create controller
    controller = SimlingoQCar2Controller(
        config_path=args.config,
        nav_mode=args.nav_mode,
        scene_name=args.scene
    )

    # Initialize
    if not controller.initialize():
        print("\nERROR: Initialization failed")
        return 1

    # Run control loop
    controller.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())

