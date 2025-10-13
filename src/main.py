"""
Main entry point for Simlingo-QCar2 integration.
Run with: python src/main.py
"""

import sys
import os
import time
import numpy as np
import argparse
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import SimlingoQCar2Config
from qcar2_interface import QCar2Interface
from camera_processor import CameraProcessor
from state_estimator import StateEstimator
from route_manager import RouteManager
from simlingo_model import SimlingoModelWrapper
from control_converter import ControlConverter


class SimlingoQCar2Controller:
    """Main controller for Simlingo-QCar2 integration."""
    
    def __init__(self, config_path=None, spawn_obstacles=False):
        """
        Initialize controller.

        Args:
            config_path: Path to custom config file (optional)
            spawn_obstacles: Whether to spawn obstacle vehicles (optional)
        """
        # Load configuration
        self.config = SimlingoQCar2Config()
        self.spawn_obstacles = spawn_obstacles

        # Initialize components
        self.qcar_interface = QCar2Interface(self.config)
        self.camera_processor = CameraProcessor(self.config)
        self.state_estimator = StateEstimator(self.config)
        self.route_manager = RouteManager(self.config)
        self.model_wrapper = SimlingoModelWrapper(self.config)
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
        if not self.qcar_interface.spawn_qcar(spawn_obstacles=self.spawn_obstacles, model_wrapper=self.model_wrapper):
            return False
        
        # Possess camera for visualization
        self.qcar_interface.possess_camera()
        
        # Load Simlingo model
        try:
            print("\nLoading Simlingo model...")
            self.model_wrapper.load_tokenizer()
            self.model_wrapper.load_model()
            print("Model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("\nModel loading failed. Please check:")
            print(f"1. Checkpoint path: {self.config.model_checkpoint_path}")
            print(f"2. Hydra config path: {self.config.hydra_config_path}")
            return False
        
        print("\nInitialization complete!")
        print("=" * 80)
        
        return True
    
    def run_step(self) -> bool:
        """
        Execute one control loop iteration.
        
        Returns:
            True if step successful, False otherwise
        """
        # Get camera image
        image = self.qcar_interface.get_camera_image()
        if image is None:
            print("ERROR: Failed to get camera image")
            return False

        # Save raw camera image (before preprocessing) for debugging
        if self.step_count == 0:
            import cv2
            import os
            os.makedirs("debug_output", exist_ok=True)
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("debug_output/camera_raw_step0.jpg", img_bgr)
            print(f"DEBUG: Saved raw camera image to debug_output/camera_raw_step0.jpg")

        # Process camera image
        camera_images, image_sizes = self.camera_processor.process_image(image)
        camera_intrinsics = self.camera_processor.get_camera_intrinsics_tensor()
        camera_extrinsics = self.camera_processor.get_camera_extrinsics_tensor()
        
        # Get current state
        location, rotation = self.qcar_interface.get_state()
        self.state_estimator.update(location, rotation)
        
        # Get velocity
        velocity = self.state_estimator.get_velocity()
        
        # Get target points
        current_position = self.state_estimator.get_position()
        current_heading = self.state_estimator.get_heading()
        target_point, next_target_point = self.route_manager.get_target_point_ego(
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
            next_target_point=next_target_point
        )

        if speed_wps is None or route_wps is None:
            print("ERROR: Model inference failed")
            return False
        
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

        # NOTE: Bias correction disabled for trajectory logging test
        # The model has a systematic leftward bias (avg Y = 0.34 for straight scenarios)
        # BIAS_SCALE_FACTOR = 0.5  # Reduce Y predictions by 50%
        # route_waypoints[:, 1] *= BIAS_SCALE_FACTOR

        # Debug: Print waypoint information and save camera image
        if self.step_count == 0:
            print(f"\n=== SimLingo Model Output ===")
            print(f"Route waypoints shape: {route_waypoints.shape} (expected: (20, 2))")
            print(f"Speed waypoints shape: {speed_waypoints.shape} (expected: (10, 2))")
            print(f"\nFirst 3 route waypoints:\n{route_waypoints[:3]}")
            print(f"\nFirst 3 speed waypoints:\n{speed_waypoints[:3]}")
            print(f"\nTarget point (ego): {target_point}")
            print(f"Next target point (ego): {next_target_point}")

            # Validate shapes
            if route_waypoints.shape != (20, 2):
                print(f"WARNING: Route waypoints shape mismatch! Expected (20, 2), got {route_waypoints.shape}")
            if speed_waypoints.shape != (10, 2):
                print(f"WARNING: Speed waypoints shape mismatch! Expected (10, 2), got {speed_waypoints.shape}")
            print("=" * 30 + "\n")

            # Save camera image to inspect what the model sees
            import cv2
            import os
            os.makedirs("debug_output", exist_ok=True)
            # camera_images shape: [1, 1, num_patches, 3, 448, 448]
            num_patches = camera_images.shape[2]

            # Denormalize parameters
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Save the first patch
            img_tensor = camera_images[0, 0, 0].cpu().numpy()  # [3, 448, 448]
            img_tensor = np.transpose(img_tensor, (1, 2, 0))  # [448, 448, 3]
            img = img_tensor * std + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("debug_output/camera_patch0_step0.jpg", img_bgr)
            print(f"DEBUG: Saved camera image to debug_output/camera_patch0_step0.jpg")

            # Save the second patch if it exists
            if num_patches > 1:
                img_tensor = camera_images[0, 0, 1].cpu().numpy()  # [3, 448, 448]
                img_tensor = np.transpose(img_tensor, (1, 2, 0))  # [448, 448, 3]
                img = img_tensor * std + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_output/camera_patch1_step0.jpg", img_bgr)
                print(f"DEBUG: Saved camera image to debug_output/camera_patch1_step0.jpg")

        # Compute control using PID
        steer, throttle, brake = self.control_converter.control_pid(
            route_waypoints, velocity, speed_waypoints
        )
        
        # Stuck detection
        if velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0

        # Initial startup boost (first 50 steps or until moving)
        # The Simlingo model is trained on moving vehicles, so we need to get it moving first
        if self.step_count < 50 and velocity < 0.5:
            throttle = max(0.3, throttle)  # Minimum 30% throttle
            brake = False
            if self.step_count == 0:
                print("DEBUG: Applying initial startup boost to get vehicle moving")

        # Stuck recovery (after initial startup period)
        elif self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration

        if self.force_move > 0:
            throttle = max(self.config.creep_throttle, throttle)
            brake = False
            self.force_move -= 1
        
        # Convert to QCar2 control
        forward_velocity, turn_angle = self.control_converter.convert_to_qcar2_control(
            steer, throttle, brake, velocity, self.config.dt
        )
        
        # Send control to QCar2
        success, location, rotation = self.qcar_interface.set_control(
            forward_velocity, turn_angle
        )

        if not success:
            print("ERROR: Failed to send control")
            return False

        # Log trajectory data
        target_world, _ = self.route_manager.get_target_point(current_position)
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

        # Close QLabs connection
        self.qcar_interface.close()
        
        print("Shutdown complete")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Simlingo-QCar2 Integration')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')
    parser.add_argument('--spawn-obstacles', action='store_true',
                        help='Spawn obstacle vehicles along the route')

    args = parser.parse_args()

    # Create controller
    controller = SimlingoQCar2Controller(config_path=args.config, spawn_obstacles=args.spawn_obstacles)

    # Initialize
    if not controller.initialize():
        print("\nERROR: Initialization failed")
        return 1

    # Run control loop
    controller.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())

