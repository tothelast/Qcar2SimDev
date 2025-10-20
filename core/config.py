"""
Configuration file for Simlingo-QCar2 integration.
All parameters are exact replicas from Simlingo to maintain feature parity.
"""

import numpy as np
import json
import os


class SimlingoQCar2Config:
    """Configuration class containing all Simlingo parameters and QCar2 settings."""
    
    def __init__(self):

        self.model_checkpoint_path = "models/simlingo/checkpoints/epoch=013.ckpt"
        self.encoder_variant = "OpenGVLab/InternVL2-1B"

        # Path to Hydra config (needed for model instantiation)
        self.hydra_config_path = "models/simlingo/.hydra/config.yaml"
        
        # Target resolution after preprocessing
        self.camera_width = 1024
        self.camera_height = 512
        
        # QCar2 CSI camera native resolution
        self.qcar2_camera_width = 820
        self.qcar2_camera_height = 410
        
        # Camera field of view
        # QCar2 CSI front camera specification: 160° FOV
        self.camera_fov = 160  # degrees

        # Camera position for QCar2 front camera (x, y, z)
        # x: forward, y: right, z: up
        # QCar2 front camera: [+1.83m forward, 0.0, +1.10m up] relative to car center
        # Reference: docs/DATA_COLLECTION.md
        self.camera_position = np.array([+1.83, 0.0, +1.10], dtype=np.float32)

        # Camera rotation in radians (roll, pitch, yaw)
        self.camera_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # ImageNet normalization constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # PID Controller Parameters (Exact Simlingo Values)
        self.turn_kp = 3.25
        self.turn_ki = 1.0
        self.turn_kd = 1.0
        self.turn_n = 20  # Buffer size

        # Control Parameters (Exact Simlingo Values)
        self.clip_throttle = 1.0
        self.max_throttle = 1.0
        
        # Kinematic Bicycle Model Parameters (Exact Simlingo Values)
        self.front_wheel_base = -0.090769015
        self.rear_wheel_base = 1.4178275
        self.steering_gain = 0.36848336
        self.brake_acceleration = -4.952399
        self.throttle_acceleration = 0.5633837
        
        # Timing and Control Loop Configuration

        self.carla_frame_rate = 1.0 / 20.0  # 50ms per frame
        self.carla_fps = 20  # 20 Hz
        self.control_frequency = 20  # Hz
        self.dt = 1.0 / self.control_frequency  # Time step
        
        # Waypoint Configuration
        self.wp_dilation = 1
        # CRITICAL: data_save_freq = 5 achieves 4 Hz sampling (20 FPS / 5 = 4 Hz)
        # This matches SimLingo CARLA training data (0.25s intervals between samples)
        self.data_save_freq = 5  # Save every 5th frame (was 1)
        self.interpolation_spacing = 0.1  # meters between interpolated waypoints
        
        # Stuck Detection and Recovery
        self.stuck_threshold = 800  # frames
        self.creep_duration = 15  # frames
        self.creep_throttle = 0.4
        self.initial_frames_delay = 5  # frames to wait before starting
        
        # Route and Navigation Configuration
        self.eval_route_as = 'target_point'  # "target_point" or "command"
        self.use_cot = True  # Use Chain-of-Thought prompts
        
        # QCar2 QLabs Configuration
        self.qlabs_host = "localhost"
        self.qcar2_actor_number = 0

        # QCar2 spawn location and rotation
        # These will be set when a route is loaded via load_route()
        # If not loaded, the system will fail with a clear error message
        self.qcar2_spawn_location = None
        self.qcar2_spawn_rotation = None

        # QCar2 camera selection
        self.qcar2_camera = 3  # CAMERA_CSI_FRONT

        # -------------------------------------------------------------------------
        # Route Waypoints (QLabs Cityscape Lite Global Coordinates)
        # -------------------------------------------------------------------------
        # Route Waypoints
        # -------------------------------------------------------------------------
        # Routes are now loaded from JSON files in the config/routes/ directory
        # Use config.load_route(route_name) to load a specific route
        # Each route contains:
        # - waypoints: List of [x, y, z] coordinates with ~1.0m spacing
        # - spawn_location: [x, y, z] starting position
        # - spawn_rotation: [roll, pitch, yaw] in radians
        # - metadata: route name, distance, number of waypoints
        #
        # If not loaded, the system will fail with a clear error message
        self.route_waypoints = None

        # Lookahead distance for target point selection
        # This determines how far ahead the vehicle looks for the target point
        # Set to 7.5m to match SimLingo training (was 5.0m)
        self.target_point_lookahead = 7.5 # meters
        
        # Visualization and Debugging
        self.enable_visualization = True
        self.save_images = False
        self.save_path = "output"

        # QLabs trajectory tracer configuration
        self.enable_trajectory_tracer = True  # Enable real-time trajectory visualization in QLabs
        self.trajectory_tracer_color = [1.0, 0.0, 0.0]  # RGB color (red for actual trajectory)
        self.trajectory_tracer_width = 0.05  # Line width in meters
        self.trajectory_tracer_update_interval = 5  # Update every N steps (reduce overhead)

        # Planned route visualization
        self.enable_planned_route_tracer = True  # Show planned route as green line
        self.planned_route_tracer_color = [0.0, 1.0, 0.0]  # RGB color (green for planned route)
        self.planned_route_tracer_width = 0.05  # Line width in meters

        # Special Tokens for Language Model
        # IMPORTANT: Only tokens that should be added to tokenizer vocabulary
        # <SAFETY> and <INSTRUCTION_FOLLOWING> are NOT special tokens - they are regular text!
        self.special_tokens = [
            '<WAYPOINTS>',
            '<WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS>',
            '<WAYPOINT_LAST>',
            '<ROUTE>',
            '<ROUTE_DIFF>',
            '<TARGET_POINT>',
        ]
        
    def get_camera_intrinsics(self, width=None, height=None, fov=None):
        """
        Generate camera intrinsics matrix.
        
        Args:
            width: Image width (default: self.camera_width)
            height: Image height (default: self.camera_height)
            fov: Field of view in degrees (default: self.camera_fov)
            
        Returns:
            3x3 camera intrinsics matrix
        """
        if width is None:
            width = self.camera_width
        if height is None:
            height = self.camera_height
        if fov is None:
            fov = self.camera_fov
            
        # Calculate focal length from FOV
        # f = width / (2 * tan(fov/2))
        f = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        # Intrinsics matrix
        intrinsics = np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        return intrinsics
    
    def get_camera_extrinsics(self):
        """
        Generate camera extrinsics matrix (4x4 transformation matrix).
        Matches the original Simlingo implementation exactly.

        Returns:
            4x4 camera extrinsics matrix
        """
        # Original Simlingo uses identity rotation and just sets translation
        # Camera position: [-1.5, 0.0, 2.0] (x, y, z)
        # Camera rotation: [0.0, 0.0, 0.0] (roll, pitch, yaw)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        extrinsics[3, 3] = 1.0
        extrinsics[:3, :3] = np.eye(3)
        extrinsics[:3, 3] = self.camera_position  # [-1.5, 0.0, 2.0]

        return extrinsics
    
    def get_prompt_template(self, speed, use_cot=None):
        """
        Generate prompt template for Simlingo model.
        
        Args:
            speed: Current vehicle speed in m/s
            use_cot: Use Chain-of-Thought (default: self.use_cot)
            
        Returns:
            Prompt string with placeholders
        """
        if use_cot is None:
            use_cot = self.use_cot
            
        if use_cot:
            prompt = f"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
        else:
            prompt = f"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints."

        return prompt

    def load_route(self, route_name):
        """
        Load a route from the config/routes/ directory.

        Args:
            route_name: Name of the route file (without .json extension)

        Returns:
            True if route loaded successfully, False otherwise
        """
        # Get the project root directory (parent of core/)
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_dir)
        route_path = os.path.join(project_root, "config", "routes", f"{route_name}.json")

        if not os.path.exists(route_path):
            print(f"ERROR: Route file not found: {route_path}")
            return False

        try:
            with open(route_path, 'r') as f:
                route_data = json.load(f)

            # Update route waypoints
            self.route_waypoints = route_data['waypoints']

            # Update spawn location and rotation
            self.qcar2_spawn_location = route_data['spawn_location']
            self.qcar2_spawn_rotation = route_data['spawn_rotation']

            print(f"Loaded route: {route_data['name']}")
            print(f"  Waypoints: {route_data['num_waypoints']}")
            print(f"  Distance: {route_data['total_distance']:.1f}m")
            print(f"  Spawn: {self.qcar2_spawn_location}")

            return True

        except Exception as e:
            print(f"ERROR loading route {route_name}: {e}")
            return False

