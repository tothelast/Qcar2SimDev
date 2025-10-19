"""
Configuration file for Simlingo-QCar2 integration.
All parameters are exact replicas from Simlingo to maintain feature parity.
"""

import numpy as np


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
        self.camera_fov = 160

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
        self.data_save_freq = 1
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

        # QCar2 spawn location (QLabs Cityscape Lite coordinates)
        # Spawn at Node 13 (roundabout route start)
        # Coordinates from SDCSRoadMap with proper scaling:
        #   SDCSRoadMap: [0.26862, 1.84981] → QLabs: [2.686, 18.498] = × 10
        # Heading: 90° (facing north/east)
        self.qcar2_spawn_location = [2.686, 18.498, 0.005]  # [x, y, z] - Node 13
        self.qcar2_spawn_rotation = [0.0, 0.0, 1.5708]  # [roll, pitch, yaw] in radians (90° = 1.5708 rad)

        # QCar2 camera selection
        self.qcar2_camera = 3  # CAMERA_CSI_FRONT

        # -------------------------------------------------------------------------
        # Route Waypoints (QLabs Cityscape Lite Global Coordinates)
        # -------------------------------------------------------------------------
        # Route: Node 13 → 19 → 17 → 20 → 22 (Roundabout Navigation)
        #
        # Route description:
        # - Generated using SDCSRoadMap.generate_path([13, 19, 17, 20, 22])
        # - Follows actual road network in QLabs Cityscape
        # - Coordinates scaled: QLabs_X = SDCSRoadMap_X × 10, QLabs_Y = SDCSRoadMap_Y × 10
        # - Total length: ~89 meters
        # - 85 waypoints (downsampled from 893 to ~1.0m spacing)
        # - Starts at Node 13 [2.686, 18.498] heading 90°
        # - Ends at Node 22 [-19.841, 29.760] heading -90°
        #
        # Spacing: ~1.0m to match SimLingo training
        # - SimLingo uses CARLA GlobalRoutePlanner with hop_resolution=1.0
        # - This creates waypoints ~1m apart for target point selection
        # - Target points fed to model are consecutive waypoints ~1m apart
        #
        # This route tests:
        # - Roundabout navigation (Node 19 → 17)
        # - Multiple direction changes
        # - Long-distance route following
        # - Curved road sections
        self.route_waypoints = [
            [  2.686,  18.498, 0.0],  # Start (spawn location)
            [  2.686,  19.498, 0.0],
            [  2.686,  20.598, 0.0],
            [  2.706,  21.614, 0.0],
            [  2.891,  22.697, 0.0],
            [  3.263,  23.731, 0.0],  # Waypoint 5
            [  3.812,  24.682, 0.0],
            [  4.519,  25.523, 0.0],
            [  5.263,  26.203, 0.0],
            [  6.007,  26.872, 0.0],
            [  6.824,  27.608, 0.0],  # Waypoint 10
            [  7.567,  28.278, 0.0],
            [  8.362,  28.993, 0.0],
            [  9.179,  29.729, 0.0],
            [  9.944,  30.379, 0.0],
            [ 10.882,  30.951, 0.0],  # Waypoint 15
            [ 11.906,  31.350, 0.0],
            [ 12.983,  31.562, 0.0],
            [ 14.082,  31.581, 0.0],
            [ 15.159,  31.448, 0.0],
            [ 16.258,  31.438, 0.0],  # Waypoint 20
            [ 17.344,  31.606, 0.0],
            [ 18.389,  31.947, 0.0],
            [ 19.364,  32.452, 0.0],
            [ 20.246,  33.108, 0.0],
            [ 21.009,  33.898, 0.0],  # Waypoint 25
            [ 21.635,  34.802, 0.0],
            [ 22.107,  35.794, 0.0],
            [ 22.412,  36.850, 0.0],
            [ 22.542,  37.941, 0.0],
            [ 22.495,  39.038, 0.0],  # Waypoint 30
            [ 22.270,  40.114, 0.0],
            [ 21.875,  41.139, 0.0],
            [ 21.319,  42.087, 0.0],
            [ 20.617,  42.933, 0.0],
            [ 19.788,  43.654, 0.0],  # Waypoint 35
            [ 18.854,  44.231, 0.0],
            [ 17.838,  44.650, 0.0],
            [ 16.768,  44.900, 0.0],
            [ 15.670,  44.974, 0.0],
            [ 14.670,  44.974, 0.0],  # Waypoint 40
            [ 13.670,  44.974, 0.0],
            [ 12.670,  44.974, 0.0],
            [ 11.570,  44.974, 0.0],
            [ 10.470,  44.974, 0.0],
            [  9.370,  44.974, 0.0],  # Waypoint 45
            [  8.270,  44.974, 0.0],
            [  7.170,  44.974, 0.0],
            [  6.070,  44.974, 0.0],
            [  4.970,  44.974, 0.0],
            [  3.870,  44.974, 0.0],  # Waypoint 50
            [  2.770,  44.974, 0.0],
            [  1.670,  44.974, 0.0],
            [  0.570,  44.974, 0.0],
            [ -0.500,  44.974, 0.0],
            [ -1.500,  44.974, 0.0],  # Waypoint 55
            [ -2.500,  44.974, 0.0],
            [ -3.500,  44.974, 0.0],
            [ -4.500,  44.974, 0.0],
            [ -5.600,  44.974, 0.0],
            [ -6.600,  44.974, 0.0],  # Waypoint 60
            [ -7.600,  44.974, 0.0],
            [ -8.600,  44.974, 0.0],
            [ -9.600,  44.974, 0.0],
            [-10.600,  44.974, 0.0],
            [-11.627,  44.946, 0.0],  # Waypoint 65
            [-12.716,  44.792, 0.0],
            [-13.777,  44.506, 0.0],
            [-14.795,  44.091, 0.0],
            [-15.754,  43.554, 0.0],
            [-16.640,  42.903, 0.0],  # Waypoint 70
            [-17.439,  42.148, 0.0],
            [-18.139,  41.300, 0.0],
            [-18.729,  40.372, 0.0],
            [-19.200,  39.379, 0.0],
            [-19.546,  38.336, 0.0],  # Waypoint 75
            [-19.760,  37.258, 0.0],
            [-19.841,  36.161, 0.0],
            [-19.841,  35.160, 0.0],
            [-19.841,  34.160, 0.0],
            [-19.841,  33.160, 0.0],  # Waypoint 80
            [-19.841,  32.160, 0.0],
            [-19.841,  31.160, 0.0],
            [-19.841,  30.160, 0.0],
            [-19.841,  29.760, 0.0],  # End
        ]

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

