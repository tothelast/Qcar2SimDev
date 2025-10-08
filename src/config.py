"""
Configuration file for Simlingo-QCar2 integration.
All parameters are exact replicas from Simlingo to maintain feature parity.
"""

import numpy as np


class SimlingoQCar2Config:
    """Configuration class containing all Simlingo parameters and QCar2 settings."""
    
    def __init__(self):
        # -------------------------------------------------------------------------
        # Simlingo Model Configuration
        # -------------------------------------------------------------------------
        # Path to the Simlingo model checkpoint (DeepSpeed ZeRO checkpoint directory)
        self.model_checkpoint_path = "models/simlingo/checkpoints/epoch=013.ckpt"
        self.encoder_variant = "OpenGVLab/InternVL2-1B"

        # Path to Hydra config (needed for model instantiation)
        self.hydra_config_path = "models/simlingo/.hydra/config.yaml"
        
        # -------------------------------------------------------------------------
        # Camera Configuration (Exact Simlingo Settings)
        # -------------------------------------------------------------------------
        # Target resolution after preprocessing
        self.camera_width = 1024
        self.camera_height = 512
        
        # QCar2 CSI camera native resolution
        self.qcar2_camera_width = 820
        self.qcar2_camera_height = 410
        
        # Camera field of view
        self.camera_fov = 110  # degrees
        
        # Camera position in CARLA coordinates (x, y, z)
        # x: forward, y: right, z: up
        self.camera_position = np.array([-1.5, 0.0, 2.0], dtype=np.float32)
        
        # Camera rotation in radians (roll, pitch, yaw)
        self.camera_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # ImageNet normalization constants
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # -------------------------------------------------------------------------
        # PID Controller Parameters (Exact Simlingo Values)
        # -------------------------------------------------------------------------
        # Turn PID Controller
        self.turn_kp = 3.25
        self.turn_ki = 1.0
        self.turn_kd = 1.0
        self.turn_n = 20  # Buffer size
        
        # Speed PID Controller
        self.speed_kp = 1.75
        self.speed_ki = 1.0
        self.speed_kd = 2.0
        self.speed_n = 20  # Buffer size
        
        # Lateral PID Controller (Advanced)
        self.lateral_pid_kp = 3.118357247806046
        self.lateral_pid_kd = 1.3782508892109167
        self.lateral_pid_ki = 0.6406067986034124
        self.lateral_pid_speed_scale = 0.9755321901954155
        self.lateral_pid_speed_offset = 1.9152884533402488
        self.lateral_pid_default_lookahead = 24
        self.lateral_pid_speed_threshold = 23.150102938235136
        self.lateral_pid_window_size = 6
        
        # Longitudinal PID Controller
        self.longitudinal_pid_proportional_gain = 5.0
        self.longitudinal_pid_derivative_gain = 1.5761818624794222
        self.longitudinal_pid_integral_gain = 0.2941563856687906
        self.longitudinal_pid_max_window_length = 0
        self.longitudinal_pid_speed_error_scaling = 0.0
        self.longitudinal_pid_braking_ratio = 1.0324622059220139
        self.longitudinal_pid_minimum_target_speed = 0.278
        
        # -------------------------------------------------------------------------
        # Control Parameters (Exact Simlingo Values)
        # -------------------------------------------------------------------------
        self.brake_speed = 0.4  # m/s
        self.brake_ratio = 1.1
        self.clip_delta = 1.0
        self.clip_throttle = 1.0
        self.max_throttle = 1.0
        
        # Aim distances for different speeds
        self.aim_distance_fast = 3.0  # meters
        self.aim_distance_slow = 2.25  # meters
        self.aim_distance_very_fast = 7.0  # meters
        self.aim_distance_threshold = 5.5  # m/s
        self.aim_distance_threshold2 = 15.0  # m/s
        
        # -------------------------------------------------------------------------
        # Kinematic Bicycle Model Parameters (Exact Simlingo Values)
        # -------------------------------------------------------------------------
        self.front_wheel_base = -0.090769015
        self.rear_wheel_base = 1.4178275
        self.steering_gain = 0.36848336
        self.brake_acceleration = -4.952399
        self.throttle_acceleration = 0.5633837
        
        # -------------------------------------------------------------------------
        # Timing and Control Loop Configuration
        # -------------------------------------------------------------------------
        self.carla_frame_rate = 1.0 / 20.0  # 50ms per frame
        self.carla_fps = 20  # 20 Hz
        self.control_frequency = 20  # Hz
        self.dt = 1.0 / self.control_frequency  # Time step
        
        # -------------------------------------------------------------------------
        # Waypoint Configuration
        # -------------------------------------------------------------------------
        self.wp_dilation = 1
        self.data_save_freq = 1
        self.interpolation_spacing = 0.1  # meters between interpolated waypoints
        
        # -------------------------------------------------------------------------
        # Stuck Detection and Recovery
        # -------------------------------------------------------------------------
        self.stuck_threshold = 800  # frames
        self.creep_duration = 15  # frames
        self.creep_throttle = 0.4
        self.initial_frames_delay = 5  # frames to wait before starting
        
        # -------------------------------------------------------------------------
        # Route and Navigation Configuration
        # -------------------------------------------------------------------------
        self.eval_route_as = 'target_point'  # "target_point" or "command"
        self.use_cot = True  # Use Chain-of-Thought prompts
        
        # -------------------------------------------------------------------------
        # QCar2 QLabs Configuration
        # -------------------------------------------------------------------------
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
        # - 36 waypoints (downsampled from 893 to ~2.5m spacing)
        # - Starts at Node 13 [2.686, 18.498] heading 90°
        # - Ends at Node 22 [-19.841, 29.760] heading -90°
        #
        # This route tests:
        # - Roundabout navigation (Node 19 → 17)
        # - Multiple direction changes
        # - Long-distance route following
        # - Curved road sections
        self.route_waypoints = [
            [  2.686,  18.498, 0.0],  # Start (spawn location - Node 13)
            [  2.686,  20.998, 0.0],
            [  3.144,  23.456, 0.0],
            [  4.591,  25.593, 0.0],
            [  6.452,  27.274, 0.0],
            [  8.362,  28.993, 0.0],  # Waypoint 5
            [ 10.273,  30.606, 0.0],
            [ 12.686,  31.523, 0.0],
            [ 15.259,  31.440, 0.0],
            [ 17.826,  31.740, 0.0],
            [ 20.094,  32.978, 0.0],  # Waypoint 10
            [ 21.733,  34.976, 0.0],
            [ 22.505,  37.442, 0.0],
            [ 22.298,  40.018, 0.0],
            [ 21.142,  42.329, 0.0],
            [ 19.204,  44.039, 0.0],  # Waypoint 15
            [ 16.768,  44.900, 0.0],
            [ 14.270,  44.974, 0.0],
            [ 11.770,  44.974, 0.0],
            [  9.170,  44.974, 0.0],
            [  6.570,  44.974, 0.0],  # Waypoint 20
            [  3.970,  44.974, 0.0],
            [  1.370,  44.974, 0.0],
            [ -1.200,  44.974, 0.0],
            [ -3.700,  44.974, 0.0],
            [ -6.200,  44.974, 0.0],  # Waypoint 25
            [ -8.800,  44.974, 0.0],
            [-11.328,  44.965, 0.0],
            [-13.872,  44.473, 0.0],
            [-16.167,  43.271, 0.0],
            [-18.019,  41.460, 0.0],  # Waypoint 30
            [-19.273,  39.193, 0.0],
            [-19.821,  36.661, 0.0],
            [-19.841,  34.160, 0.0],
            [-19.841,  31.660, 0.0],
            [-19.841,  29.760, 0.0],  # End (Node 22)
        ]

        # Lookahead distance for target point selection
        # This determines how far ahead the vehicle looks for the target point
        # Set to 5.0m for this 89m route (appropriate for curved roads and roundabout)
        # Original SimLingo uses 10m for highway driving
        # Route length: 89m, so lookahead should be < 44m
        self.target_point_lookahead = 5.0  # meters
        
        # -------------------------------------------------------------------------
        # Visualization and Debugging
        # -------------------------------------------------------------------------
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
        
        # -------------------------------------------------------------------------
        # Special Tokens for Language Model
        # -------------------------------------------------------------------------
        self.special_tokens = [
            '<WAYPOINTS>',
            '<WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS>',
            '<WAYPOINT_LAST>',
            '<ROUTE>',
            '<ROUTE_DIFF>',
            '<TARGET_POINT>',
            '<INSTRUCTION_FOLLOWING>',
            '<SAFETY>',
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

