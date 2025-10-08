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
        # Spawn location adjusted to be near the route start
        # The route starts at [0, 1.3] but we spawn slightly behind at [0, -1.3]
        # Keep original heading of 90° (facing +Y direction)
        self.qcar2_spawn_location = [0.0, -1.300, 0.005]  # [x, y, z]
        self.qcar2_spawn_rotation = [0.0, 0.0, np.pi/2]  # [roll, pitch, yaw] in radians (90° = facing +Y)

        # QCar2 camera selection
        self.qcar2_camera = 3  # CAMERA_CSI_FRONT

        # -------------------------------------------------------------------------
        # Route Waypoints (QLabs Cityscape Lite Global Coordinates)
        # -------------------------------------------------------------------------
        # Route: Simple straight-line test route for visual verification
        #
        # This is a SIMPLE TEST ROUTE to verify that waypoints are correctly
        # placed on the roads in QLabs Cityscape Lite.
        #
        # Route description:
        # - Starts at spawn location [0, -1.3] heading 90° (facing +Y/North)
        # - Goes straight north along X=0 for ~41 meters
        # - Waypoints spaced every 2 meters for smooth control
        # - Total length: 41.3 meters
        # - 22 waypoints total
        #
        # This route should be easy to verify visually in the simulation:
        # - The car should drive straight forward from spawn
        # - No turns, just a straight line
        # - If the car veers left/right, it indicates model bias or control issues
        self.route_waypoints = [
            [  0.000,  -1.300, 0.0],  # Spawn location
            [  0.000,   0.700, 0.0],
            [  0.000,   2.700, 0.0],
            [  0.000,   4.700, 0.0],
            [  0.000,   6.700, 0.0],
            [  0.000,   8.700, 0.0],
            [  0.000,  10.700, 0.0],
            [  0.000,  12.700, 0.0],
            [  0.000,  14.700, 0.0],
            [  0.000,  16.700, 0.0],
            [  0.000,  18.700, 0.0],
            [  0.000,  20.700, 0.0],
            [  0.000,  22.700, 0.0],
            [  0.000,  24.700, 0.0],
            [  0.000,  26.700, 0.0],
            [  0.000,  28.700, 0.0],
            [  0.000,  30.700, 0.0],
            [  0.000,  32.700, 0.0],
            [  0.000,  34.700, 0.0],
            [  0.000,  36.700, 0.0],
            [  0.000,  38.700, 0.0],
            [  0.000,  40.000, 0.0],  # End of route
        ]

        # Lookahead distance for target point selection
        # This determines how far ahead the vehicle looks for the target point
        # 10 meters works well with the waypoint spacing of 5-8 meters
        self.target_point_lookahead = 10.0  # meters
        
        # -------------------------------------------------------------------------
        # Visualization and Debugging
        # -------------------------------------------------------------------------
        self.enable_visualization = True
        self.save_images = False
        self.save_path = "output"
        
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

