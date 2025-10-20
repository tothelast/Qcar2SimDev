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
        # Model paths
        self.model_checkpoint_path = "models/simlingo/checkpoints/epoch=013.ckpt"
        self.encoder_variant = "OpenGVLab/InternVL2-1B"
        self.hydra_config_path = "models/simlingo/.hydra/config.yaml"

        # Camera resolution and FOV
        self.camera_width = 1024
        self.camera_height = 512
        self.qcar2_camera_width = 820
        self.qcar2_camera_height = 410
        self.camera_fov = 160  # degrees

        # Camera position (x: forward, y: right, z: up) and rotation (roll, pitch, yaw)
        self.camera_position = np.array([+1.83, 0.0, +1.10], dtype=np.float32)
        self.camera_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ImageNet normalization
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # PID controller (Simlingo values)
        self.turn_kp = 3.25
        self.turn_ki = 1.0
        self.turn_kd = 1.0
        self.turn_n = 20

        # Control parameters (Simlingo values)
        self.clip_throttle = 1.0
        self.max_throttle = 1.0

        # Kinematic bicycle model (Simlingo values)
        self.front_wheel_base = -0.090769015
        self.rear_wheel_base = 1.4178275
        self.steering_gain = 0.36848336
        self.brake_acceleration = -4.952399
        self.throttle_acceleration = 0.5633837

        # Timing: 20 Hz control loop
        self.carla_frame_rate = 1.0 / 20.0
        self.carla_fps = 20
        self.control_frequency = 20
        self.dt = 1.0 / self.control_frequency

        # Waypoint configuration
        self.wp_dilation = 1
        self.data_save_freq = 5  # 4 Hz sampling (20 FPS / 5)
        self.interpolation_spacing = 0.1

        # Stuck detection
        self.stuck_threshold = 800
        self.creep_duration = 15
        self.creep_throttle = 0.4
        self.initial_frames_delay = 5

        # Navigation
        self.eval_route_as = 'target_point'
        self.use_cot = True

        # QCar2 QLabs
        self.qlabs_host = "localhost"
        self.qcar2_actor_number = 0
        self.qcar2_spawn_location = None
        self.qcar2_spawn_rotation = None
        self.qcar2_camera = 3  # CAMERA_CSI_FRONT

        # Routes loaded from config/routes/*.json
        self.route_waypoints = None
        self.target_point_lookahead = 7.5

        # Visualization
        self.enable_visualization = True
        self.save_images = False
        self.save_path = "output"

        # Trajectory visualization (red = actual, green = planned)
        self.enable_trajectory_tracer = True
        self.trajectory_tracer_color = [1.0, 0.0, 0.0]
        self.trajectory_tracer_width = 0.05
        self.trajectory_tracer_update_interval = 5

        self.enable_planned_route_tracer = True
        self.planned_route_tracer_color = [0.0, 1.0, 0.0]
        self.planned_route_tracer_width = 0.05

        # Special tokens for language model
        self.special_tokens = [
            '<WAYPOINTS>', '<WAYPOINTS_DIFF>', '<ORG_WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS>', '<WAYPOINT_LAST>', '<ROUTE>',
            '<ROUTE_DIFF>', '<TARGET_POINT>',
        ]
        
    def get_camera_intrinsics(self, width=None, height=None, fov=None):
        """Generate 3x3 camera intrinsics matrix from FOV."""
        width = width or self.camera_width
        height = height or self.camera_height
        fov = fov or self.camera_fov

        # Focal length: f = width / (2 * tan(fov/2))
        f = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        cx, cy = width / 2.0, height / 2.0

        return np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    def get_camera_extrinsics(self):
        """Generate 4x4 camera extrinsics matrix (identity rotation + translation)."""
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        extrinsics[3, 3] = 1.0
        extrinsics[:3, :3] = np.eye(3)
        extrinsics[:3, 3] = self.camera_position
        return extrinsics
    
    def get_prompt_template(self, speed, use_cot=None):
        """Generate prompt template for Simlingo model."""
        use_cot = use_cot if use_cot is not None else self.use_cot
        base = f"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>."
        return f"{base} What should the ego do next?" if use_cot else f"{base} Predict the waypoints."

    def load_route(self, route_name):
        """Load route from config/routes/*.json."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        route_path = os.path.join(project_root, "config", "routes", f"{route_name}.json")

        if not os.path.exists(route_path):
            print(f"ERROR: Route file not found: {route_path}")
            return False

        try:
            with open(route_path, 'r') as f:
                route_data = json.load(f)

            self.route_waypoints = route_data['waypoints']
            self.qcar2_spawn_location = route_data['spawn_location']
            self.qcar2_spawn_rotation = route_data['spawn_rotation']

            print(f"Loaded route: {route_data['name']} ({route_data['num_waypoints']} waypoints, {route_data['total_distance']:.1f}m)")
            return True
        except Exception as e:
            print(f"ERROR loading route {route_name}: {e}")
            return False

