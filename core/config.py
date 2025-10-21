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

        # CARLA training camera parameters (legacy - required by DrivingInput interface but not used by model)
        self.camera_width = 1024
        self.camera_height = 512
        self.camera_fov = 110  # degrees (CARLA training FOV)
        self.camera_position = np.array([-1.5, 0.0, 2.0], dtype=np.float32)  # CARLA camera position

        # ImageNet normalization
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # PID controller parameters (from PDM-Lite expert)
        self.turn_kp = 3.25
        self.turn_ki = 1.0
        self.turn_kd = 1.0
        self.turn_n = 20

        # Control parameters (must match teleop_controller.py for consistent fine-tuning)
        self.clip_throttle = 1.0
        self.steering_gain = np.pi / 9  # ~20° turn angle (matches teleop max_steering_angle)

        # Timing: 10 Hz control loop (matches teleop dt = 0.1)
        self.carla_fps = 20
        self.control_frequency = 10
        self.dt = 1.0 / self.control_frequency

        # Waypoint configuration (must match training data)
        self.wp_dilation = 1
        self.data_save_freq = 5  # Waypoint spacing: 0.25s (20 FPS / 5)
        self.interpolation_spacing = 0.1

        # QCar2 physical limits and scaling (must match teleop_controller.py)
        # speed_scale is used bidirectionally:
        # - Input: qcar2_speed / speed_scale -> model sees CARLA-range speeds (0-10 m/s)
        # - Output: model_speed * speed_scale -> QCar2-range speeds (0-4 m/s)
        self.speed_scale = 0.4  # Maps CARLA speeds (0-10 m/s) to QCar2 range (0-4 m/s)
        self.qcar2_max_speed = 4.0  # m/s (matches teleop max_forward_velocity)
        self.qcar2_max_acceleration = 1.0  # m/s² (matches teleop acceleration)
        self.qcar2_max_deceleration = 2.0  # m/s² (matches teleop deceleration)

        # Stuck detection
        self.stuck_threshold = 800
        self.creep_duration = 15
        self.creep_throttle = 0.4

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

        # Special token for language model
        self.special_tokens = ['<TARGET_POINT>']
        
    def get_camera_intrinsics(self, width=None, height=None, fov=None):
        """Generate 3x3 camera intrinsics matrix (legacy - required by DrivingInput but not used by model)."""
        width = width or self.camera_width
        height = height or self.camera_height
        fov = fov or self.camera_fov
        f = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        cx, cy = width / 2.0, height / 2.0
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def get_camera_extrinsics(self):
        """Generate 4x4 camera extrinsics matrix (legacy - required by DrivingInput but not used by model)."""
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

