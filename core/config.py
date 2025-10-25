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

        # CARLA training camera parameters (exact match to training data)
        self.camera_width = 1024
        self.camera_height = 512
        self.camera_fov = 110  # degrees - Match CARLA training FOV
        self.camera_position = np.array([-1.5, 0.0, 2.0], dtype=np.float32)  # CARLA camera position
        self.camera_bottom_crop_ratio = 0.0  # Fraction of image height to remove from bottom during preprocessing

        # ImageNet normalization
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # PID controller parameters (exact Simlingo values from config_simlingo.py)
        self.turn_kp = 1.25
        self.turn_ki = 0.75
        self.turn_kd = 0.3
        self.turn_n = 20

        # Timing: Match CARLA exactly - 20 Hz control, 4 Hz model inference
        self.carla_fps = 20  # Control loop frequency (matches CARLA simulator FPS)
        self.dt = 1.0 / self.carla_fps  # 0.05s timestep
        self.data_save_freq = 5  # Model inference every 5 iterations = 4 Hz (matches CARLA training data)

        # Waypoint configuration
        self.interpolation_spacing = 0.1

        # Stuck detection
        self.stuck_threshold = 800
        self.creep_duration = 15
        self.creep_throttle = 0.4

        # QCar2 physical constraints
        self.qcar2_max_speed = 4.0  # m/s
        self.qcar2_max_acceleration = 2.0  # m/s^2
        self.qcar2_max_deceleration = 4.0  # m/s^2
        self.qcar2_max_steering = np.pi / 9  # radians (~20 degrees)

        # QCar2 QLabs
        self.qlabs_host = "localhost"
        self.qcar2_actor_number = 0
        self.qcar2_spawn_location = None
        self.qcar2_spawn_rotation = None
        self.qcar2_camera = 3  # CAMERA_CSI_FRONT

        # Routes loaded from config/routes/*.json
        self.route_waypoints = None
        self.target_point_lookahead = 7.5

        # Trajectory visualization (red = actual, green = planned)
        self.enable_trajectory_tracer = True
        self.trajectory_tracer_color = [1.0, 0.0, 0.0]
        self.trajectory_tracer_width = 0.05
        self.trajectory_tracer_update_interval = 5

        self.enable_planned_route_tracer = True
        self.planned_route_tracer_color = [0.0, 1.0, 0.0]
        self.planned_route_tracer_width = 0.05
        
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
