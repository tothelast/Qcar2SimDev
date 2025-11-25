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
        # self.model_checkpoint_path = "new_chekpoint"
        self.encoder_variant = "OpenGVLab/InternVL2-1B"
        self.hydra_config_path = "models/simlingo/.hydra/config.yaml"

        # CARLA training camera parameters 
        self.camera_width = 820
        self.camera_height = 410  # QLabs CSI native resolution
        self.camera_fov = 160     # degrees - Match QLabs QCar2 CSI front camera FOV
        self.camera_position = np.array([0.183, 0.0, 0.110], dtype=np.float32)  # QLabs QCar2 CSI front camera       
        self.resize_input_to_training_resolution = True  # Resize inference frames to match training resolution

        # ImageNet normalization
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # PID controller parameters (revert to Carla agent tuning)
        self.turn_kp = 3.0
        self.turn_ki = 1.0
        self.turn_kd = 1.0  # Reduced from 2.5 to prevent jitter at 4Hz
        self.turn_n = 3     # Reduced from 6 to match 3Hz (approx 1s window)

        # Braking Logic
        self.brake_speed = 0.3  # desired speed below which brake is triggered (Reduced from 0.4 to fix start delay)
        self.brake_ratio = 1.2  # ratio of speed/desired_speed at which brake is triggered

        # Timing
        self.carla_fps = 3  # Control loop frequency (Reduced to 3Hz to match ~280ms inference latency)
        self.dt = 1.0 / self.carla_fps  # 0.25s timestep
        self.data_save_freq = 2  # Data saving cadence during recording (every N control ticks)
        self.inference_stride = 1  # Inference cadence during control (every N control ticks) - Set to 1 for max responsiveness

        # Waypoint configuration
        self.interpolation_spacing = 0.1

        # QCar2 physical constraints
        self.qcar2_max_speed = 4.0  # m/s
        self.qcar2_max_acceleration = 0.8  # m/s^2 (Reduced from 2.0 to match 4Hz dt)
        self.qcar2_max_deceleration = 1.6  # m/s^2 (Reduced from 4.0 to match 4Hz dt)
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
