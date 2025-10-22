"""Control conversion with PID controllers for QCar2."""

import numpy as np
from collections import deque
from scipy.interpolate import PchipInterpolator
from typing import Tuple


class LateralPIDController:
    """Lateral PID controller for steering."""

    def __init__(self, config):
        """Initialize lateral PID controller."""
        self.k_p = config.turn_kp
        self.k_i = config.turn_ki
        self.k_d = config.turn_kd
        self.n = config.turn_n
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)

        # Speed-dependent aim distances
        self.aim_distance_slow = 2.25
        self.aim_distance_fast = 3.0
        self.aim_distance_very_fast = 7.0
        self.aim_distance_threshold = 5.5
        self.aim_distance_threshold2 = 15.0
    
    def step(self, route_np: np.ndarray, current_speed: float) -> float:
        """Compute steering control."""
        # Speed-based aim distance
        speed_kmh = current_speed * 3.6
        if speed_kmh < self.aim_distance_threshold:
            aim_distance = self.aim_distance_slow
        elif speed_kmh < self.aim_distance_threshold2:
            aim_distance = self.aim_distance_fast
        else:
            aim_distance = self.aim_distance_very_fast

        n_lookahead = int(min(aim_distance * 10, len(route_np) - 1))
        desired_heading_vec = route_np[n_lookahead]

        # Calculate heading error
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        self._window.append(heading_error)

        # PID terms
        integral = sum(self._window) / len(self._window) if len(self._window) >= 2 else 0.0
        derivative = self._window[-1] - self._window[-2] if len(self._window) >= 2 else 0.0

        steering = self.k_p * heading_error + self.k_i * integral + self.k_d * derivative
        return np.clip(steering, -1.0, 1.0)

    def reset(self):
        """Reset controller state."""
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)


class LongitudinalLinearRegressionController:
    """Longitudinal controller using linear regression."""

    def __init__(self, config):
        """Initialize longitudinal controller."""
        self.minimum_target_speed = 0.278
        self.params = np.array([
            1.1990342347353184, -0.8057602384167799, 1.710818710950062,
            0.921890257450335, 1.556497522998393, -0.7013479734904027,
            1.031266635497984
        ])
        self.max_acceleration = 1.89
        self.max_deceleration = -4.82

    def get_throttle_and_brake(self, target_speed: float, current_speed: float) -> Tuple[float, bool]:
        """Get throttle and brake values using linear regression."""
        if target_speed < 1e-5:
            return 0.0, True

        target_speed = max(self.minimum_target_speed, target_speed)
        current_speed_kmh = current_speed * 3.6
        target_speed_kmh = target_speed * 3.6
        speed_error = target_speed_kmh - current_speed_kmh

        if speed_error > self.max_acceleration:
            return 1.0, False

        if current_speed_kmh / target_speed_kmh > self.params[-1]:
            return 0.0, True

        # Normalize values (scaling is leftover from optimization)
        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed_norm = current_speed_kmh / 100.0

        # Construct feature vector
        features = np.array([
            current_speed_norm,
            current_speed_norm**2,
            100 * speed_error_cl,
            speed_error_cl**2,
            current_speed_norm * speed_error_cl,
            current_speed_norm**2 * speed_error_cl
        ])

        # Linear regression: throttle = features @ coefficients
        throttle = np.clip(features @ self.params[:-1], 0.0, 1.0)

        return throttle, False


class ControlConverter:
    """Converts Simlingo model outputs to QCar2 control commands."""

    def __init__(self, config):
        """
        Initialize control converter.

        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config

        # Initialize controllers
        self.speed_controller = LongitudinalLinearRegressionController(config)
        self.turn_controller = LateralPIDController(config)
        
    def control_pid(
        self,
        route_waypoints: np.ndarray,
        velocity: float,
        speed_waypoints: np.ndarray
    ) -> Tuple[float, float, bool, float]:
        """
        Compute control from waypoints using PID (exact Simlingo implementation).

        Args:
            route_waypoints: Route waypoints [F, 2]
            velocity: Current velocity in m/s
            speed_waypoints: Speed waypoints [F, 2]

        Returns:
            Tuple of (steer, throttle, brake, desired_speed)
        """
        # Calculate desired speed from speed waypoints
        # Reference: simlingo/team_code/agent_simlingo.py control_pid() method
        # Uses waypoints[0] to waypoints[2] (first 0.5s of predictions)

        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2

        # Indices: [half_second - 2, one_second - 2] = [0, 2]
        idx1 = half_second - 2  # Index 0
        idx2 = one_second - 2   # Index 2

        if len(speed_waypoints) > idx2:
            desired_speed = np.linalg.norm(
                speed_waypoints[idx2] - speed_waypoints[idx1]
            ) * 2.0
        elif len(speed_waypoints) >= 2:
            # Fallback: use available waypoints
            desired_speed = np.linalg.norm(
                speed_waypoints[-1] - speed_waypoints[0]
            ) * 2.0 / len(speed_waypoints)
        else:
            # Not enough waypoints
            desired_speed = 0.0

        # Get throttle and brake from linear regression controller
        throttle, brake = self.speed_controller.get_throttle_and_brake(
            target_speed=desired_speed,
            current_speed=velocity
        )

        # Clip throttle to configured maximum
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

        # Steering calculation
        route_interp = self.interpolate_waypoints(route_waypoints)
        steer = self.turn_controller.step(route_interp, velocity)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer, throttle, brake, desired_speed
    
    def interpolate_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Interpolate waypoints to fixed spacing (exact Simlingo implementation).

        Args:
            waypoints: Waypoints array (N, 2)

        Returns:
            Interpolated waypoints with 0.1m spacing
        """
        waypoints = waypoints.copy()

        # Filter out waypoints too close to origin (< 0.05m)
        # These are essentially noise and cause sharp initial turns
        distances_from_origin = np.linalg.norm(waypoints, axis=1)
        valid_mask = distances_from_origin >= 0.05

        # Keep at least one waypoint
        if not np.any(valid_mask):
            valid_mask[0] = True

        waypoints = waypoints[valid_mask]

        # Add origin point
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        
        # Calculate cumulative distances
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]
        
        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4  # Prevent non-strictly increasing
        
        # Interpolate
        interp = PchipInterpolator(dists, waypoints, axis=0)
        
        x = np.arange(self.config.interpolation_spacing, dists[-1], self.config.interpolation_spacing)
        
        interp_points = interp(x)
        
        # Handle edge case: no points at 0.1m spacing
        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]
        
        return interp_points
    
    def convert_to_qcar2_control(
        self,
        desired_speed: float,
        steer: float,
        current_speed: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Convert Simlingo control to QCar2 control.

        Args:
            desired_speed: Target speed from waypoints in m/s
            steer: Steering value [-1, 1]
            current_speed: Current speed in m/s
            dt: Time step in seconds

        Returns:
            Tuple of (forward_velocity, turn_angle)
        """
        # Apply smooth acceleration limits for QCar2 physical constraints
        speed_diff = desired_speed - current_speed
        max_speed_change = self.config.qcar2_max_acceleration * dt if speed_diff > 0 else self.config.qcar2_max_deceleration * dt

        # Limit speed change to physical constraints
        speed_diff = np.clip(speed_diff, -max_speed_change, max_speed_change)
        forward_velocity = current_speed + speed_diff

        # Clip to QCar2 max speed
        forward_velocity = np.clip(forward_velocity, 0.0, self.config.qcar2_max_speed)

        # Convert steering to turn angle
        # NOTE: QCar2 convention is opposite to CARLA/Simlingo:
        # - CARLA/Simlingo: positive steering = left turn
        # - QCar2: positive turn_angle = right turn
        # Map normalized steering [-1, 1] to QCar2's physical steering range
        turn_angle = -steer * self.config.qcar2_max_steering

        return forward_velocity, turn_angle

    def reset(self):
        """Reset controller state."""
        self.turn_controller.reset()

