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
        # Speed-dependent aim distances - MATCH PRETRAINED SIMLINGO
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

        # Legacy longitudinal controller retained for reference (unused in direct-speed mode)
        self.speed_controller = LongitudinalLinearRegressionController(config)
        self.turn_controller = LateralPIDController(config)
        
        # Minimal state: just track previous brake for velocity hold
        self.prev_brake = False
        self.smoothed_desired_speed = 0.0
        
    def control_pid(
        self,
        route_waypoints: np.ndarray,
        velocity: float,
        speed_waypoints: np.ndarray
    ) -> Tuple[float, float, bool, float]:
        """
        Compute control from waypoints using PID.
        
        Args:
            route_waypoints: Route waypoints [F, 2]
            velocity: Current velocity in m/s
            speed_waypoints: Speed waypoints [F, 2]

        Returns:
            Tuple of (steer, target_speed_command, brake, desired_speed)
        """
        # Calculate desired speed from speed waypoints
        # Data is collected at 5Hz (0.2s interval).
        # Slicing [1:-1] in dataset means:
        # wp[0] is t=0.4s
        # wp[1] is t=0.6s
        # wp[2] is t=0.8s
        # Interval wp[2] - wp[0] is 0.4s.
        # Speed = Distance / Time = Distance * 2.5
        if len(speed_waypoints) > 2:
            desired_speed = np.linalg.norm(speed_waypoints[2] - speed_waypoints[0]) * 2.5
        elif len(speed_waypoints) >= 2:
            # Interval wp[1] - wp[0] is 0.2s. Multiplier 5.0.
            desired_speed = np.linalg.norm(speed_waypoints[1] - speed_waypoints[0]) * 5.0
        else:
            desired_speed = 0.0
        
        # Clamp desired speed to physical max
        raw_desired_speed = float(np.clip(desired_speed, 0.0, self.config.qcar2_max_speed))
        
        # Asymmetric Exponential Moving Average (EMA) Smoothing
        # Filter out single-frame high-speed spikes while preserving fast braking response
        if raw_desired_speed < self.smoothed_desired_speed:
            # Braking: Fast response (trust the brake signal immediately)
            alpha = 1.0 
        else:
            # Accelerating: Slow response (filter out noise/spikes)
            # If we were at 0 and model predicts 3.1 for 1 frame, we only go to ~0.6
            alpha = 0.2
            
        self.smoothed_desired_speed = alpha * raw_desired_speed + (1.0 - alpha) * self.smoothed_desired_speed
        
        # Use smoothed speed for control logic
        desired_speed = self.smoothed_desired_speed
        
        # Steering calculation
        route_interp = self.interpolate_waypoints(route_waypoints)
        steer = self.turn_controller.step(route_interp, velocity)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)
        
        # SIMPLIFIED BRAKE LOGIC - Trust the model's predictions!
        # Brake if model predicts very low speed OR if we are going much faster than desired
        brake_threshold = self.config.brake_speed  # 0.4 m/s
        
        if desired_speed < brake_threshold:
            brake = True
            target_speed_cmd = 0.0
        elif (velocity / (desired_speed + 1e-6)) > self.config.brake_ratio:
            # Emergency Brake: We are going much faster than the model wants (e.g. 4.0 vs 1.0)
            brake = True
            target_speed_cmd = desired_speed # Keep target but apply brake flag for sharp decel
        else:
            brake = False
            target_speed_cmd = desired_speed
        
        # COLD START LOGIC - Handle initial movement from stopped position
        # Problem: Model trained on moving data struggles to predict initial movement from standstill
        # Solution: Apply small minimum speed when stopped and model wants to move
        cold_start_threshold = 0.05  # m/s - consider stopped below this
        cold_start_min_speed = 0.3   # m/s - minimum target to break standstill
        min_desired_for_cold_start = 0.4  # m/s - Only cold start if we actually want to move (was 0.1)
        
        if velocity < cold_start_threshold and desired_speed > min_desired_for_cold_start:
            # Vehicle is stopped but model wants to move (even slightly)
            # Apply minimum speed to overcome initial friction and get into moving state
            target_speed_cmd = max(target_speed_cmd, cold_start_min_speed)
            brake = False
        
        # Minimal velocity hold: if we were braking and still moving very slowly,
        # keep braking to prevent jitter at near-zero speeds
        velocity_hold_threshold = 0.15  # m/s
        if self.prev_brake and velocity > 0 and velocity < velocity_hold_threshold:
            if desired_speed < brake_threshold:
                brake = True
                target_speed_cmd = 0.0
        
        # Remember previous brake state
        self.prev_brake = brake
        
        return steer, target_speed_cmd, brake, desired_speed


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
        dt: float,
        target_speed_cmd: float,
        brake: bool
    ) -> Tuple[float, float]:
        """
        Convert Simlingo control to QCar2 control.

        Args:
            desired_speed: Raw model target speed in m/s (for diagnostics/display)
            steer: Steering value [-1, 1]
            current_speed: Current speed in m/s
            dt: Time step in seconds
            target_speed_cmd: Desired forward velocity in m/s (after any adjustments)
            brake: Brake flag

        Returns:
            Tuple of (forward_velocity, turn_angle)
        """
        # Directly track commanded speed with simple rate limiting for physical plausibility
        target_velocity = max(target_speed_cmd, 0.0)
        speed_diff = target_velocity - current_speed

        max_accel = self.config.qcar2_max_acceleration * dt
        max_decel = self.config.qcar2_max_deceleration * dt

        if speed_diff > 0:
            speed_diff = min(speed_diff, max_accel)
        else:
            speed_diff = max(speed_diff, -max_decel)

        forward_velocity = current_speed + speed_diff

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
