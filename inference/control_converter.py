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
        # Speed-based aim distance (Match Simlingo reference implementation)
        # Reference: nav_planner.py LateralPIDController.step
        # n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
        # Default values from nav_planner.py: speed_scale=0.9755, speed_offset=1.915
        
        speed_kmh = current_speed * 3.6
        
        # Linear lookahead formula from reference
        lookahead_index = np.clip(0.9755 * speed_kmh + 1.915, 24, 105)
        
        # Convert index to distance (reference uses 0.1m spacing)
        # However, our route_np is already interpolated to 0.1m spacing in interpolate_waypoints
        # So we can use the index directly
        n_lookahead = int(min(lookahead_index, len(route_np) - 1))
        
        desired_heading_vec = route_np[n_lookahead]

        # Calculate heading error
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        # Scale the heading error (Match Simlingo reference implementation)
        # Simlingo scales by 180/pi/90 (approx 0.637), effectively normalizing 90 degrees to 1.0
        # Without this, QCar2 steering is ~1.57x more aggressive than intended
        heading_error = heading_error * 180.0 / np.pi / 90.0

        self._window.append(heading_error)

        # PID terms
        integral = sum(self._window) / len(self._window) if len(self._window) >= 2 else 0.0
        
        # Derivative term
        # Reference code calculates derivative as (current - prev) without dividing by dt
        # Simlingo runs at 20Hz (dt=0.05s). QCar2 runs at 4Hz (dt=0.25s).
        # For the same physical movement, the error change per step is ~5x larger in QCar2.
        # We must divide by 5.0 to normalize the derivative term to the 20Hz expectation.
        raw_derivative = self._window[-1] - self._window[-2] if len(self._window) >= 2 else 0.0
        derivative = raw_derivative / 5.0

        steering = self.k_p * heading_error + self.k_i * integral + self.k_d * derivative
        return np.clip(steering, -1.0, 1.0)

    def reset(self):
        """Reset controller state."""
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)


class ControlConverter:
    """Converts Simlingo model outputs to QCar2 control commands."""

    def __init__(self, config):
        """
        Initialize control converter.

        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config

        self.turn_controller = LateralPIDController(config)
        
        # Minimal state: just track previous brake for velocity hold
        self.prev_brake = False


        
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
        # Determine time delta between waypoints based on config
        # Data collection interval = dt * data_save_freq
        dt_model = self.config.dt * self.config.data_save_freq
        
        if len(speed_waypoints) > 2:
            # Interval wp[2] - wp[0] is 2 steps
            time_delta = 2.0 * dt_model
            desired_speed = np.linalg.norm(speed_waypoints[2] - speed_waypoints[0]) / time_delta
        elif len(speed_waypoints) >= 2:
            # Interval wp[1] - wp[0] is 1 step
            time_delta = dt_model
            desired_speed = np.linalg.norm(speed_waypoints[1] - speed_waypoints[0]) / time_delta
        else:
            desired_speed = 0.0
        
        # Clamp desired speed to physical max
        desired_speed = float(np.clip(desired_speed, 0.0, self.config.qcar2_max_speed))
        
        # Steering calculation
        route_interp = self.interpolate_waypoints(route_waypoints)
        steer = self.turn_controller.step(route_interp, velocity)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)
        
        # Brake logic: 
        # 1. Stop if desired speed is low (e.g. < 0.5 m/s)
        # If the model predicts a crawl, we interpret it as a desire to stop.
        if desired_speed < 0.5:
            brake = True
            target_speed_cmd = 0.0
        # 2. Apply brake if we need to decelerate (e.g. > 0.2 m/s difference)
        # Catch even smooth deceleration trends from the model
        elif (velocity - desired_speed) > 0.2:
            brake = True
            target_speed_cmd = desired_speed
        else:
            brake = False
            target_speed_cmd = desired_speed
            
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
        
        # Panic Brake Logic: If brake flag is set, force 0.0 velocity immediately
        # This overrides the smooth deceleration ramp to ensure we stop ASAP
        if brake:
            forward_velocity = 0.0

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
