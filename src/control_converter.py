"""
Control Conversion Module.
Implements Simlingo PID controllers and converts outputs to QCar2 commands.
"""

import numpy as np
import math
from collections import deque
from scipy.interpolate import PchipInterpolator
from typing import Tuple


class PIDController:
    """Basic PID controller (exact Simlingo implementation)."""
    
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        """
        Initialize PID controller.
        
        Args:
            k_p: Proportional gain
            k_i: Integral gain
            k_d: Derivative gain
            n: Window size for integral and derivative
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self._window = deque([0 for _ in range(n)], maxlen=n)
    
    def step(self, error):
        """
        Compute PID control output.
        
        Args:
            error: Control error
            
        Returns:
            Control output
        """
        self._window.append(error)
        
        if len(self._window) >= 2:
            integral = sum(self._window) / len(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        
        return self.k_p * error + self.k_i * integral + self.k_d * derivative
    
    def reset(self):
        """Reset controller state."""
        self._window = deque([0 for _ in range(self._window.maxlen)], maxlen=self._window.maxlen)


class LateralPIDController:
    """Lateral PID controller for steering (exact Simlingo implementation)."""
    
    def __init__(self, config):
        """
        Initialize lateral PID controller.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.k_p = config.lateral_pid_kp
        self.k_d = config.lateral_pid_kd
        self.k_i = config.lateral_pid_ki
        self.speed_scale = config.lateral_pid_speed_scale
        self.speed_offset = config.lateral_pid_speed_offset
        self.default_lookahead = config.lateral_pid_default_lookahead
        self.speed_threshold = config.lateral_pid_speed_threshold
        self.n = config.lateral_pid_window_size
        self.inference_mode = True  # Always True for trained model
        
        self._window = []
    
    def step(self, route_np: np.ndarray, current_speed: float) -> float:
        """
        Compute steering control.
        
        Args:
            route_np: Route waypoints array (N, 2)
            current_speed: Current speed in m/s
            
        Returns:
            Steering value [-1, 1]
        """
        # Convert speed to km/h
        current_speed_kmh = current_speed * 3.6
        
        # Calculate lookahead distance
        if self.inference_mode:
            n_lookahead = np.clip(
                self.speed_scale * current_speed_kmh + self.speed_offset,
                24, 105
            ) / 10
            n_lookahead = n_lookahead - 2
            n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
        else:
            n_lookahead = int(min(
                np.clip(self.speed_scale * current_speed_kmh + self.speed_offset, 24, 105),
                route_np.shape[0] - 1
            ))
        
        n_lookahead = min(n_lookahead, len(route_np) - 1)
        
        # Get desired heading vector
        desired_heading_vec = route_np[n_lookahead]
        
        # Calculate heading error
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        
        # Scale heading error (legacy from previous implementation)
        heading_error = heading_error * 180.0 / np.pi / 90.0
        
        # Update window
        self._window.append(heading_error)
        self._window = self._window[-self.n:]
        
        # Calculate derivative and integral
        derivative = 0.0 if len(self._window) == 1 else self._window[-1] - self._window[-2]
        integral = np.mean(self._window)
        
        # PID control law
        steering = np.clip(
            self.k_p * heading_error + self.k_d * derivative + self.k_i * integral,
            -1.0, 1.0
        ).item()
        
        return steering
    
    def reset(self):
        """Reset controller state."""
        self._window = []


class ControlConverter:
    """Converts Simlingo model outputs to QCar2 control commands."""
    
    def __init__(self, config):
        """
        Initialize control converter.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config
        
        # Initialize PID controllers
        self.speed_controller = PIDController(
            k_p=config.speed_kp,
            k_i=config.speed_ki,
            k_d=config.speed_kd,
            n=config.speed_n
        )
        
        self.turn_controller = LateralPIDController(config)
        
        # State for kinematic bicycle model
        self.current_speed = 0.0
        
    def control_pid(
        self,
        route_waypoints: np.ndarray,
        velocity: float,
        speed_waypoints: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Compute control from waypoints using PID (exact Simlingo implementation).
        
        Args:
            route_waypoints: Route waypoints [F, 2]
            velocity: Current velocity in m/s
            speed_waypoints: Speed waypoints [F, 2]
            
        Returns:
            Tuple of (steer, throttle, brake)
        """
        # Calculate desired speed from speed waypoints
        # NOTE: The model was trained with data_save_freq=4, so it predicts 10 waypoints
        # But our config has data_save_freq=1, so we need to adjust the calculation
        # Original: one_second = carla_fps // (wp_dilation * data_save_freq) = 20 // 4 = 5
        # With 10 waypoints, we use indices 3 and 8 (half_second-2=3, one_second-2=8)
        model_data_save_freq = 4  # The model was trained with this value
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * model_data_save_freq))
        half_second = one_second // 2

        # Ensure we have enough waypoints
        if len(speed_waypoints) < one_second:
            # Fallback: use all available waypoints
            if len(speed_waypoints) >= 2:
                desired_speed = np.linalg.norm(
                    speed_waypoints[-1] - speed_waypoints[0]
                ) * 2.0 / len(speed_waypoints)
            else:
                desired_speed = 0.0
        else:
            desired_speed = np.linalg.norm(
                speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]
            ) * 2.0

        # Debug output for first call
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG control_pid: one_second={one_second}, half_second={half_second}")
            print(f"DEBUG control_pid: speed_waypoints length={len(speed_waypoints)}")
            print(f"DEBUG control_pid: desired_speed={desired_speed:.3f} m/s")
            print(f"DEBUG control_pid: velocity={velocity:.3f} m/s")
            print(f"DEBUG control_pid: brake_speed threshold={self.config.brake_speed:.3f} m/s")
            self._debug_printed = True

        # Brake logic
        brake = (
            (desired_speed < self.config.brake_speed) or
            ((velocity / (desired_speed + 1e-6)) > self.config.brake_ratio)
        )

        # Throttle calculation
        delta = np.clip(desired_speed - velocity, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        
        # Steering calculation
        route_interp = self.interpolate_waypoints(route_waypoints)
        steer = self.turn_controller.step(route_interp, velocity)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Debug output for first call
        if not hasattr(self, '_steer_debug_printed'):
            print(f"DEBUG steering: route_waypoints[:3] = {route_waypoints[:3]}")
            print(f"DEBUG steering: route_interp[:5] = {route_interp[:5]}")
            print(f"DEBUG steering: velocity = {velocity:.3f} m/s")
            print(f"DEBUG steering: n_lookahead = {self.turn_controller._window[-1] if hasattr(self.turn_controller, '_window') and self.turn_controller._window else 'N/A'}")
            print(f"DEBUG steering: heading_error = {self.turn_controller._window[-1] if hasattr(self.turn_controller, '_window') and self.turn_controller._window else 'N/A'}")
            print(f"DEBUG steering: steer = {steer}")
            self._steer_debug_printed = True

        return steer, throttle, brake
    
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
        steer: float,
        throttle: float,
        brake: bool,
        current_speed: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Convert Simlingo control to QCar2 control.
        
        Args:
            steer: Steering value [-1, 1]
            throttle: Throttle value [0, 1]
            brake: Brake flag
            current_speed: Current speed in m/s
            dt: Time step in seconds
            
        Returns:
            Tuple of (forward_velocity, turn_angle)
            - forward_velocity: Target speed in m/s
            - turn_angle: Turn angle in radians
        """
        # Update current speed using kinematic bicycle model
        self.current_speed = self.bicycle_model_step(
            current_speed, dt, steer, throttle, brake
        )

        # Convert steer to turn angle
        # NOTE: QCar2 convention is opposite to CARLA/Simlingo:
        # - CARLA/Simlingo: positive steering = left turn
        # - QCar2: positive turn_angle = right turn
        # So we negate the steering value
        turn_angle = -steer * self.config.steering_gain

        # Forward velocity is the predicted speed
        forward_velocity = self.current_speed

        return forward_velocity, turn_angle
    
    def bicycle_model_step(
        self,
        speed: float,
        dt: float,
        steer: float,
        throttle: float,
        brake: bool
    ) -> float:
        """
        Kinematic bicycle model for speed prediction (exact Simlingo implementation).
        
        Args:
            speed: Current speed in m/s
            dt: Time step in seconds
            steer: Steering value [-1, 1]
            throttle: Throttle value [0, 1]
            brake: Brake flag
            
        Returns:
            Next speed in m/s
        """
        # Calculate acceleration
        if brake:
            accel = self.config.brake_acceleration
        else:
            accel = self.config.throttle_acceleration * throttle
        
        # Update speed
        next_speed = speed + accel * dt
        next_speed = max(next_speed, 0.0)  # ReLU
        
        return next_speed
    
    def reset(self):
        """Reset controller state."""
        self.speed_controller.reset()
        self.turn_controller.reset()
        self.current_speed = 0.0

