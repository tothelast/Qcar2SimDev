"""
Control Conversion Module.
Implements Simlingo PID controllers and converts outputs to QCar2 commands.
"""

import sys
import os
import numpy as np
import math
from collections import deque
from scipy.interpolate import PchipInterpolator
from typing import Tuple

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class LateralPIDController:
    """Lateral PID controller for steering (exact Simlingo implementation)."""

    def __init__(self, config):
        """
        Initialize lateral PID controller.

        Args:
            config: SimlingoQCar2Config instance
        """
        self.k_p = config.turn_kp
        self.k_i = config.turn_ki
        self.k_d = config.turn_kd
        self.n = config.turn_n
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)

        # Speed-dependent aim distances (meters)
        self.aim_distance_slow = 2.25
        self.aim_distance_fast = 3.0
        self.aim_distance_very_fast = 7.0
        self.aim_distance_threshold = 5.5  # m/s
        self.aim_distance_threshold2 = 15.0  # m/s
    
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

        # Calculate aim distance based on speed
        if current_speed_kmh < self.aim_distance_threshold:
            aim_distance = self.aim_distance_slow
        elif current_speed_kmh < self.aim_distance_threshold2:
            aim_distance = self.aim_distance_fast
        else:
            aim_distance = self.aim_distance_very_fast

        # Convert to waypoint index (assuming 0.1m spacing between waypoints)
        n_lookahead = int(min(aim_distance * 10, len(route_np) - 1))

        # Get desired heading vector
        desired_heading_vec = route_np[n_lookahead]

        # Calculate heading error
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

        # Update window
        self._window.append(heading_error)

        # Calculate derivative and integral
        if len(self._window) >= 2:
            integral = sum(self._window) / len(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        # PID control law
        steering = self.k_p * heading_error + self.k_i * integral + self.k_d * derivative
        steering = np.clip(steering, -1.0, 1.0)

        return steering

    def reset(self):
        """Reset controller state."""
        self._window = deque([0 for _ in range(self.n)], maxlen=self.n)


class LongitudinalLinearRegressionController:
    """
    Longitudinal controller using linear regression (exact SimLingo implementation).
    This is the DEFAULT controller used by SimLingo, not the PID controller.
    """

    def __init__(self, config):
        """
        Initialize longitudinal linear regression controller.

        Args:
            config: SimlingoQCar2Config instance
        """
        # Minimum threshold for target speed (< 1 km/h)
        self.minimum_target_speed = 0.278  # m/s

        # Coefficients of the linear regression model
        # Source: team_code/config.py - longitudinal_linear_regression_params
        self.params = np.array([
            1.1990342347353184,   # current_speed coefficient
            -0.8057602384167799,  # current_speed^2 coefficient
            1.710818710950062,    # 100*speed_error_cl coefficient
            0.921890257450335,    # speed_error_cl^2 coefficient
            1.556497522998393,    # current_speed*speed_error_cl coefficient
            -0.7013479734904027,  # current_speed^2*speed_error_cl coefficient
            1.031266635497984     # braking ratio threshold
        ])

        # Maximum acceleration rate (approximately 1.9 m/tick)
        self.max_acceleration = 1.89

        # Maximum deceleration rate (approximately -4.82 m/tick)
        self.max_deceleration = -4.82

    def get_throttle_and_brake(
        self,
        target_speed: float,
        current_speed: float
    ) -> Tuple[float, bool]:
        """
        Get throttle and brake values using linear regression.

        Args:
            target_speed: Desired target speed in m/s
            current_speed: Current speed of the vehicle in m/s

        Returns:
            Tuple of (throttle, brake) where:
                throttle: float in [0, 1]
                brake: bool (True to brake, False otherwise)
        """
        # If target speed is very small, apply braking
        if target_speed < 1e-5:
            return 0.0, True

        # Avoid very small target speeds
        target_speed = max(self.minimum_target_speed, target_speed)

        # Convert to km/h for calculation
        current_speed_kmh = current_speed * 3.6
        target_speed_kmh = target_speed * 3.6

        speed_error = target_speed_kmh - current_speed_kmh

        # Maximum acceleration check (1.9 m/tick)
        if speed_error > self.max_acceleration:
            return 1.0, False

        # Braking check using ratio threshold
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

        # QCar2-specific physical limits
        # These are hardcoded here since they're hardware-specific constants
        self.qcar2_max_speed = 3.0  # m/s (QCar2 physical maximum speed)
        self.qcar2_max_acceleration = 3.0  # m/s² (for smooth velocity transitions)
        
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
        # NOTE: The model outputs 10 speed waypoints, but empirically the original
        # calculation using indices [0, 3] works better than the "correct" [2, 4].
        # This may be compensating for a systematic bias in the model's speed predictions.
        #
        # Original calculation (empirically better):
        # - Uses data_save_freq=4 (waypoints at 0.25s intervals)
        # - one_second = carla_fps // (wp_dilation * data_save_freq) = 20 // 4 = 5
        # - half_second = 5 // 2 = 2
        # - Indices: [half_second-2, one_second-2] = [0, 3]
        # - Time between waypoints[0] and [3]: 0.75s
        # - Formula: distance * 2.0 (assumes 0.5s, so underestimates by 33%)
        # - This underestimation appears to compensate for model over-prediction

        model_data_save_freq = 4
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * model_data_save_freq))
        half_second = one_second // 2

        # IMPORTANT: Speed calculation method (EXACT SimLingo implementation)
        # =====================================================================
        # We use EUCLIDEAN DISTANCE to match the original SimLingo training.
        #
        # The model was trained with this calculation, so changing it would
        # invalidate the model's learned speed-waypoint relationships.
        #
        # For teleop fine-tuning: Use the SAME calculation during training and deployment.
        # This ensures consistency and allows transfer learning from pre-trained weights.
        #
        # Note: Waypoints are cumsum-decoded positions (see adaptors.py line 175)
        # representing absolute future positions in ego frame.

        # Calculate desired speed using Euclidean distance (original SimLingo)
        idx1 = half_second - 2  # Index 0
        idx2 = one_second - 2   # Index 3

        if len(speed_waypoints) > idx2:
            # Standard case: use waypoints[0] to waypoints[3]
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
        steer: float,
        throttle: float,
        brake: bool,
        desired_speed: float,
        current_speed: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Convert Simlingo control to QCar2 control using direct velocity control.

        This method uses QCar2's native velocity-based control interface instead of
        simulating CARLA's acceleration-based dynamics with the bicycle model.

        Args:
            steer: Steering value [-1, 1]
            throttle: Throttle value [0, 1] (kept for signature compatibility, not used)
            brake: Brake flag
            desired_speed: Target speed from model's speed waypoints (m/s)
            current_speed: Current speed in m/s
            dt: Time step in seconds

        Returns:
            Tuple of (forward_velocity, turn_angle)
            - forward_velocity: Target speed in m/s
            - turn_angle: Turn angle in radians
        """
        # Compute target velocity using direct velocity control
        # (instead of CARLA's bicycle model with throttle/brake → acceleration → speed)
        if brake:
            # Brake: target zero velocity
            target_speed = 0.0
        else:
            # Use desired speed from model's speed waypoints directly
            # NOTE: We don't scale by throttle because QCar2 uses direct velocity control,
            # not acceleration-based control like CARLA. The throttle value was computed
            # for CARLA's acceleration dynamics and is not needed for velocity commands.
            # The smoothing with qcar2_max_acceleration already handles gradual transitions.
            target_speed = desired_speed

            # IMPORTANT: Clip to QCar2's physical maximum speed
            # The model may predict speeds beyond QCar2's capabilities (e.g., 7-10 m/s)
            # since it was trained on full-scale CARLA vehicles
            target_speed = min(target_speed, self.qcar2_max_speed)

        # Apply smooth velocity transitions with QCar2-appropriate acceleration limits
        # This prevents jerky motion and respects QCar2's physical capabilities
        speed_diff = target_speed - current_speed
        max_speed_change = self.qcar2_max_acceleration * dt

        # Clip speed change to acceleration limits
        forward_velocity = current_speed + np.clip(
            speed_diff,
            -max_speed_change,  # Max deceleration
            max_speed_change    # Max acceleration
        )

        # Ensure non-negative velocity and respect maximum speed
        forward_velocity = np.clip(forward_velocity, 0.0, self.qcar2_max_speed)

        # Convert steer to turn angle
        # NOTE: QCar2 convention is opposite to CARLA/Simlingo:
        # - CARLA/Simlingo: positive steering = left turn
        # - QCar2: positive turn_angle = right turn
        # So we negate the steering value
        turn_angle = -steer * self.config.steering_gain

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

        NOTE: This method is NO LONGER USED in the current implementation.
        We now use direct velocity control (convert_to_qcar2_control) instead of
        simulating CARLA's acceleration-based dynamics with the bicycle model.

        This method is kept for reference and potential future use (e.g., if deploying
        to physical QCar hardware that requires PWM-based control).

        Args:
            speed: Current speed in m/s
            dt: Time step in seconds
            steer: Steering value [-1, 1] (unused but kept for signature compatibility)
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

