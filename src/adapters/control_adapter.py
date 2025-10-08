"""
Control Adapter for converting SimLingo outputs to Qcar2 control commands.

Uses the original SimLingo finite-difference speed calculation and braking logic.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Qcar2ControlAdapter:
    """Adapter for converting SimLingo outputs to Qcar2 control commands."""

    def __init__(self,
                 max_forward_speed: float = 5.0,
                 max_turn_angle: float = 0.6,
                 brake_speed: float = 0.4,
                 brake_ratio: float = 1.1):
        """
        Initialize the control adapter.

        Args:
            max_forward_speed: Maximum forward speed in m/s
            max_turn_angle: Maximum turn angle in radians
            brake_speed: Speed threshold below which brake is triggered (m/s)
            brake_ratio: Ratio of current/desired speed above which brake is triggered
        """
        self.max_forward_speed = max_forward_speed
        self.max_turn_angle = max_turn_angle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio

    def process_simlingo_output(self, model_output: Dict[str, Any], current_speed: float = 0.0) -> tuple[float, float]:
        """
        Convert SimLingo predictions to (forward_speed, turn_angle) using original SimLingo logic.

        Args:
            model_output: Dict containing 'pred_speed_wps' [N,2] waypoints in vehicle frame
            current_speed: Current vehicle speed in m/s

        Returns:
            (forward_speed, turn_angle) tuple
        """
        # SimLingo outputs pred_speed_wps: [N,2] waypoints at 5 Hz (0.2s intervals)
        wps = np.asarray(model_output['pred_speed_wps'], dtype=float)

        logger.debug("="*60)
        logger.debug("CONTROL ADAPTER: Processing SimLingo Output")
        logger.debug(f"  Waypoints shape: {wps.shape}")
        logger.debug(f"  Current speed: {current_speed:.3f} m/s")
        logger.debug(f"  First 5 waypoints:")
        for i in range(min(5, len(wps))):
            logger.debug(f"    wps[{i}]: [{wps[i,0]:7.3f}, {wps[i,1]:7.3f}]")

        # Original SimLingo speed calculation (finite-difference)
        # Waypoints at 5 Hz: use half_second=2 (~0.4s) and one_second=4 (~0.8-1.0s)
        # Formula: desired_speed = ||wps[half_second-2] - wps[one_second-2]|| * 2.0
        #        = ||wps[0] - wps[2]|| * 2.0
        idx_start = 0  # half_second - 2 = 2 - 2 = 0
        idx_end = 2    # one_second - 2 = 4 - 2 = 2
        wp_start = wps[idx_start]
        wp_end = wps[idx_end]
        distance = float(np.linalg.norm(wp_start - wp_end))
        desired_speed = distance * 2.0

        logger.debug(f"  Speed calculation:")
        logger.debug(f"    Using waypoints: wps[{idx_start}] - wps[{idx_end}]")
        logger.debug(f"    wps[{idx_start}]: [{wp_start[0]:7.3f}, {wp_start[1]:7.3f}]")
        logger.debug(f"    wps[{idx_end}]: [{wp_end[0]:7.3f}, {wp_end[1]:7.3f}]")
        logger.debug(f"    Distance: {distance:.4f} m")
        logger.debug(f"    Desired speed (distance * 2.0): {desired_speed:.4f} m/s")

        # Original SimLingo braking logic
        brake_condition_1 = desired_speed < self.brake_speed
        brake_condition_2 = current_speed > 0.01 and (current_speed / max(desired_speed, 0.01)) > self.brake_ratio
        should_brake = brake_condition_1 or brake_condition_2

        logger.debug(f"  Braking logic:")
        logger.debug(f"    Brake threshold: {self.brake_speed} m/s")
        logger.debug(f"    Brake ratio threshold: {self.brake_ratio}")
        logger.debug(f"    Condition 1 (desired_speed < {self.brake_speed}): {brake_condition_1}")
        if current_speed > 0.01:
            speed_ratio = current_speed / max(desired_speed, 0.01)
            logger.debug(f"    Condition 2 (speed_ratio {speed_ratio:.3f} > {self.brake_ratio}): {brake_condition_2}")
        else:
            logger.debug(f"    Condition 2: False (current_speed too low)")
        logger.debug(f"    Should brake: {should_brake}")

        if should_brake:
            forward_speed = 0.0
            logger.debug(f"  BRAKING: forward_speed set to 0.0")
        else:
            forward_speed_raw = desired_speed
            forward_speed = np.clip(desired_speed, 0.0, self.max_forward_speed)
            logger.debug(f"  Forward speed (raw): {forward_speed_raw:.4f} m/s")
            logger.debug(f"  Forward speed (clipped to max {self.max_forward_speed}): {forward_speed:.4f} m/s")

        # Steering: angle to waypoint at index 2 (0.4s ahead)
        steer_idx = 2
        dx, dy = float(wps[steer_idx, 0]), float(wps[steer_idx, 1])
        angle = float(np.arctan2(dy, dx))
        turn_angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)

        logger.debug(f"  Steering calculation:")
        logger.debug(f"    Target waypoint: wps[{steer_idx}] = [{dx:7.3f}, {dy:7.3f}]")
        logger.debug(f"    Angle (raw): {np.degrees(angle):7.2f}°")
        logger.debug(f"    Turn angle (clipped to ±{np.degrees(self.max_turn_angle):.1f}°): {np.degrees(turn_angle):7.2f}°")

        logger.debug(f"  FINAL CONTROL: forward={forward_speed:.4f} m/s, turn={np.degrees(turn_angle):.2f}°")
        logger.debug("="*60)

        return float(forward_speed), float(turn_angle)

    def send_control_command(self,
                           qcar2_vehicle,
                           forward_speed: float,
                           turn_angle: float) -> tuple[bool, dict]:
        """
        Send control command to Qcar2 vehicle.

        Args:
            qcar2_vehicle: QLabsQCar2 instance
            forward_speed: Forward speed in m/s
            turn_angle: Turn angle in radians

        Returns:
            (success, info) where info includes location, rotation, front_hit, rear_hit
        """

        # Send command to vehicle
        success, location, rotation, front_hit, rear_hit = qcar2_vehicle.set_velocity_and_request_state(
            forward=forward_speed,
            turn=turn_angle,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            brakeSignal=False,
            reverseSignal=False
        )

        info = {
            "location": location,
            "rotation": rotation,
            "front_hit": front_hit,
            "rear_hit": rear_hit,
        }

        return success, info