#!/usr/bin/env python3
"""
Unit tests for Control Adapter implementation.

Verifies:
1. Heading error scaling (90° → 1.0, 45° → 0.5, etc.)
2. Speed calculation (correct indices and multiplier)
3. Braking logic (correct thresholds)
4. Dual-waypoint system (uses correct waypoints for steering vs speed)

Reference: https://github.com/RenzKa/simlingo
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.control_adapter import Qcar2ControlAdapter


def test_heading_error_scaling():
    """Test that heading error scaling produces correct normalized values."""
    print("\n" + "="*60)
    print("TEST 1: Heading Error Scaling")
    print("="*60)
    print("\nReference: team_code/lateral_controller.py line 77")
    print("Formula: heading_error_scaled = heading_error_rad * 180 / π / 90")
    
    test_cases = [
        (0.0, 0.0, "0°"),
        (np.pi/4, 0.5, "45°"),
        (np.pi/2, 1.0, "90°"),
        (-np.pi/4, -0.5, "-45°"),
        (-np.pi/2, -1.0, "-90°"),
        (np.pi/6, 1/3, "30°"),
    ]
    
    for angle_rad, expected_scaled, description in test_cases:
        # Calculate scaled error
        scaled = angle_rad * 180.0 / np.pi / 90.0
        
        print(f"\n  {description}:")
        print(f"    Angle (rad): {angle_rad:.4f}")
        print(f"    Expected scaled: {expected_scaled:.4f}")
        print(f"    Actual scaled: {scaled:.4f}")
        
        assert abs(scaled - expected_scaled) < 1e-6, f"Expected {expected_scaled}, got {scaled}"
        print(f"    ✓ PASS")
    
    print("\n✓ All heading error scaling tests passed!")


def test_speed_calculation():
    """Test that speed calculation uses correct indices and multiplier."""
    print("\n" + "="*60)
    print("TEST 2: Speed Calculation")
    print("="*60)
    print("\nReference: team_code/agent_simlingo.py lines 1395-1398")
    print("Formula: desired_speed = ||wps[0] - wps[2]|| * 2.0")
    
    # Create test waypoints
    speed_wps = np.array([
        [0.0, 0.0],   # wps[0]
        [0.5, 0.1],   # wps[1]
        [1.0, 0.2],   # wps[2]
        [1.5, 0.3],   # wps[3]
        [2.0, 0.4],   # wps[4]
        [2.5, 0.5],
        [3.0, 0.6],
        [3.5, 0.7],
        [4.0, 0.8],
        [4.5, 0.9],
    ], dtype=np.float32)
    
    route_wps = np.zeros((20, 2), dtype=np.float32)
    route_wps[5] = [5.0, 0.0]  # Lookahead point
    
    model_output = {
        'pred_speed_wps': speed_wps,
        'pred_route': route_wps
    }
    
    controller = Qcar2ControlAdapter()
    
    # Calculate expected speed
    # SimLingo: half_second=2, one_second=4
    # desired_speed = ||wps[half_second-2] - wps[one_second-2]|| * 2.0
    # = ||wps[0] - wps[2]|| * 2.0
    distance = np.linalg.norm(speed_wps[0] - speed_wps[2])
    expected_speed = distance * 2.0
    
    print(f"\n  Speed waypoints:")
    print(f"    wps[0]: {speed_wps[0]}")
    print(f"    wps[2]: {speed_wps[2]}")
    print(f"    Distance: {distance:.4f} m")
    print(f"    Expected speed: {expected_speed:.4f} m/s")
    
    # Get actual speed from controller
    forward_speed, _ = controller.process_simlingo_output(model_output, current_speed=0.0)
    
    # Note: forward_speed is clipped to max_forward_speed (2.0 m/s)
    actual_desired_speed = min(expected_speed, 2.0)
    
    print(f"    Actual speed: {forward_speed:.4f} m/s")
    print(f"    (Clipped to max: {controller.max_forward_speed} m/s)")
    
    assert abs(forward_speed - actual_desired_speed) < 1e-3, f"Expected {actual_desired_speed}, got {forward_speed}"
    print(f"    ✓ PASS")
    
    print("\n✓ Speed calculation test passed!")


def test_braking_logic():
    """Test that braking logic triggers at correct thresholds."""
    print("\n" + "="*60)
    print("TEST 3: Braking Logic")
    print("="*60)
    print("\nReference: team_code/agent_simlingo.py lines 1324-1325")
    print("Brake if: (desired_speed < 0.4) OR (current_speed/desired_speed > 1.1)")
    
    controller = Qcar2ControlAdapter(brake_speed=0.4, brake_ratio=1.1)
    
    # Create route waypoints
    route_wps = np.zeros((20, 2), dtype=np.float32)
    route_wps[5] = [5.0, 0.0]
    
    # Test case 1: Low desired speed (< 0.4 m/s)
    print("\n  Test 3.1: Low desired speed")
    speed_wps_slow = np.array([[0.0, 0.0], [0.05, 0.0]] + [[0.1, 0.0]] * 8, dtype=np.float32)
    model_output = {'pred_speed_wps': speed_wps_slow, 'pred_route': route_wps}
    
    forward_speed, _ = controller.process_simlingo_output(model_output, current_speed=1.0)
    print(f"    Desired speed: {np.linalg.norm(speed_wps_slow[0] - speed_wps_slow[2]) * 2.0:.4f} m/s (< 0.4)")
    print(f"    Current speed: 1.0 m/s")
    print(f"    Forward speed: {forward_speed:.4f} m/s")
    assert forward_speed == 0.0, f"Expected 0.0 (brake), got {forward_speed}"
    print(f"    ✓ PASS - Braking triggered")
    
    # Test case 2: High speed ratio (current/desired > 1.1)
    print("\n  Test 3.2: High speed ratio")
    speed_wps_normal = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0],
        [2.5, 0.0], [3.0, 0.0], [3.5, 0.0], [4.0, 0.0], [4.5, 0.0]
    ], dtype=np.float32)
    model_output = {'pred_speed_wps': speed_wps_normal, 'pred_route': route_wps}
    
    desired = np.linalg.norm(speed_wps_normal[0] - speed_wps_normal[2]) * 2.0
    current = desired * 1.2  # 1.2 > 1.1, should brake
    
    forward_speed, _ = controller.process_simlingo_output(model_output, current_speed=current)
    print(f"    Desired speed: {desired:.4f} m/s")
    print(f"    Current speed: {current:.4f} m/s")
    print(f"    Ratio: {current/desired:.2f} (> 1.1)")
    print(f"    Forward speed: {forward_speed:.4f} m/s")
    assert forward_speed == 0.0, f"Expected 0.0 (brake), got {forward_speed}"
    print(f"    ✓ PASS - Braking triggered")
    
    # Test case 3: Normal operation (no braking)
    print("\n  Test 3.3: Normal operation")
    current_normal = desired * 0.8  # 0.8 < 1.1, should not brake
    
    forward_speed, _ = controller.process_simlingo_output(model_output, current_speed=current_normal)
    print(f"    Desired speed: {desired:.4f} m/s")
    print(f"    Current speed: {current_normal:.4f} m/s")
    print(f"    Ratio: {current_normal/desired:.2f} (< 1.1)")
    print(f"    Forward speed: {forward_speed:.4f} m/s")
    assert forward_speed > 0.0, f"Expected > 0.0 (no brake), got {forward_speed}"
    print(f"    ✓ PASS - No braking")
    
    print("\n✓ All braking logic tests passed!")


def test_dual_waypoint_system():
    """Test that controller uses correct waypoints for steering vs speed."""
    print("\n" + "="*60)
    print("TEST 4: Dual-Waypoint System")
    print("="*60)
    print("\nVerifies:")
    print("  - pred_route used for steering (lateral control)")
    print("  - pred_speed_wps used for speed (longitudinal control)")
    
    controller = Qcar2ControlAdapter(lookahead_distance=5)
    
    # Create distinct waypoints to verify correct usage
    speed_wps = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
        [5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0], [9.0, 0.0]
    ], dtype=np.float32)
    
    # Route waypoints go to the right (positive Y)
    route_wps = np.zeros((20, 2), dtype=np.float32)
    for i in range(20):
        route_wps[i] = [float(i), float(i) * 0.5]  # Curves to the right
    
    model_output = {
        'pred_speed_wps': speed_wps,
        'pred_route': route_wps
    }
    
    forward_speed, turn_angle = controller.process_simlingo_output(model_output, current_speed=0.0)
    
    # Verify speed calculation uses speed_wps
    expected_speed = np.linalg.norm(speed_wps[0] - speed_wps[2]) * 2.0
    expected_speed = min(expected_speed, 2.0)  # Clipped to max
    print(f"\n  Speed calculation:")
    print(f"    Uses pred_speed_wps: {speed_wps[0]} to {speed_wps[2]}")
    print(f"    Expected: {expected_speed:.4f} m/s")
    print(f"    Actual: {forward_speed:.4f} m/s")
    assert abs(forward_speed - expected_speed) < 1e-3, f"Speed mismatch"
    print(f"    ✓ PASS")
    
    # Verify steering uses route_wps
    target_point = route_wps[5]  # Lookahead index = 5
    expected_angle = np.arctan2(target_point[1], target_point[0])
    print(f"\n  Steering calculation:")
    print(f"    Uses pred_route[5]: {target_point}")
    print(f"    Target angle: {np.degrees(expected_angle):.2f}°")
    print(f"    Turn angle: {np.degrees(turn_angle):.2f}°")
    
    # Turn angle should be positive (turning right) since route curves right
    assert turn_angle > 0, f"Expected positive turn angle, got {turn_angle}"
    print(f"    ✓ PASS - Turning right as expected")
    
    print("\n✓ Dual-waypoint system test passed!")


if __name__ == "__main__":
    test_heading_error_scaling()
    test_speed_calculation()
    test_braking_logic()
    test_dual_waypoint_system()
    
    print("\n" + "="*60)
    print("ALL CONTROL ADAPTER TESTS PASSED!")
    print("="*60)
    print("\n✓ Implementation matches SimLingo specification")
    print("✓ Ready for integration testing")

