#!/usr/bin/env python3
"""
Unit tests for PID controller implementation.

Verifies that our PID controller matches SimLingo's behavior exactly.
Reference: https://github.com/RenzKa/simlingo/blob/main/simlingo_training/utils/transfuser_utils.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from controllers.pid_controller import PIDController


def test_pid_basic():
    """Test basic PID functionality with known inputs/outputs."""
    print("\n" + "="*60)
    print("TEST 1: Basic PID Controller")
    print("="*60)
    
    # Create PID controller with simple gains
    pid = PIDController(k_p=1.0, k_i=0.5, k_d=0.2, n=5)
    
    # Test case 1: First error
    # Window is initialized with n=5 zeros: [0, 0, 0, 0, 0]
    # After first step (deque pops oldest): [0, 0, 0, 0, 1.0]
    error1 = 1.0
    output1 = pid.step(error1)
    # P = 1.0 * 1.0 = 1.0
    # I = 0.5 * mean([0,0,0,0,1.0]) = 0.5 * 0.2 = 0.1
    # D = 0.2 * (1.0 - 0.0) = 0.2
    expected1 = 1.0 + 0.1 + 0.2
    print(f"\nTest 1.1: First error")
    print(f"  Error: {error1}")
    print(f"  Window after: [0,0,0,0,{error1}]")
    print(f"  Expected output: {expected1:.4f}")
    print(f"  Actual output: {output1:.4f}")
    assert abs(output1 - expected1) < 1e-6, f"Expected {expected1}, got {output1}"
    print("  ✓ PASS")
    
    # Test case 2: Second error
    # Window after first: [0,0,0,0,1.0]
    # Window after second: [0,0,0,1.0,1.5]
    error2 = 1.5
    output2 = pid.step(error2)
    # P = 1.0 * 1.5 = 1.5
    # I = 0.5 * mean([0,0,0,1.0,1.5]) = 0.5 * 0.5 = 0.25
    # D = 0.2 * (1.5 - 1.0) = 0.2 * 0.5 = 0.1
    expected2 = 1.5 + 0.25 + 0.1
    print(f"\nTest 1.2: Second error")
    print(f"  Error: {error2}")
    print(f"  Expected output: {expected2:.4f}")
    print(f"  Actual output: {output2:.4f}")
    assert abs(output2 - expected2) < 1e-6, f"Expected {expected2}, got {output2}"
    print("  ✓ PASS")
    
    # Test case 3: Windowing (fill window and verify)
    pid_window = PIDController(k_p=1.0, k_i=1.0, k_d=1.0, n=3)
    errors = [1.0, 2.0, 3.0, 4.0]
    for e in errors:
        pid_window.step(e)
    
    # After 4 steps, window should contain [2.0, 3.0, 4.0] (last 3)
    # After step(5.0), window becomes [3.0, 4.0, 5.0]
    output = pid_window.step(5.0)
    # P = 1.0 * 5.0 = 5.0
    # I = 1.0 * mean([3.0, 4.0, 5.0]) = 1.0 * 4.0 = 4.0
    # D = 1.0 * (5.0 - 4.0) = 1.0
    expected = 5.0 + 4.0 + 1.0
    print(f"\nTest 1.3: Windowing")
    print(f"  Window size: 3")
    print(f"  Errors: {errors} -> 5.0")
    print(f"  Expected output: {expected:.4f}")
    print(f"  Actual output: {output:.4f}")
    assert abs(output - expected) < 1e-6, f"Expected {expected}, got {output}"
    print("  ✓ PASS")
    
    print("\n✓ All PID basic tests passed!")


def test_pid_simlingo_params():
    """Test PID with SimLingo's actual parameters."""
    print("\n" + "="*60)
    print("TEST 2: PID with SimLingo Parameters")
    print("="*60)
    
    # Lateral PID parameters from SimLingo
    lateral_pid = PIDController(k_p=3.25, k_i=1.0, k_d=1.0, n=20)

    # Test with typical heading error (scaled)
    # Example: 10° heading error -> scaled = 10/90 = 0.111
    heading_error_scaled = 10.0 / 90.0  # 0.111

    output1 = lateral_pid.step(heading_error_scaled)
    # First step: Window [0]*19 + [0.111]
    # P = 3.25 * 0.111 = 0.361
    # I = 1.0 * mean([0]*19 + [0.111]) = 1.0 * 0.111/20 = 0.00556
    # D = 1.0 * (0.111 - 0.0) = 0.111
    expected1 = 3.25 * heading_error_scaled + 1.0 * (heading_error_scaled/20) + 1.0 * heading_error_scaled
    print(f"\nTest 2.1: Lateral PID - First step")
    print(f"  Heading error (scaled): {heading_error_scaled:.4f}")
    print(f"  Expected output: {expected1:.4f}")
    print(f"  Actual output: {output1:.4f}")
    assert abs(output1 - expected1) < 1e-6, f"Expected {expected1}, got {output1}"
    print("  ✓ PASS")
    
    # Speed PID parameters from SimLingo
    speed_pid = PIDController(k_p=1.75, k_i=1.0, k_d=2.0, n=20)

    # Test with typical speed error
    speed_error = 0.5  # m/s

    output2 = speed_pid.step(speed_error)
    # P = 1.75 * 0.5 = 0.875
    # I = 1.0 * 0.5/20 = 0.025
    # D = 2.0 * 0.5 = 1.0
    expected2 = 1.75 * speed_error + 1.0 * (speed_error/20) + 2.0 * speed_error
    print(f"\nTest 2.2: Speed PID - First step")
    print(f"  Speed error: {speed_error:.4f} m/s")
    print(f"  Expected output: {expected2:.4f}")
    print(f"  Actual output: {output2:.4f}")
    assert abs(output2 - expected2) < 1e-6, f"Expected {expected2}, got {output2}"
    print("  ✓ PASS")
    
    print("\n✓ All SimLingo parameter tests passed!")


def test_pid_reset():
    """Test PID reset functionality."""
    print("\n" + "="*60)
    print("TEST 3: PID Reset")
    print("="*60)
    
    pid = PIDController(k_p=1.0, k_i=1.0, k_d=1.0, n=5)
    
    # Add some errors
    for i in range(5):
        pid.step(float(i))
    
    # Reset
    pid.reset()

    # First step after reset
    # Window after reset: [0,0,0,0,0]
    # After step(2.0): [0,0,0,0,2.0]
    output = pid.step(2.0)
    # P = 1.0 * 2.0 = 2.0
    # I = 1.0 * 2.0/5 = 0.4
    # D = 1.0 * 2.0 = 2.0
    expected = 2.0 + 0.4 + 2.0
    print(f"\nAfter reset, first step:")
    print(f"  Expected output: {expected:.4f}")
    print(f"  Actual output: {output:.4f}")
    assert abs(output - expected) < 1e-6, f"Expected {expected}, got {output}"
    print("  ✓ PASS")
    
    print("\n✓ PID reset test passed!")


if __name__ == "__main__":
    test_pid_basic()
    test_pid_simlingo_params()
    test_pid_reset()
    
    print("\n" + "="*60)
    print("ALL PID CONTROLLER TESTS PASSED!")
    print("="*60)

