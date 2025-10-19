#!/usr/bin/env python3
"""
Teleop Controller Module

Provides keyboard-based teleop control for QCar2 vehicles.

Controls:
- Arrow Up / W: Accelerate forward
- Arrow Down / S: Brake / Reverse
- Arrow Left / A: Steer left
- Arrow Right / D: Steer right
- Space: Emergency stop
- Q / ESC: Quit
"""

import time
import numpy as np
from pynput import keyboard


class TeleopController:
    """
    Keyboard-based teleop controller for QCar2.

    Controls:
    - Arrow Up / W: Accelerate forward
    - Arrow Down / S: Brake / Reverse
    - Arrow Left / A: Steer left
    - Arrow Right / D: Steer right
    - Space: Emergency stop
    - Q / ESC: Quit
    """

    def __init__(self):
        """Initialize teleop controller."""
        # Control parameters
        self.max_forward_velocity = 4.0  # m/s (full-scale) - reduced for better control
        self.max_reverse_velocity = -2.0  # m/s
        self.max_steering_angle = np.pi / 9  # 20 degrees - reduced for smoother lane-keeping

        # Acceleration/deceleration rates
        self.acceleration = 1.0  # m/s²
        self.deceleration = 2.0  # m/s²
        self.steering_rate = np.pi / 4  # 45°/s - reduced for smoother steering transitions

        # Current state
        self.target_velocity = 0.0
        self.current_velocity = 0.0
        self.target_steering = 0.0
        self.current_steering = 0.0

        # Pressed keys tracking
        self.pressed_keys = set()

        # Control flags
        self.running = True
        self.emergency_stop = False

        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        """Handle key press events."""
        try:
            # Handle special keys
            if key == keyboard.Key.up:
                self.pressed_keys.add('up')
            elif key == keyboard.Key.down:
                self.pressed_keys.add('down')
            elif key == keyboard.Key.left:
                self.pressed_keys.add('left')
            elif key == keyboard.Key.right:
                self.pressed_keys.add('right')
            elif key == keyboard.Key.space:
                self.emergency_stop = True
                self.target_velocity = 0.0
                self.current_velocity = 0.0
            elif key == keyboard.Key.esc:
                self.running = False
            # Handle character keys
            elif hasattr(key, 'char'):
                if key.char in ['w', 'W']:
                    self.pressed_keys.add('up')
                elif key.char in ['s', 'S']:
                    self.pressed_keys.add('down')
                elif key.char in ['a', 'A']:
                    self.pressed_keys.add('left')
                elif key.char in ['d', 'D']:
                    self.pressed_keys.add('right')
                elif key.char in ['q', 'Q']:
                    self.running = False
        except Exception as e:
            print(f"Key press error: {e}")

    def _on_release(self, key):
        """Handle key release events."""
        try:
            # Handle special keys
            if key == keyboard.Key.up:
                self.pressed_keys.discard('up')
            elif key == keyboard.Key.down:
                self.pressed_keys.discard('down')
            elif key == keyboard.Key.left:
                self.pressed_keys.discard('left')
            elif key == keyboard.Key.right:
                self.pressed_keys.discard('right')
            # Handle character keys
            elif hasattr(key, 'char'):
                if key.char in ['w', 'W']:
                    self.pressed_keys.discard('up')
                elif key.char in ['s', 'S']:
                    self.pressed_keys.discard('down')
                elif key.char in ['a', 'A']:
                    self.pressed_keys.discard('left')
                elif key.char in ['d', 'D']:
                    self.pressed_keys.discard('right')
        except Exception as e:
            print(f"Key release error: {e}")

    def update(self, dt):
        """
        Update control state based on pressed keys.

        Args:
            dt: Time delta in seconds
        """
        # Reset emergency stop if any key is pressed
        if len(self.pressed_keys) > 0:
            self.emergency_stop = False

        # Update target velocity based on pressed keys
        if 'up' in self.pressed_keys:
            self.target_velocity = self.max_forward_velocity
        elif 'down' in self.pressed_keys:
            self.target_velocity = self.max_reverse_velocity
        else:
            self.target_velocity = 0.0

        # Update target steering based on pressed keys
        if 'left' in self.pressed_keys and 'right' not in self.pressed_keys:
            self.target_steering = -self.max_steering_angle  # Negative = left
        elif 'right' in self.pressed_keys and 'left' not in self.pressed_keys:
            self.target_steering = self.max_steering_angle  # Positive = right
        else:
            self.target_steering = 0.0

        # Smooth velocity transition
        velocity_diff = self.target_velocity - self.current_velocity
        if abs(velocity_diff) > 0.01:
            if velocity_diff > 0:
                # Accelerating
                self.current_velocity += min(self.acceleration * dt, velocity_diff)
            else:
                # Decelerating
                self.current_velocity += max(-self.deceleration * dt, velocity_diff)
        else:
            self.current_velocity = self.target_velocity

        # Smooth steering transition
        steering_diff = self.target_steering - self.current_steering
        if abs(steering_diff) > 0.01:
            max_change = self.steering_rate * dt
            self.current_steering += np.clip(steering_diff, -max_change, max_change)
        else:
            self.current_steering = self.target_steering

    def get_control(self):
        """
        Get current control values.

        Returns:
            Tuple of (velocity, steering_angle)
        """
        if self.emergency_stop:
            return 0.0, 0.0
        return self.current_velocity, self.current_steering

    def stop(self):
        """Stop the keyboard listener."""
        self.running = False
        self.listener.stop()


def teleop_control_loop(qcar, teleop_controller):
    """
    Control loop for teleop-controlled QCar2.

    Args:
        qcar: QLabsQCar2 instance
        teleop_controller: TeleopController instance
    """
    print(f"  Starting teleop control loop...")
    print(f"  Max forward velocity: {teleop_controller.max_forward_velocity} m/s")
    print(f"  Max reverse velocity: {teleop_controller.max_reverse_velocity} m/s")
    print(f"  Max steering: {np.degrees(teleop_controller.max_steering_angle):.1f}°")

    # Control loop timing
    dt = 0.1  # 10 Hz control loop
    iteration = 0

    # Error tracking
    consecutive_errors = 0
    max_consecutive_errors = 10

    print(f"  Teleop QCar2 ready for manual control!")

    while teleop_controller.running:
        try:
            iteration += 1

            # Update teleop controller state
            teleop_controller.update(dt)

            # Get control values
            velocity, steering = teleop_controller.get_control()

            # Send control command
            try:
                success, location, rotation, _, _ = qcar.set_velocity_and_request_state(
                    forward=velocity,
                    turn=steering,
                    headlights=True,
                    leftTurnSignal=steering > 0.1,  # Left turn signal when steering left
                    rightTurnSignal=steering < -0.1,  # Right turn signal when steering right
                    brakeSignal=velocity < -0.1,  # Brake signal when reversing
                    reverseSignal=velocity < -0.1  # Reverse signal when reversing
                )
            except Exception:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print(f"  ✗ Too many consecutive errors, stopping teleop")
                    break
                time.sleep(0.2)
                continue

            if not success:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print(f"  ✗ Too many consecutive failures, stopping teleop")
                    break
                time.sleep(0.2)
                continue

            # Reset error counter on success
            consecutive_errors = 0

            # Print status every 2 seconds
            if iteration % 20 == 0:
                print(f"  Teleop: v={velocity:.2f} m/s, steer={np.degrees(steering):.1f}°, "
                      f"pos=[{location[0]:.1f}, {location[1]:.1f}]")

            # Control loop rate - 10 Hz
            time.sleep(dt)

        except Exception as e:
            print(f"  ✗ Teleop control error: {e}")
            time.sleep(1.0)

    print(f"  Teleop control loop stopped")

