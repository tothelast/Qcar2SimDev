"""
QCar2 QLabs Interface Module.
Handles connection to QLabs, QCar2 spawning, camera capture, and control commands.
"""

import sys
import os
import numpy as np
import cv2
from typing import Tuple, Optional

# Add python directory to path for QVL imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python'))

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2


class QCar2Interface:
    """Interface for controlling QCar2 in QLabs."""
    
    def __init__(self, config):
        """
        Initialize QCar2 interface.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config
        self.qlabs = None
        self.qcar = None
        self.connected = False
        
        # State tracking
        self.current_location = None
        self.current_rotation = None
        self.previous_location = None
        self.previous_time = None
        self.collision_detected = False
        
    def connect(self) -> bool:
        """
        Connect to QLabs.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.qlabs = QuanserInteractiveLabs()
            print(f"Connecting to QLabs at {self.config.qlabs_host}...")
            
            if not self.qlabs.open(self.config.qlabs_host):
                print("ERROR: Unable to connect to QLabs")
                return False
            
            print("Connected to QLabs successfully")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to connect to QLabs: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def spawn_qcar(self) -> bool:
        """
        Spawn QCar2 in QLabs.
        
        Returns:
            True if spawn successful, False otherwise
        """
        if not self.connected:
            print("ERROR: Not connected to QLabs")
            return False
        
        try:
            # Destroy any existing actors
            print("Destroying existing actors...")
            self.qlabs.destroy_all_spawned_actors()
            
            # Create QCar2 instance
            self.qcar = QLabsQCar2(self.qlabs)
            
            # Spawn QCar2
            print(f"Spawning QCar2 at location {self.config.qcar2_spawn_location}...")
            status = self.qcar.spawn_id(
                actorNumber=self.config.qcar2_actor_number,
                location=self.config.qcar2_spawn_location,
                rotation=self.config.qcar2_spawn_rotation,
                waitForConfirmation=True
            )
            
            if status != 0:
                print(f"ERROR: Failed to spawn QCar2, status code: {status}")
                return False
            
            print("QCar2 spawned successfully")
            
            # Initialize state
            self.current_location = np.array(self.config.qcar2_spawn_location, dtype=np.float32)
            self.current_rotation = np.array(self.config.qcar2_spawn_rotation, dtype=np.float32)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to spawn QCar2: {e}")
            return False
    
    def get_camera_image(self) -> Optional[np.ndarray]:
        """
        Capture image from QCar2 camera.

        Returns:
            RGB image as numpy array (H, W, 3) or None if failed
        """
        if not self.connected or self.qcar is None:
            print("ERROR: QCar2 not initialized")
            return None

        try:
            # Get image from CSI front camera
            # Note: QCar2's get_image() already decodes the JPG and returns a numpy array
            success, image = self.qcar.get_image(camera=self.config.qcar2_camera)

            if not success or image is None:
                print("ERROR: Failed to capture camera image")
                return None

            # QCar2 returns BGR image, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        except Exception as e:
            print(f"ERROR: Failed to get camera image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_control(self, forward_velocity: float, turn_angle: float) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Send control command to QCar2.
        
        Args:
            forward_velocity: Forward speed in m/s (full-scale)
            turn_angle: Turn angle in radians (positive = right)
            
        Returns:
            Tuple of (success, location, rotation)
            - success: True if command successful
            - location: [x, y, z] position
            - rotation: [roll, pitch, yaw] in radians
        """
        if not self.connected or self.qcar is None:
            print("ERROR: QCar2 not initialized")
            return False, None, None
        
        try:
            # Send velocity and turn command
            success, location, rotation, front_hit, rear_hit = self.qcar.set_velocity_and_request_state(
                forward=forward_velocity,
                turn=turn_angle,
                headlights=False,
                leftTurnSignal=False,
                rightTurnSignal=False,
                brakeSignal=False,
                reverseSignal=False
            )
            
            if not success:
                print("ERROR: Failed to set control")
                return False, None, None
            
            # Update state
            self.previous_location = self.current_location
            self.current_location = np.array(location, dtype=np.float32)
            self.current_rotation = np.array(rotation, dtype=np.float32)

            # Check for collisions
            self.collision_detected = front_hit or rear_hit
            if front_hit:
                print("WARNING: Front bumper collision detected")
            if rear_hit:
                print("WARNING: Rear bumper collision detected")

            return True, self.current_location, self.current_rotation
            
        except Exception as e:
            print(f"ERROR: Failed to set control: {e}")
            return False, None, None
    
    def check_collision(self) -> bool:
        """
        Check if a collision was detected in the last control update.

        Returns:
            True if collision detected, False otherwise
        """
        return self.collision_detected

    def get_state(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get current QCar2 state.

        Returns:
            Tuple of (location, rotation)
            - location: [x, y, z] position
            - rotation: [roll, pitch, yaw] in radians
        """
        return self.current_location, self.current_rotation
    
    def possess_camera(self, camera: int = None) -> bool:
        """
        Possess (take control of) a QCar2 camera view.
        
        Args:
            camera: Camera constant (default: CAMERA_TRAILING)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or self.qcar is None:
            print("ERROR: QCar2 not initialized")
            return False
        
        try:
            if camera is None:
                camera = QLabsQCar2.CAMERA_TRAILING
            
            success = self.qcar.possess(camera=camera)
            
            if not success:
                print(f"ERROR: Failed to possess camera {camera}")
                return False
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to possess camera: {e}")
            return False
    
    def close(self):
        """Close QLabs connection and cleanup."""
        if self.qlabs is not None:
            try:
                print("Closing QLabs connection...")
                self.qlabs.close()
                print("QLabs connection closed")
            except Exception as e:
                print(f"ERROR: Failed to close QLabs connection: {e}")
        
        self.connected = False
        self.qcar = None
        self.qlabs = None

