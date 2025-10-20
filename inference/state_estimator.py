"""
Vehicle State Estimation Module.
Tracks vehicle position, velocity, and heading from QCar2 state feedback.
"""

import numpy as np
import time
from typing import Tuple, Optional
from collections import deque

from core.coordinate_utils import CoordinateTransformer


class StateEstimator:
    """Estimates vehicle state from QCar2 feedback."""
    
    def __init__(self, config):
        """
        Initialize state estimator.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config
        
        # State variables
        self.position = None  # [x, y, z]
        self.rotation = None  # [roll, pitch, yaw]
        self.velocity = 0.0  # m/s
        self.heading = 0.0  # radians
        
        # History for filtering
        self.position_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        # GPS and compass for ego frame conversion
        self.gps = None  # Global position [x, y]
        self.compass = None  # Heading in radians
        
        # Timestamp
        self.last_update_time = None
        
    def update(self, location: np.ndarray, rotation: np.ndarray):
        """
        Update state estimate with new measurements.
        
        Args:
            location: [x, y, z] position from QCar2
            rotation: [roll, pitch, yaw] in radians from QCar2
        """
        current_time = time.time()
        
        # Update position and rotation
        self.position = location.copy()
        self.rotation = rotation.copy()
        self.heading = rotation[2]  # Yaw angle
        
        # Update GPS and compass
        self.gps = location[:2].copy()  # [x, y]
        self.compass = self.heading
        
        # Add to history
        self.position_history.append(self.position.copy())
        self.time_history.append(current_time)
        
        # Calculate velocity from position changes
        if len(self.position_history) >= 2 and len(self.time_history) >= 2:
            # Use last two positions
            pos_prev = self.position_history[-2]
            pos_curr = self.position_history[-1]
            time_prev = self.time_history[-2]
            time_curr = self.time_history[-1]
            
            # Calculate displacement
            displacement = pos_curr - pos_prev
            dt = time_curr - time_prev
            
            if dt > 0:
                # Calculate speed (magnitude of velocity)
                speed = np.linalg.norm(displacement[:2]) / dt  # Use only x, y
                self.velocity_history.append(speed)
                
                # Use filtered velocity (moving average)
                self.velocity = np.mean(self.velocity_history)
            else:
                self.velocity = 0.0
        else:
            self.velocity = 0.0
        
        self.last_update_time = current_time
    
    def get_velocity(self) -> float:
        """
        Get current velocity estimate.
        
        Returns:
            Velocity in m/s
        """
        return self.velocity
    
    def get_position(self) -> np.ndarray:
        """
        Get current position.
        
        Returns:
            Position [x, y, z]
        """
        return self.position if self.position is not None else np.zeros(3)
    
    def get_rotation(self) -> np.ndarray:
        """
        Get current rotation.
        
        Returns:
            Rotation [roll, pitch, yaw] in radians
        """
        return self.rotation if self.rotation is not None else np.zeros(3)
    
    def get_heading(self) -> float:
        """
        Get current heading (yaw angle).
        
        Returns:
            Heading in radians
        """
        return self.heading
    
    def get_gps(self) -> np.ndarray:
        """
        Get GPS position (x, y).
        
        Returns:
            GPS position [x, y]
        """
        return self.gps if self.gps is not None else np.zeros(2)
    
    def get_compass(self) -> float:
        """
        Get compass heading.
        
        Returns:
            Compass heading in radians
        """
        return self.compass if self.compass is not None else 0.0
    
    def world_to_ego(self, world_point: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to ego vehicle frame.

        Args:
            world_point: Point in world coordinates [x, y] or [x, y, z]

        Returns:
            Point in ego frame [x, y] or [x, y, z]
        """
        if self.position is None or self.rotation is None:
            return world_point

        # Use shared coordinate transformation utility
        return CoordinateTransformer.world_to_ego(
            world_point,
            self.position[:2],
            self.heading
        )

    def ego_to_world(self, ego_point: np.ndarray) -> np.ndarray:
        """
        Convert ego vehicle frame to world coordinates.

        Args:
            ego_point: Point in ego frame [x, y] or [x, y, z]

        Returns:
            Point in world coordinates [x, y] or [x, y, z]
        """
        if self.position is None or self.rotation is None:
            return ego_point

        # Use shared coordinate transformation utility
        return CoordinateTransformer.ego_to_world(
            ego_point,
            self.position[:2],
            self.heading
        )
    
    def reset(self):
        """Reset state estimator."""
        self.position = None
        self.rotation = None
        self.velocity = 0.0
        self.heading = 0.0
        self.gps = None
        self.compass = None
        self.position_history.clear()
        self.time_history.clear()
        self.velocity_history.clear()
        self.last_update_time = None

