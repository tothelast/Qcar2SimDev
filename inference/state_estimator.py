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
        
        # History for state tracking
        # Note: We use instantaneous velocity (single-frame delta) to match training data.
        # Training data uses:
        #   - CARLA: vehicle.get_velocity().length() (instantaneous)
        #   - QLabs: distance / dt (single-frame delta)
        # Using moving average here would create train-inference mismatch and lag.
        self.position_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        self.prev_position = None
        self.prev_time = None
        
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
        
        # Calculate velocity from position changes (instantaneous, single-frame delta)
        # This matches the training data collection method:
        #   - CARLA: vehicle.get_velocity().length()
        #   - QLabs DataRecorder: distance / dt (single frame)
        if self.prev_position is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                # Calculate speed from single-frame displacement (matches training)
                displacement = self.position - self.prev_position
                self.velocity = np.linalg.norm(displacement[:2]) / dt
            else:
                self.velocity = 0.0
        else:
            self.velocity = 0.0

        # Store current as previous for next frame
        self.prev_position = self.position.copy()
        self.prev_time = current_time
        
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
        self.prev_position = None
        self.prev_time = None
        self.last_update_time = None

