"""
Route Management Module.
Manages waypoints and converts global target points to ego frame.
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Optional

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class RouteManager:
    """Manages route waypoints and target point selection."""
    
    def __init__(self, config):
        """
        Initialize route manager.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config
        
        # Route waypoints in world coordinates
        self.route_waypoints = np.array(config.route_waypoints, dtype=np.float32)
        
        # Current waypoint index
        self.current_waypoint_index = 0
        
        # Target point lookahead distance
        self.lookahead_distance = config.target_point_lookahead
        
    def get_target_point(self, current_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get target point based on current position.

        Args:
            current_position: Current vehicle position [x, y, z]

        Returns:
            Tuple of (target_point, next_target_point, hlc) where:
            - target_point: [x, y, z] in world coordinates
            - next_target_point: [x, y, z] in world coordinates
            - hlc: High-level command (1-6)
                1: Turn left at intersection
                2: Turn right at intersection
                3: Go straight at intersection
                4: Follow the road (default)
                5: Lane change left
                6: Lane change right
        """
        # Find nearest waypoint AHEAD of current position to update progress
        search_start = self.current_waypoint_index
        search_end = min(self.current_waypoint_index + 10, len(self.route_waypoints))  # Look ahead max 10 waypoints

        if search_end > search_start:
            distances_ahead = np.linalg.norm(
                self.route_waypoints[search_start:search_end, :2] - current_position[:2],
                axis=1
            )
            nearest_idx_relative = np.argmin(distances_ahead)
            nearest_idx = search_start + nearest_idx_relative

            # Update current waypoint index (only move forward)
            if nearest_idx > self.current_waypoint_index:
                self.current_waypoint_index = nearest_idx
        
        # Find target waypoint based on lookahead distance
        target_idx = self.current_waypoint_index
        accumulated_distance = 0.0

        # Start accumulating from current position to first waypoint
        if self.current_waypoint_index < len(self.route_waypoints):
            accumulated_distance = np.linalg.norm(
                self.route_waypoints[self.current_waypoint_index, :2] - current_position[:2]
            )

        for i in range(self.current_waypoint_index, len(self.route_waypoints) - 1):
            if accumulated_distance >= self.lookahead_distance:
                target_idx = i
                break

            segment_distance = np.linalg.norm(
                self.route_waypoints[i + 1, :2] - self.route_waypoints[i, :2]
            )
            accumulated_distance += segment_distance
            target_idx = i + 1

        # Ensure we don't go past the end
        target_idx = min(target_idx, len(self.route_waypoints) - 1)
        # Get target point
        target_point = self.route_waypoints[target_idx]
        # Get next target point (one waypoint ahead)
        next_target_idx = min(target_idx + 1, len(self.route_waypoints) - 1)
        next_target_point = self.route_waypoints[next_target_idx]

        # Compute HLC (High-Level Command)
        # For now, default to HLC=4 ("follow the road")
        # TODO: Implement proper HLC computation based on route geometry
        hlc = 4

        return target_point, next_target_point, hlc
    
    def get_target_point_ego(self, current_position: np.ndarray, current_heading: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Get target point in ego vehicle frame.

        Args:
            current_position: Current vehicle position [x, y, z]
            current_heading: Current vehicle heading in radians

        Returns:
            Tuple of (target_point_ego, next_target_point_ego, hlc) where:
            - target_point_ego: [x, y] in ego frame
            - next_target_point_ego: [x, y] in ego frame
            - hlc: High-level command (1-6)
        """
        # Get target points in world frame
        target_world, next_target_world, hlc = self.get_target_point(current_position)

        # Convert to ego frame
        target_ego = self._world_to_ego(target_world[:2], current_position[:2], current_heading)
        next_target_ego = self._world_to_ego(next_target_world[:2], current_position[:2], current_heading)

        return target_ego, next_target_ego, hlc
    
    def _world_to_ego(self, world_point: np.ndarray, vehicle_pos: np.ndarray, vehicle_heading: float) -> np.ndarray:
        """
        Convert world coordinates to ego vehicle frame.
        Matches the original Simlingo implementation (inverse_conversion_2d).

        Args:
            world_point: Point in world coordinates [x, y]
            vehicle_pos: Vehicle position [x, y]
            vehicle_heading: Vehicle heading in radians (yaw)

        Returns:
            Point in ego frame [x, y]
        """
        # Create rotation matrix (same as original Simlingo)
        rotation_matrix = np.array([
            [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
            [np.sin(vehicle_heading), np.cos(vehicle_heading)]
        ])

        # Apply transformation: R^T @ (point - translation)
        # This matches the original Simlingo inverse_conversion_2d function
        ego_point = rotation_matrix.T @ (world_point - vehicle_pos)

        return ego_point.astype(np.float32)
    
    def is_route_complete(self, current_position: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Check if route is complete.
        
        Args:
            current_position: Current vehicle position [x, y, z]
            threshold: Distance threshold to final waypoint (meters)
            
        Returns:
            True if route is complete, False otherwise
        """
        final_waypoint = self.route_waypoints[-1]
        distance_to_final = np.linalg.norm(final_waypoint[:2] - current_position[:2])
        
        return distance_to_final < threshold
    
    def get_progress(self, current_position: np.ndarray) -> float:
        """
        Get route progress as percentage.
        
        Args:
            current_position: Current vehicle position [x, y, z]
            
        Returns:
            Progress percentage (0.0 to 1.0)
        """
        if len(self.route_waypoints) <= 1:
            return 1.0
        
        # Find nearest waypoint
        distances = np.linalg.norm(self.route_waypoints[:, :2] - current_position[:2], axis=1)
        nearest_idx = np.argmin(distances)
        
        # Calculate progress
        progress = nearest_idx / (len(self.route_waypoints) - 1)
        
        return np.clip(progress, 0.0, 1.0)
    
    def add_waypoint(self, waypoint: np.ndarray):
        """
        Add waypoint to route.
        
        Args:
            waypoint: Waypoint [x, y, z]
        """
        self.route_waypoints = np.vstack([self.route_waypoints, waypoint])
    
    def set_route(self, waypoints: List[List[float]]):
        """
        Set new route waypoints.
        
        Args:
            waypoints: List of waypoints [[x, y, z], ...]
        """
        self.route_waypoints = np.array(waypoints, dtype=np.float32)
        self.current_waypoint_index = 0
    
    def reset(self):
        """Reset route manager to start of route."""
        self.current_waypoint_index = 0

