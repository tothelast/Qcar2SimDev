"""
Coordinate Frame Transformation Utilities.

Provides utilities for converting between world and ego vehicle coordinate frames.
This module consolidates coordinate transformation logic used throughout the codebase.
"""

import numpy as np


class CoordinateTransformer:
    """Handles coordinate frame transformations between world and ego vehicle frames."""
    
    @staticmethod
    def world_to_ego(
        world_point: np.ndarray,
        vehicle_pos: np.ndarray,
        vehicle_heading: float
    ) -> np.ndarray:
        """
        Convert world coordinates to ego vehicle frame.
        
        Transforms a point from the global world coordinate system to the vehicle's
        local ego frame, where x-axis points forward and y-axis points left.
        
        This implementation matches the original Simlingo inverse_conversion_2d function
        used in the CARLA simulator training.
        
        Args:
            world_point: Point in world coordinates [x, y] or [x, y, z]
            vehicle_pos: Vehicle position in world frame [x, y]
            vehicle_heading: Vehicle heading (yaw angle) in radians
        
        Returns:
            Point in ego frame [x, y] or [x, y, z] (preserves z-coordinate if present)
        """
        
        # Create rotation matrix for the vehicle heading
        # This rotates from world frame to ego frame
        rotation_matrix = np.array([
            [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
            [np.sin(vehicle_heading), np.cos(vehicle_heading)]
        ])
        
        # Apply transformation: R^T @ (point - translation)
        # Translate to vehicle origin, then rotate to ego frame
        ego_point = rotation_matrix.T @ (world_point[:2] - vehicle_pos)
        
        # Preserve z-coordinate if present
        if len(world_point) == 3:
            return np.array([ego_point[0], ego_point[1], world_point[2]], dtype=np.float32)
        
        return ego_point.astype(np.float32)
    
    @staticmethod
    def ego_to_world(
        ego_point: np.ndarray,
        vehicle_pos: np.ndarray,
        vehicle_heading: float
    ) -> np.ndarray:
        """
        Convert ego vehicle frame to world coordinates.
        
        Transforms a point from the vehicle's local ego frame back to the global
        world coordinate system.
        
        Args:
            ego_point: Point in ego frame [x, y] or [x, y, z]
            vehicle_pos: Vehicle position in world frame [x, y]
            vehicle_heading: Vehicle heading (yaw angle) in radians
        
        Returns:
            Point in world coordinates [x, y] or [x, y, z] (preserves z-coordinate if present)
        """

        # Create rotation matrix for the vehicle heading
        # This rotates from ego frame back to world frame
        rotation_matrix = np.array([
            [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
            [np.sin(vehicle_heading), np.cos(vehicle_heading)]
        ])
        
        # Apply transformation: R @ point + translation
        # Rotate from ego frame to world frame, then translate to world origin
        world_point_2d = rotation_matrix @ ego_point[:2] + vehicle_pos
        
        # Preserve z-coordinate if present
        if len(ego_point) == 3:
            return np.array([world_point_2d[0], world_point_2d[1], ego_point[2]], dtype=np.float32)
        
        return world_point_2d.astype(np.float32)
    
    @staticmethod
    def verify_round_trip(
        original_point: np.ndarray,
        vehicle_pos: np.ndarray,
        vehicle_heading: float,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Verify that world->ego->world conversion is accurate.
        
        Useful for testing and debugging coordinate transformations.
        
        Args:
            original_point: Original point in world coordinates
            vehicle_pos: Vehicle position in world frame
            vehicle_heading: Vehicle heading in radians
            tolerance: Maximum allowed error for round-trip conversion
        
        Returns:
            True if round-trip conversion is accurate within tolerance
        """
        # Convert world -> ego -> world
        ego_point = CoordinateTransformer.world_to_ego(original_point, vehicle_pos, vehicle_heading)
        recovered_point = CoordinateTransformer.ego_to_world(ego_point, vehicle_pos, vehicle_heading)
        
        # Check if recovered point matches original
        error = np.linalg.norm(recovered_point[:2] - original_point[:2])
        return error < tolerance

