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
        
        Raises:
            ValueError: If inputs have invalid shapes or types
        
        Example:
            >>> world_pt = np.array([10.0, 5.0])
            >>> vehicle_pos = np.array([0.0, 0.0])
            >>> heading = 0.0  # Facing +x direction
            >>> ego_pt = CoordinateTransformer.world_to_ego(world_pt, vehicle_pos, heading)
            >>> np.allclose(ego_pt, [10.0, 5.0])
            True
        """
        # Validate inputs
        if not isinstance(world_point, np.ndarray):
            raise ValueError(f"world_point must be numpy array, got {type(world_point)}")
        if not isinstance(vehicle_pos, np.ndarray):
            raise ValueError(f"vehicle_pos must be numpy array, got {type(vehicle_pos)}")
        if world_point.size < 2:
            raise ValueError(f"world_point must have at least 2 elements, got {world_point.size}")
        if vehicle_pos.size < 2:
            raise ValueError(f"vehicle_pos must have at least 2 elements, got {vehicle_pos.size}")
        
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
        
        Raises:
            ValueError: If inputs have invalid shapes or types
        
        Example:
            >>> ego_pt = np.array([10.0, 5.0])
            >>> vehicle_pos = np.array([0.0, 0.0])
            >>> heading = 0.0  # Facing +x direction
            >>> world_pt = CoordinateTransformer.ego_to_world(ego_pt, vehicle_pos, heading)
            >>> np.allclose(world_pt, [10.0, 5.0])
            True
        """
        # Validate inputs
        if not isinstance(ego_point, np.ndarray):
            raise ValueError(f"ego_point must be numpy array, got {type(ego_point)}")
        if not isinstance(vehicle_pos, np.ndarray):
            raise ValueError(f"vehicle_pos must be numpy array, got {type(vehicle_pos)}")
        if ego_point.size < 2:
            raise ValueError(f"ego_point must have at least 2 elements, got {ego_point.size}")
        if vehicle_pos.size < 2:
            raise ValueError(f"vehicle_pos must have at least 2 elements, got {vehicle_pos.size}")
        
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
        
        Example:
            >>> pt = np.array([10.0, 5.0])
            >>> pos = np.array([1.0, 2.0])
            >>> heading = np.pi / 4
            >>> CoordinateTransformer.verify_round_trip(pt, pos, heading)
            True
        """
        # Convert world -> ego -> world
        ego_point = CoordinateTransformer.world_to_ego(original_point, vehicle_pos, vehicle_heading)
        recovered_point = CoordinateTransformer.ego_to_world(ego_point, vehicle_pos, vehicle_heading)
        
        # Check if recovered point matches original
        error = np.linalg.norm(recovered_point[:2] - original_point[:2])
        return error < tolerance

