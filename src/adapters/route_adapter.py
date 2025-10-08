"""
Route Adapter for managing navigation waypoints from route.json.

Handles:
- Loading route waypoints (world frame)
- Tracking progress along route
- Computing target points in vehicle ego frame
"""

import json
import numpy as np 
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
 

class RouteAdapter:
    """Manages route waypoints and computes target points for SimLingo."""
    
    def __init__(self):
        self.waypoints_world: List[Tuple[float, float]] = []
        self.route_name: str = ""
        self.spacing_m: float = 0.5
        self.current_waypoint_idx: int = 0
        self.route_complete: bool = False
        
    def load_route(self, route_path: str) -> bool:
        """
        Load route from JSON file created by route_builder.py.
        
        Args:
            route_path: Path to route.json file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(route_path, 'r') as f:
                data = json.load(f)
            
            self.route_name = data.get('name', 'unknown')
            self.spacing_m = data.get('spacing_m', 0.5)
            self.waypoints_world = [tuple(pt) for pt in data['points_world']]
            self.current_waypoint_idx = 0
            self.route_complete = False
            
            logger.info(f"Loaded route '{self.route_name}' with {len(self.waypoints_world)} waypoints")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load route from {route_path}: {e}")
            return False
    
    def find_nearest_waypoint(self, position: Tuple[float, float]) -> int:
        """
        Find index of nearest waypoint to current position.
        
        Args:
            position: Current vehicle position [x, y] in world frame
            
        Returns:
            Index of nearest waypoint
        """
        if not self.waypoints_world:
            return 0
        
        min_dist = float('inf')
        min_idx = self.current_waypoint_idx
        
        # Search forward from current index (vehicle should be progressing)
        for i in range(self.current_waypoint_idx, len(self.waypoints_world)):
            wp = self.waypoints_world[i]
            dist = np.linalg.norm([wp[0] - position[0], wp[1] - position[1]])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        return min_idx
    
    def get_lookahead_waypoint(self, 
                               position: Tuple[float, float],
                               lookahead_distance: float = 10.0) -> int:
        """
        Get waypoint index that is approximately lookahead_distance ahead.
        
        Args:
            position: Current vehicle position [x, y] in world frame
            lookahead_distance: Desired lookahead distance in meters
            
        Returns:
            Index of lookahead waypoint
        """
        if not self.waypoints_world:
            return 0
        
        # Update current waypoint to nearest
        self.current_waypoint_idx = self.find_nearest_waypoint(position)
        
        # Find waypoint approximately lookahead_distance ahead
        cumulative_dist = 0.0
        for i in range(self.current_waypoint_idx, len(self.waypoints_world) - 1):
            wp_curr = self.waypoints_world[i]
            wp_next = self.waypoints_world[i + 1]
            segment_dist = np.linalg.norm([wp_next[0] - wp_curr[0], 
                                          wp_next[1] - wp_curr[1]])
            cumulative_dist += segment_dist
            
            if cumulative_dist >= lookahead_distance:
                return i + 1
        
        # If we can't find a waypoint far enough ahead, return the last one
        return len(self.waypoints_world) - 1
    
    def world_to_ego(self, 
                     world_point: Tuple[float, float],
                     vehicle_position: Tuple[float, float],
                     vehicle_yaw_rad: float) -> Tuple[float, float]:
        """
        Transform point from world frame to vehicle ego frame.
        
        QLabs world frame: X-East, Y-North, Z-Up
        Vehicle ego frame: X-forward, Y-left, Z-up
        
        Args:
            world_point: Point in world frame [x, y]
            vehicle_position: Vehicle position in world frame [x, y]
            vehicle_yaw_rad: Vehicle yaw in radians (0 = East, π/2 = North)
            
        Returns:
            Point in ego frame [x_forward, y_left]
        """
        # Translate to vehicle origin
        dx = world_point[0] - vehicle_position[0]
        dy = world_point[1] - vehicle_position[1]
        
        # Rotate by -yaw to align with vehicle heading
        # In QLabs: yaw=0 is East, yaw=π/2 is North
        # We want ego frame where X is forward (along vehicle heading)
        cos_yaw = np.cos(-vehicle_yaw_rad)
        sin_yaw = np.sin(-vehicle_yaw_rad)
        
        x_ego = dx * cos_yaw - dy * sin_yaw
        y_ego = dx * sin_yaw + dy * cos_yaw
        
        return (float(x_ego), float(y_ego))
    
    def get_target_point(self,
                        position: Tuple[float, float],
                        orientation: float,
                        lookahead_distance: float = 10.0) -> Tuple[float, float]:
        """
        Get target navigation point in vehicle ego frame.
        
        Args:
            position: Current vehicle position [x, y] in world frame
            orientation: Current vehicle yaw in radians
            lookahead_distance: How far ahead to look for target (meters)
            
        Returns:
            Target point in ego frame [x_forward, y_left]
        """
        if not self.waypoints_world:
            logger.warning("No route loaded, returning origin as target")
            return (0.0, 0.0)
        
        # Get lookahead waypoint index
        target_idx = self.get_lookahead_waypoint(position, lookahead_distance)
        
        # Check if route is complete
        if target_idx >= len(self.waypoints_world) - 1:
            self.route_complete = True
        
        # Get target waypoint in world frame
        target_world = self.waypoints_world[target_idx]
        
        # Transform to ego frame
        target_ego = self.world_to_ego(target_world, position, orientation)
        
        return target_ego
    
    def get_progress(self) -> float:
        """
        Get route completion progress as percentage.
        
        Returns:
            Progress from 0.0 to 1.0
        """
        if not self.waypoints_world:
            return 0.0
        return self.current_waypoint_idx / max(1, len(self.waypoints_world) - 1)
    
    def is_route_complete(self, position: Tuple[float, float], threshold: float = 2.0) -> bool:
        """
        Check if vehicle has reached the final waypoint.
        
        Args:
            position: Current vehicle position [x, y]
            threshold: Distance threshold in meters
            
        Returns:
            True if within threshold of final waypoint
        """
        if not self.waypoints_world:
            return False
        
        final_wp = self.waypoints_world[-1]
        dist = np.linalg.norm([final_wp[0] - position[0], final_wp[1] - position[1]])
        return dist < threshold

