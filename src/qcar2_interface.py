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
from qvl.spline_line import QLabsSplineLine


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

        # Trajectory tracers
        self.trajectory_tracer = None  # Actual trajectory (red)
        self.trajectory_points = []
        self.trajectory_update_counter = 0

        self.planned_route_tracer = None  # Planned route (green)
        
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

            # Initialize trajectory tracers if enabled
            if self.config.enable_trajectory_tracer:
                self._initialize_trajectory_tracer()

            if self.config.enable_planned_route_tracer:
                self._initialize_planned_route_tracer()

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

            # Update trajectory tracer
            self.update_trajectory_tracer(self.current_location)

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
        self.trajectory_tracer = None
        self.planned_route_tracer = None

    def _initialize_trajectory_tracer(self):
        """Initialize QLabs trajectory tracer (spline line)."""
        try:
            print("Initializing trajectory tracer...")
            self.trajectory_tracer = QLabsSplineLine(self.qlabs)

            # Spawn the spline line actor at origin
            # Configuration 1 = CURVE mode for smooth trajectory
            status = self.trajectory_tracer.spawn_id(
                actorNumber=100,  # Use actor number 100 for trajectory
                location=[0, 0, 0.01],  # Slightly above ground
                rotation=[0, 0, 0],
                configuration=1,  # CURVE configuration
                waitForConfirmation=True
            )

            if status == 0:
                # Initialize with starting position
                start_pos = self.config.qcar2_spawn_location
                self.trajectory_points = [[start_pos[0], start_pos[1], 0.01, self.config.trajectory_tracer_width]]
                print("Trajectory tracer initialized successfully")
            else:
                print(f"WARNING: Failed to spawn trajectory tracer, status: {status}")
                self.trajectory_tracer = None

        except Exception as e:
            print(f"WARNING: Failed to initialize trajectory tracer: {e}")
            self.trajectory_tracer = None

    def update_trajectory_tracer(self, location: np.ndarray):
        """
        Update trajectory tracer with new vehicle position.

        Args:
            location: Current vehicle position [x, y, z]
        """
        if self.trajectory_tracer is None or not self.config.enable_trajectory_tracer:
            return

        try:
            # Update counter
            self.trajectory_update_counter += 1

            # Only update every N steps to reduce overhead
            if self.trajectory_update_counter < self.config.trajectory_tracer_update_interval:
                return

            self.trajectory_update_counter = 0

            # Add new point to trajectory
            # Format: [x, y, z, width]
            new_point = [location[0], location[1], 0.01, self.config.trajectory_tracer_width]
            self.trajectory_points.append(new_point)

            # Limit trajectory length to prevent performance issues (keep last 500 points)
            if len(self.trajectory_points) > 500:
                self.trajectory_points = self.trajectory_points[-500:]

            # Update spline line with new points
            # Need at least 2 points to draw a line
            if len(self.trajectory_points) >= 2:
                self.trajectory_tracer.set_points(
                    color=self.config.trajectory_tracer_color,
                    pointList=self.trajectory_points,
                    alignEndPointTangents=False,
                    waitForConfirmation=False  # Non-blocking for performance
                )

        except Exception as e:
            # Silently fail to avoid disrupting control loop
            pass

    def _initialize_planned_route_tracer(self):
        """Initialize QLabs planned route tracer (green spline line)."""
        try:
            print("Initializing planned route tracer...")
            self.planned_route_tracer = QLabsSplineLine(self.qlabs)

            # Spawn the spline line actor at origin
            # Configuration 1 = CURVE mode for smooth route visualization
            status = self.planned_route_tracer.spawn_id(
                actorNumber=101,  # Use actor number 101 for planned route
                location=[0, 0, 0.02],  # Slightly higher than actual trajectory
                rotation=[0, 0, 0],
                configuration=1,  # CURVE configuration
                waitForConfirmation=True
            )

            if status == 0:
                # Convert route waypoints to spline line format
                # Format: [x, y, z, width]
                route_points = []
                for waypoint in self.config.route_waypoints:
                    route_points.append([
                        waypoint[0],  # x
                        waypoint[1],  # y
                        0.02,  # z (slightly above actual trajectory)
                        self.config.planned_route_tracer_width
                    ])

                # Draw the planned route
                if len(route_points) >= 2:
                    self.planned_route_tracer.set_points(
                        color=self.config.planned_route_tracer_color,
                        pointList=route_points,
                        alignEndPointTangents=False,
                        waitForConfirmation=True
                    )
                    print(f"Planned route tracer initialized successfully ({len(route_points)} waypoints)")
                else:
                    print("WARNING: Not enough route waypoints to display planned route")

            else:
                print(f"WARNING: Failed to spawn planned route tracer, status: {status}")
                self.planned_route_tracer = None

        except Exception as e:
            print(f"WARNING: Failed to initialize planned route tracer: {e}")
            self.planned_route_tracer = None

