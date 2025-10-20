"""QCar2 QLabs interface for connection, spawning, camera, and control."""

import numpy as np
import cv2
from typing import Tuple, Optional

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.spline_line import QLabsSplineLine

# Optional commentary window for inference
try:
    from inference.commentary_window import CommentaryWindow
except ImportError:
    CommentaryWindow = None


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
        """Connect to QLabs and set timeout for scene actors."""
        self.qlabs = QuanserInteractiveLabs()
        print(f"Connecting to QLabs at {self.config.qlabs_host}...")
        self.qlabs.open(self.config.qlabs_host)

        # Increase timeout to 10s for scene actors (default 5s)
        self.qlabs.set_wait_for_container_timeout(10.0)

        print("Connected to QLabs successfully")
        self.connected = True
        return True
            

    
    def spawn_qcar(self, model_wrapper=None) -> bool:
        """
        Spawn QCar2 in QLabs.

        Args:
            model_wrapper: SimlingoModelWrapper instance for HLC support

        Returns:
            True if spawn successful, False otherwise
        """

        # Destroy any existing actors
        print("Destroying existing actors...")
        self.qlabs.destroy_all_spawned_actors()

        # Validate that route has been loaded
        if self.config.qcar2_spawn_location is None or self.config.qcar2_spawn_rotation is None:
            print("\nERROR: No route loaded!")
            print("Please load a route using config.load_route(route_name) before spawning QCar2")
            print("Available routes are in the routes/ directory")
            return False

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

        print("QCar2 spawned successfully")

        # Initialize state
        self.current_location = np.array(self.config.qcar2_spawn_location, dtype=np.float32)
        self.current_rotation = np.array(self.config.qcar2_spawn_rotation, dtype=np.float32)

        # Initialize trajectory tracers if enabled
        if self.config.enable_trajectory_tracer:
            self._initialize_trajectory_tracer()

        if self.config.enable_planned_route_tracer:
            self._initialize_planned_route_tracer()

        # Initialize commentary window (with HLC support)
        self._initialize_commentary_widget(model_wrapper=model_wrapper)
        return True
    
    def get_camera_image(self) -> Optional[np.ndarray]:
        """Capture image from QCar2 camera, return RGB or None if failed."""
        success, image = self.qcar.get_image(camera=self.config.qcar2_camera)

        if not success or image is None:
            print("ERROR: Failed to get camera image")
            return None

        if not isinstance(image, np.ndarray) or image.size == 0:
            print(f"ERROR: Invalid image - type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')}")
            return None

        # Convert BGR to RGB
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"ERROR: Color conversion failed: {e}")
            return None

        return image
    
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

        # Send velocity and turn command
        _, location, rotation, front_hit, rear_hit = self.qcar.set_velocity_and_request_state(
            forward=forward_velocity,
            turn=turn_angle,
            headlights=False,
            leftTurnSignal=False,
                rightTurnSignal=False,
                brakeSignal=False,
                reverseSignal=False
            )
        
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

        if camera is None:
            camera = QLabsQCar2.CAMERA_TRAILING
        
        self.qcar.possess(camera=camera)
        
        return True
    
    def close(self):
        """Close QLabs connection and cleanup."""
        print("Closing QLabs connection...")
        self.qlabs.close()
        print("QLabs connection closed")

        self.connected = False
        self.qcar = None
        self.qlabs = None
        self.trajectory_tracer = None
        self.planned_route_tracer = None

    def _initialize_trajectory_tracer(self):
        """Initialize QLabs trajectory tracer (spline line)."""
        print("Initializing trajectory tracer...")
        self.trajectory_tracer = QLabsSplineLine(self.qlabs)

        # Spawn the spline line actor at origin
        # Configuration 1 = CURVE mode for smooth trajectory
        self.trajectory_tracer.spawn_id(
            actorNumber=100,  # Use actor number 100 for trajectory
            location=[0, 0, 0.01],  # Slightly above ground
            rotation=[0, 0, 0],
            configuration=1,  # CURVE configuration
            waitForConfirmation=True
        )

        # Initialize with starting position
        start_pos = self.config.qcar2_spawn_location
        self.trajectory_points = [[start_pos[0], start_pos[1], 0.01, self.config.trajectory_tracer_width]]
        print("Trajectory tracer initialized successfully")

    def update_trajectory_tracer(self, location: np.ndarray):
        """
        Update trajectory tracer with new vehicle position.

        Args:
            location: Current vehicle position [x, y, z]
        """
        if self.trajectory_tracer is None or not self.config.enable_trajectory_tracer:
            return

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

    def _initialize_planned_route_tracer(self):
        """Initialize QLabs planned route tracer (green spline line)."""
        print("Initializing planned route tracer...")
        self.planned_route_tracer = QLabsSplineLine(self.qlabs)

        # Spawn the spline line actor at origin
        # Configuration 1 = CURVE mode for smooth route visualization
        self.planned_route_tracer.spawn_id(
            actorNumber=101,  # Use actor number 101 for planned route
            location=[0, 0, 0.02],  # Slightly higher than actual trajectory
            rotation=[0, 0, 0],
            configuration=1,  # CURVE configuration
            waitForConfirmation=True
        )

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


    def _initialize_commentary_widget(self, model_wrapper=None):
        """Initialize commentary display window."""
        if CommentaryWindow is None:
            print("Warning: CommentaryWindow not available (inference package not imported)")
            self.commentary_widget = None
            return

        print("Initializing commentary window...")
        self.commentary_widget = CommentaryWindow(model_wrapper=model_wrapper)
        self.commentary_widget.start()
        print("Commentary window initialized successfully")

    def update_commentary(self, text: str):
        """
        Update the commentary window with new text.

        Args:
            text: Commentary text to display
        """
        if self.commentary_widget is not None and text:
            self.commentary_widget.update_commentary(text)

    def update_speed(self, speed):
        """
        Update the speed display in the commentary window.

        Args:
            speed: Current vehicle speed in m/s
        """
        if self.commentary_widget is not None:
            self.commentary_widget.update_speed(speed)

    def update_waypoints(self, route_waypoints, speed_waypoints):
        """
        Update the waypoint display in the commentary window.

        Args:
            route_waypoints: Route waypoints array [F, 2] in ego frame
            speed_waypoints: Speed waypoints array [F, 2] in ego frame
        """
        if self.commentary_widget is not None:
            self.commentary_widget.update_waypoints(route_waypoints, speed_waypoints)

