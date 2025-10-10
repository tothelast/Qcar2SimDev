# Custom Implementation for QCar2 Integration

## Part 2: What We Do Differently from Original SimLingo

### src/qcar2_interface.py
Handles connection to QLabs and QCar2 hardware, replacing SimLingo's CARLA interface. Manages vehicle spawning, camera capture from QCar2 CSI camera, and sends control commands to QCar2 actuators.
- **Uses:** `QLabsQCar2` (spawns vehicle, sends commands), `get_image_data()` (QCar2 camera → RGB image), `possess()` (enables control)
- **Replaces:** CARLA sensor interface and vehicle control API

### src/state_estimator.py
Estimates vehicle state (position, velocity, heading) from QCar2 odometry, replacing SimLingo's use of CARLA's perfect ground truth. Implements velocity estimation from position changes with moving average filtering.
- **Uses:** `StateEstimator.update()` (QCar2 location/rotation → filtered velocity), `get_velocity()` (→ current speed estimate)
- **Replaces:** CARLA's ground truth GPS, IMU, and speed sensors

### src/route_manager.py
Manages predefined route waypoints and converts between world and ego frames, replacing SimLingo's CARLA GlobalRoutePlanner. Handles target point selection based on lookahead distance and current position.
- **Uses:** `get_target_point_ego()` (current position/heading → ego-frame target points), `world_to_ego()` (coordinate transformation)
- **Replaces:** CARLA's GlobalRoutePlanner and route following logic

### src/camera_processor.py
Adapts camera preprocessing for QCar2 camera characteristics while maintaining SimLingo's InternVL2 pipeline. Adjusts bottom crop from 30% to 10% since QCar2 camera doesn't show vehicle hood.
- **Uses:** `process_image()` (QCar2 RGB image → preprocessed patches), JPEG compression (matches training data)
- **Replaces:** CARLA camera preprocessing (different crop ratio)

### src/control_converter.py
Converts SimLingo's steering/throttle outputs to QCar2 control format using kinematic bicycle model. Implements same PID controllers as SimLingo but adapts for QCar2's opposite steering convention.
- **Uses:** `control_pid()` (waypoints → steer/throttle/brake), `convert_to_qcar2_control()` (CARLA controls → QCar2 velocity/turn_angle)
- **Replaces:** Direct CARLA vehicle control (different control interface)

### src/commentary_window.py
Provides real-time GUI for displaying model commentary and accepting high-level commands. Not present in original SimLingo evaluation setup.
- **Uses:** `CommentaryWindow` (displays commentary, accepts HLC input), `update_commentary()` (shows model output)
- **Replaces:** Nothing (new feature for user interaction)

### src/config.py
Stores QCar2-specific configuration including spawn location, camera FOV (160° for QCar2 vs 110° for CARLA), and route waypoints for QLabs environment. Maintains same PID parameters as SimLingo.
- **Uses:** `SimlingoQCar2Config` (centralized configuration), `get_camera_intrinsics()` (→ camera matrix), `route_waypoints` (predefined path)
- **Replaces:** CARLA-specific configuration and route files

### src/main.py
Main control loop that orchestrates all components for QCar2 integration. Handles initialization, model inference, control conversion, and trajectory logging.
- **Uses:** `SimlingoQCar2Controller.run()` (main control loop), `step()` (single iteration: sense → predict → act)
- **Replaces:** SimLingo's CARLA leaderboard agent structure

### src/visualize_trajectory.py
Generates trajectory comparison plots showing predicted vs actual paths with detailed analysis. Used for debugging and performance evaluation.
- **Uses:** `visualize_trajectory()` (trajectory data → comparison plot), plots waypoints and actual path
- **Replaces:** Nothing (new debugging tool)

### src/visualize_route.py
Visualizes the predefined route waypoints in QLabs coordinate system. Helper tool for route design and verification.
- **Uses:** `visualize_route()` (waypoints → matplotlib plot)
- **Replaces:** Nothing (new route design tool)

## What We Do Differently

Original SimLingo runs in CARLA simulator with perfect ground truth sensors and direct vehicle control. Our integration adapts SimLingo to run on real QCar2 hardware in QLabs by implementing custom interfaces for hardware communication, state estimation from noisy sensors, route management without CARLA's planner, and control conversion to QCar2's API. We keep SimLingo's AI model and PID controllers unchanged but replace all simulator-specific code with QCar2/QLabs equivalents.

