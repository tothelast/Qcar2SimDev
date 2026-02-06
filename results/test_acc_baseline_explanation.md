## What `results/test_acc_baseline.py` does (simple overview)

This script runs a **baseline Adaptive Cruise Control (ACC)** demo in QLabs:

- The ego **QCar2 follows a predefined route** using a simple steering controller (pure pursuit).
- The car **drives at a constant speed** (`CRUISE_SPEED`) until it thinks there is an obstacle.
- It reads the **2D LiDAR** every control cycle. If LiDAR points indicate an obstacle **inside the lane**, the car **stops**.
- When it stops, it saves a single debug image (`lidar_stop.png`) showing the route, lane boundaries, the car, the detection zone, and the LiDAR points that triggered stopping.

---

## Key parameters (top of the file)

- `CRUISE_SPEED`: forward speed when not stopped.
- `CONTROL_HZ`: how often the loop runs (10 Hz by default).
- `LOOKAHEAD`: pure pursuit lookahead distance along the route.

Obstacle detection:
- `LIDAR_HALF_WIDTH`: **wide** left/right half-width of the LiDAR “detection box” in the car frame.
- `STOP_DISTANCE`: how far ahead the detection box extends.
- `MIN_DETECT_DIST`: ignores very close returns (often parts of the ego car).
- `MIN_POINTS`: minimum number of LiDAR points required to confirm an obstacle (helps reduce noise).

Lane filtering:
- `ROAD_LANE_HALF_WIDTH`: half-width of the lane used to decide whether a LiDAR point is “on the lane”.

---

## Route + lane boundary geometry

- `load_route(route_name)` loads a JSON route from `config/routes/<route>.json`.
- `calculate_lane_boundaries(waypoints, lane_half_width)` computes two dashed polylines offset from the centerline.
  - It estimates a tangent direction at each waypoint.
  - Then it offsets left/right by `lane_half_width` using a perpendicular vector.

---

## How LiDAR obstacle detection works

### 1) Read LiDAR
`get_obstacle_distance(...)` calls:
- `qcar.get_lidar(samplePoints=400)` → returns arrays of `angles` and `distances`.

### 2) Convert polar to the vehicle frame
The script converts (angle, distance) to (x, y) in the **vehicle frame**:
- `y = cos(angle) * distance` → forward
- `x = sin(angle) * distance` → left/right

### 3) Pick points inside a forward “detection box”
A point is considered “in the detection zone” if:
- `MIN_DETECT_DIST < y < STOP_DISTANCE` and `|x| < LIDAR_HALF_WIDTH`

### 4) Transform those points to world coordinates
Using the ego pose (`position`, `heading`), it rotates/translates each detected (x, y) point into world coordinates.

### 5) Keep only points that lie inside the lane around the route
For each world point, `is_point_in_lane(...)`:
- Finds the **nearest route waypoint**.
- Computes the **perpendicular distance** from the point to the route centerline at that waypoint.
- Returns **True** if that distance ≤ `ROAD_LANE_HALF_WIDTH`.

Only LiDAR points that pass this lane test can cause stopping.

### 6) Output
If there are at least `MIN_POINTS` in-lane points, the function returns:
- The closest forward distance (`min(y)` in vehicle frame)
- The in-lane detected points (world coordinates) for plotting

---

## Steering (pure pursuit)

`compute_steering(pos, heading, waypoints)`:
- Finds the closest waypoint to the car.
- Walks forward along the route until it accumulates `LOOKAHEAD` meters.
- Computes the angle from the car to that target point in the car frame.
- Converts that to a normalized steering command (clipped).

The main loop converts this to an actual turn command:
- `turn = -steering * MAX_STEERING`

---

## Main loop (what happens each cycle)

Inside `main()`:

1. Load the scene + route.
2. Connect to QLabs, spawn the ego car + scene actors.
3. Loop at `CONTROL_HZ`:
   - Read LiDAR and run `get_obstacle_distance(...)`.
   - If an obstacle is detected and we haven’t stopped yet:
     - Set `stopped = True`, save `lidar_stop.png`.
   - If route end is reached: exit.
   - If not stopped: drive at `CRUISE_SPEED`; otherwise speed is 0.
   - Send commands and request the updated state.

---

## What the saved image means (`lidar_stop.png`)

- Blue solid line: route centerline (waypoints)
- Blue dashed lines: lane boundaries (± `ROAD_LANE_HALF_WIDTH`)
- Green marker/arrow: ego car pose
- Orange rectangle: LiDAR detection zone (based on `LIDAR_HALF_WIDTH`, `MIN_DETECT_DIST`, `STOP_DISTANCE`)
- Red X markers: LiDAR points (world frame) that were both:
  1) inside the orange box in the car frame, and
  2) inside the lane around the route

