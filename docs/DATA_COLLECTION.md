## Data Collection Requirements for SimLingo Fine-Tuning in QLabs

SimLingo uses **route-based episodes** with predefined start/end points. Each episode is saved as a separate directory with synchronized camera images and vehicle measurements.

**CRITICAL:** This document reflects the **actual SimLingo model requirements** validated against the CARLA training data format and model architecture.

---

## Data Organization

### Directory Structure
```
database/qcar2_simlingo/
├── routes_training/
│   ├── QLabs_Rep0_0/              # Route 0, Repetition 0
│   │   ├── rgb/                   # Camera images (JPEG)
│   │   │   ├── 0000.jpg           # Frame 0 (1024×512)
│   │   │   ├── 0001.jpg           # Frame 1
│   │   │   └── ...
│   │   ├── measurements/          # Vehicle state (gzipped JSON)
│   │   │   ├── 0000.json.gz
│   │   │   ├── 0001.json.gz
│   │   │   └── ...
│   │   └── results.json.gz        # Route completion metrics
│   ├── QLabs_Rep0_1/              # Route 1, Repetition 0
│   └── ...
└── routes_validation/
```

### Episode Requirements
- **Sampling rate:** **4 Hz (save every 0.25 seconds)** - Matches SimLingo CARLA training data
- **Control frequency:** 20 Hz (QCar2 runs at 20 FPS, save every 5th frame)
- **Minimum length:** 50 frames (12.5 seconds at 4 Hz)
- **Recommended length:** 100-150 frames (25-37.5 seconds at 4 Hz)
- **Route completion:** Aim for 100% (no crashes/infractions)

---

## Per-Frame Data Requirements

### 1. Camera Image (`rgb/XXXX.jpg`)
- **Source:** QCar2 CSI front camera (CAMERA_CSI_FRONT)
- **Native resolution:** 820×410 px (QCar2 CSI camera specification)
- **Saved resolution:** 1024×512 px (resize before saving to match CARLA)
- **Format:** JPEG (apply JPEG compression to match training data)
- **Naming:** 4-digit zero-padded frame number (e.g., `0000.jpg`, `0001.jpg`)

### 2. Measurement File (`measurements/XXXX.json.gz`)

**REQUIRED fields for model inference and training:**
```json
{
  "ego_matrix": [[...], ...],         // 4×4 transformation matrix (CRITICAL - see below)
  "speed": float,                     // Current speed (m/s)
  "target_point": [x, y],             // Current target point (ego frame)
  "target_point_next": [x, y],        // Next target point (ego frame) - REQUIRED
  "route": [[x, y], ...],             // 40+ route waypoints (will be resampled to 20)
  "route_original": [[x, y], ...],    // Original route (same as route if no modifications)
  "command": int,                     // High-level command (1-6, use 4=STRAIGHT if unknown)
  "next_command": int,                // Next HLC (same as command if no change)
  "augmentation_translation": 0.0,    // Camera shift augmentation (set to 0.0)
  "augmentation_rotation": 0.0        // Camera rotation augmentation (set to 0.0)
}
```

**OPTIONAL fields (recommended for debugging/visualization):**
```json
{
  "pos_global": [x, y],               // World position (meters) - for debugging
  "theta": float                      // Heading angle (radians) - for debugging
}
```

**CRITICAL:** `ego_matrix` is a 4×4 homogeneous transformation matrix representing the vehicle's pose in world coordinates. It is used to compute future waypoints by transforming future vehicle positions into the current ego frame.

---

## Ground-Truth Waypoint Computation

### Speed Waypoints (10 points, 0.25s spacing)
**CORRECTED:** Model outputs **10 waypoints**, NOT 11!

- **Temporal spacing:** 0.25 seconds (4 Hz sampling rate)
- **Prediction horizon:** 10 waypoints × 0.25s = **2.5 seconds**
- **Computed from:** Future vehicle positions (next 10 frames after current)
- **Algorithm:**
  1. Load current frame + 11 future frames (12 total measurements)
  2. Extract `ego_matrix` from each of the 12 measurements
  3. Slice positions `[1:-1]` to get 10 intermediate positions (skip first and last)
  4. Transform these 10 future positions to current ego frame using:
     ```python
     ego_point_2d = R^T @ (future_position - current_translation)
     ```
  5. Result: `[[x₀, y₀], [x₁, y₁], ..., [x₉, y₉]]` (10 waypoints in ego frame)

### Path Waypoints (20 points, 1m spacing)
- **Extracted from:** `route` field in current measurement
- **Resampling:** Use equal spacing algorithm to resample route to exactly 20 points with 1m spacing
- **Coordinate frame:** Ego vehicle frame (already transformed)
- **Result:** `[[x₀, y₀], [x₁, y₁], ..., [x₁₉, y₁₉]]` (20 waypoints in ego frame)

**Coordinate Frame:** All waypoints in ego vehicle frame (X=forward, Y=left, origin at vehicle center, BEV projection with Z-coordinate dropped)

---

## Camera Calibration

### Intrinsics (3×3 matrix)
- **FOV:** **160°** (QCar2 CSI front camera specification)
- **Focal length:** `f = width / (2 * tan(fov/2))` = 1024 / (2 * tan(80°)) ≈ **90.28 pixels**
- **Principal point:** `[cx, cy] = [512, 256]` (image center)
- **Intrinsics matrix:**
  ```
  [[90.28,   0.0, 512.0],
   [ 0.0 , 90.28, 256.0],
   [ 0.0 ,   0.0,   1.0]]
  ```

### Extrinsics (4×4 matrix)
- **Camera position:** `[+1.83, 0.0, +1.10]` meters (QCar2 CSI front camera, from Quanser docs)
  - X: +1.83m forward from vehicle center
  - Y: 0.0m (centered laterally)
  - Z: +1.10m above ground
- **Camera rotation:** Identity (no rotation, aligned with vehicle frame)
- **Extrinsics matrix:**
  ```
  [[1.0, 0.0, 0.0, +1.83],
   [0.0, 1.0, 0.0,  0.0 ],
   [0.0, 0.0, 1.0, +1.10],
   [0.0, 0.0, 0.0,  1.0 ]]
  ```

---

## Language Prompt & Answer

### Prompt Template (Target Point Mode)
```
"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
```
- `<TARGET_POINT>` tokens are replaced with learned embeddings during model forward pass
- Two target points: current and next (for smooth navigation)

### Prompt Template (Command Mode - Alternative)
```
"Current speed: {speed:.2f} m/s. High-level command: {command_text}. What should the ego do next?"
```
- Command text: "Turn left", "Turn right", "Go straight", "Follow lane", "Change lane left", "Change lane right"

### Answer
- Default: `"Waypoints:"`
- No numeric coordinates in text
- Used as separator before waypoint token generation

---

## Route Definition

### Route-Based Collection
- Define 5-10 routes using QLabs Cityscape nodes (e.g., Node 13 → Node 18)
- Collect 30-50 demonstrations per route
- Vary driving style (speed, smoothness, scenarios)
- **QCar2 coordinate scaling:** QLabs coordinates are 10× physical QCar2 scale

### Route Completion Metrics (`results.json.gz`)
```json
{
  "score_composed": 100.0,    // Overall score (0-100)
  "score_route": 100.0,       // Route completion
  "num_infractions": 0
}
```

**Training filter:** Only use routes with `score_composed >= 100.0`

---

## Fields NOT Required for QLabs (CARLA-Specific)

The following fields from CARLA are **NOT needed** for QLabs data collection:

**Traffic & Hazard Detection (CARLA simulator only):**
- `vehicle_hazard`, `vehicle_affecting_id`
- `light_hazard` (traffic lights)
- `walker_hazard`, `walker_affecting_id`, `walker_close`, `walker_close_id` (pedestrians)
- `stop_sign_hazard`, `stop_sign_close`
- `junction` (intersection detection)

**Speed Reduction Logic (CARLA-specific):**
- `speed_reduced_by_obj_type`, `speed_reduced_by_obj_id`, `speed_reduced_by_obj_distance`

**Control Inputs (not used by model):**
- `steer`, `throttle`, `brake`, `control_brake`
- `target_speed`, `speed_limit`
- `angle` (steering angle)

**Route Modification Flags:**
- `aim_wp` (intermediate waypoint)
- `changed_route`

---

**Summary:**
SimLingo expects route-based episodes with synchronized images and measurements at **4 Hz (0.25s intervals)**. Each measurement must include `ego_matrix` for waypoint computation, plus `target_point`, `target_point_next`, `route`, and HLC fields. The model outputs **10 speed waypoints** (2.5s horizon) and **20 path waypoints** (1m spacing). Collect 30-50 demonstrations per route with 100% completion rate.

