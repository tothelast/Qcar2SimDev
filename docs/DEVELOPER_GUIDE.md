# Developer Guide - Simlingo-QCar2 Integration

Quick reference guide for navigating and developing this project.

---

## Project Structure

```
Qcar2SimDev/
├── src/                           # Integration code (our implementation)
│   ├── main.py                    # Main entry point
│   ├── config.py                  # Configuration (SimLingo parameters)
│   ├── simlingo_model.py          # SimLingo model wrapper
│   ├── qcar2_interface.py         # QCar2 vehicle interface
│   ├── camera_processor.py        # Camera image processing
│   ├── control_converter.py       # Control conversion (PID + Linear Regression)
│   ├── route_manager.py           # Route waypoint management
│   ├── state_estimator.py         # Vehicle state estimation
│   ├── visualize_trajectory.py    # Trajectory visualization tool
│   ├── visualize_route.py         # Route preview tool
│   └── generate_route_coordinates.py  # Route generation helper
├── python/                        # Quanser HAL library (DO NOT MODIFY)
├── models/                        # Model checkpoints (LoRA weights)
├── pretrained/                    # Hugging Face model cache (~1.8GB, DO NOT DELETE)
├── debug_output/                  # Test results and logs
├── SIMLINGO_PAPER_VERIFICATION.md # Implementation verification report
├── IMPLEMENTATION_CHANGES.md      # Recent implementation fixes
├── POST_FIX_ANALYSIS.md          # Latest test analysis
├── QCAR2_PID_TUNING.md           # PID tuning documentation
└── README.md                      # Project overview
```

**Note:** The `simlingo/` directory is NOT used. We use the official SimLingo model from Hugging Face directly.

---

## Core Integration Files (`src/`)

### Main Entry Point
- **`main.py`** - Main integration script
  - Initializes all components
  - Runs the control loop (20 Hz)
  - Logs trajectory data
  - Handles shutdown

### Configuration
- **`config.py`** - All configuration settings (SimlingoQCar2Config class)
  - **Model parameters:** InternVL2-1B with LoRA (r=32, alpha=64)
  - **Camera settings:** Resolution (1024×512), FOV (110°), crop (30% bottom)
  - **Route waypoints:** Full route from spawn to destination
  - **Spawn location:** QCar2 initial position and heading
  - **Control parameters:**
    - Lateral PID: `turn_kp=3.25`, `turn_ki=1.0`, `turn_kd=1.0` (Official SimLingo)
    - Longitudinal: Linear Regression (7 coefficients)
    - Throttle limits: `clip_throttle=1.0`, `max_throttle=1.0`
  - **Kinematic bicycle model:** Wheelbase, steering gain, acceleration parameters

### Model & Interface
- **`simlingo_model.py`** - SimLingo model wrapper
  - **Model:** InternVL2-1B from Hugging Face (`OpenGVLab/InternVL2-1B`)
  - **LoRA weights:** Loaded from `models/simlingo/checkpoints/epoch=013.ckpt`
  - **Input processing:**
    - Splits 1024×512 image into 2 patches (512×512 each)
    - Applies ImageNet normalization
    - Adds speed scalar input
  - **Output:**
    - 20 route waypoints (ego frame, cumulative sum encoding)
    - 10 speed waypoints (ego frame, cumulative sum encoding)
  - **Inference:** ~0.5s per frame on GPU

- **`qcar2_interface.py`** - QCar2 vehicle interface
  - Connects to QLabs simulator
  - Spawns QCar2 at configured location
  - **Control interface:** Sends velocity and turn_angle commands
  - **State reading:** Position, rotation, velocity from QLabs
  - **Collision detection:** Monitors vehicle collisions
  - **Camera:** CSI camera (1024×512 resolution, 110° FOV)

### Processing & Control
- **`camera_processor.py`** - Camera image processing
  - Captures images from QCar2 CSI camera (1024×512)
  - Crops bottom 30% (removes hood/dashboard)
  - Splits into 2 patches (512×512 each) for InternVL2
  - Applies ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Returns preprocessed tensor for model input

- **`control_converter.py`** - Control conversion (Official SimLingo implementation)
  - **Two controller classes:**
    1. **`LateralPIDController`** - Steering control
       - **Official SimLingo PID gains:** Kp=3.25, Ki=1.0, Kd=1.0
       - **Discrete lookahead:** 2.25m (slow), 3.0m (medium), 7.0m (fast)
       - Speed thresholds: 5.5 m/s and 15.0 m/s
       - Outputs steering angle [-1, 1]
    2. **`LongitudinalLinearRegressionController`** - Speed control
       - Linear regression with 7 coefficients (official SimLingo)
       - Max acceleration: 1.89 m/tick
       - Max deceleration: -4.82 m/tick
       - Outputs throttle [0, 1] and brake (bool)
  - **`ControlConverter`** class:
    - Converts model waypoint predictions to QCar2 commands
    - Calculates desired speed from speed waypoints
    - Interpolates route waypoints for smooth tracking
    - Applies kinematic bicycle model for velocity/turn_angle conversion

- **`route_manager.py`** - Route management
  - Manages route waypoints (world frame)
  - Tracks current waypoint index based on vehicle position
  - Converts waypoints between world frame and ego frame
  - Checks route completion (distance to final waypoint)
  - Provides waypoints for visualization

- **`state_estimator.py`** - State estimation
  - Estimates vehicle velocity from position changes
  - Filters noisy sensor readings
  - Provides smooth state estimates for control

### Analysis & Utility Tools
- **`visualize_trajectory.py`** - Trajectory visualization and analysis
  - Loads trajectory logs from `debug_output/trajectory_log_*.json`
  - Creates comprehensive 6-panel visualization:
    * Full route view (planned vs actual trajectory)
    * Zoomed start area
    * Lateral deviation over time
    * Speed profile over time
    * Steering profile over time
    * Statistics summary (success rate, collisions, metrics)
  - Saves PNG to `debug_output/trajectory_comparison_*.png`
  - Calculates success rate (% of points within 1.0m of route)

- **`visualize_route.py`** - Route preview tool
  - Visualizes planned route before running
  - Shows waypoints and spawn location
  - Useful for route validation

- **`generate_route_coordinates.py`** - Route generation helper
  - Generates route waypoints from SDCSRoadMap node sequences
  - Handles 10x scaling for QLabs coordinates
  - Outputs formatted waypoint lists for `config.py`

---

## External Libraries

### Quanser HAL (`python/`)
**DO NOT MODIFY** - Original Quanser library

Key files to reference:
- **`python/hal/products/mats.py`** - SDCSRoadMap class
  - Defines road network for QLabs Cityscape Lite
  - 24 nodes with positions and headings
  - `generate_path(nodeSequence)` - Creates smooth waypoint paths
  - **Important:** Coordinates need 10x scaling for QLabs (SDCSRoadMap × 10 = QLabs)

- **`python/qvl/qlabs.py`** - QLabs connection
- **`python/qvl/qcar.py`** - QCar2 vehicle class

---

## Model Files

### Model Checkpoint
**Location:** `models/simlingo/checkpoints/epoch=013.ckpt`

**Model Details:**
- **Base Model:** InternVL2-1B from Hugging Face (`OpenGVLab/InternVL2-1B`)
- **Fine-tuning:** LoRA (r=32, alpha=64, dropout=0.1)
- **Training Data:** CARLA simulator (official SimLingo dataset)
- **Input:**
  - 2 image patches (512×512 each, from 1024×512 camera)
  - Current speed (scalar)
- **Output:**
  - 20 route waypoints (ego frame, cumulative sum encoding)
  - 10 speed waypoints (ego frame, cumulative sum encoding)
- **Inference Time:** ~0.5s per frame on GPU

### Pretrained Model Cache
**Location:** `pretrained/InternVL2-1B/`

**Purpose:**
- Hugging Face cache for InternVL2-1B base model
- Size: ~1.8 GB
- **DO NOT DELETE** - Required for model loading
- Avoids re-downloading from Hugging Face (saves time and bandwidth)
- Essential for offline use

**What's Cached:**
- InternVL2-1B vision encoder weights
- InternVL2-1B language model weights
- Tokenizer files
- Model configuration files

---

## Coordinate Systems

### World Frame (QLabs)
- **X:** East (+) / West (-)
- **Y:** North (+) / South (-)
- **Z:** Up (+) / Down (-)
- **Heading:** 0° = East, 90° = North, 180° = West, -90° = South

### Ego Frame (Vehicle-Centric)
- **X:** Forward (direction vehicle is facing)
- **Y:** Left (perpendicular to forward direction)
- **Origin:** Vehicle position
- **Note:** Model predictions are in ego frame (cumulative sum encoding)

### Conversion (World ↔ Ego)
```python
# World → Ego (from route_manager.py)
rotation_matrix = np.array([
    [np.cos(heading), np.sin(heading)],
    [-np.sin(heading), np.cos(heading)]
])
ego_point = rotation_matrix.T @ (world_point - vehicle_pos)

# Ego → World
world_point = rotation_matrix @ ego_point + vehicle_pos
```

**Important:** Model outputs waypoints in ego frame with cumulative sum encoding. These must be converted to world frame for visualization and waypoint tracking.

---

## Route Configuration

### Current Route
The current route is defined in `config.py` → `SimlingoQCar2Config.route_waypoints`

**Route Details:**
- **Total waypoints:** 36
- **Approximate length:** ~90 meters
- **Includes:** Straight sections, curves, and a roundabout
- **Spawn location:** [2.686, 18.498, 0.005]
- **Spawn heading:** 90° (facing North)
- **Destination:** [-19.841, 29.760, 0.0]

### Modifying Routes

**Option 1: Use SDCSRoadMap (Recommended)**
1. Open `python/hal/products/mats.py` to see available nodes (0-23)
2. Choose a node sequence (e.g., `[0, 1, 2, 3]`)
3. Use `generate_route_coordinates.py` to generate waypoints:
   ```python
   from python.hal.products.mats import SDCSRoadMap
   roadmap = SDCSRoadMap()
   path = roadmap.generate_path([0, 1, 2, 3])
   waypoints_scaled = path * 10.0  # Scale for QLabs
   ```
4. Copy the generated waypoints to `config.py` → `self.route_waypoints`

**Option 2: Manual Waypoints**
1. Define waypoints as `[x, y, z]` coordinates in QLabs world frame
2. Ensure waypoints are spaced appropriately (~2-5 meters apart)
3. Update `config.py` → `self.route_waypoints`
4. Update spawn location to match route start

**Important:** Always use `visualize_route.py` to preview the route before running!

---

## Testing & Debugging

### Running Tests
```bash
# Run main integration
python src/main.py

# Visualize latest trajectory
python src/visualize_trajectory.py

# Visualize specific trajectory
python src/visualize_trajectory.py --log debug_output/trajectory_log_YYYYMMDD_HHMMSS.json
```

### Debug Output
- **Trajectory logs:** `debug_output/trajectory_log_*.json`
  - Position, heading, speed, steering at each step
  - Collision events
  - Target waypoints
  - Metadata (route, config)

- **Visualizations:** `debug_output/trajectory_comparison_*.png`
  - 6-panel analysis of trajectory vs planned route

- **Camera images:** `debug_output/camera_*.jpg`
  - Raw and processed camera images (when debug enabled)

### Common Issues & Solutions

**Moderate lateral deviation (mean ~1.5m):**
- **Current State:** Success rate ~30% (within 1.0m threshold)
- **Cause:** Domain gap between CARLA training and QLabs QCar2
- **Status:** This is expected with official SimLingo parameters
- **Solution:**
  - Fine-tune model on QLabs data for better performance
  - Consider hybrid control (model + MPC/LQR)
  - Note: Custom PID tuning performed worse than official parameters

**Vehicle speed (~0.7 m/s average):**
- **Current State:** Mean 0.7 m/s, max 1.27 m/s
- **Cause:** QCar2 has lower acceleration than CARLA vehicles (1/10 scale)
- **Status:** This is a hardware limitation
- **Solution:**
  - Linear regression controller is already at max throttle (0.994 mean)
  - Fine-tuning model on QLabs data may help
  - Consider speed scaling in waypoint predictions

**Collisions (~4% of steps):**
- **Current State:** 19 collisions in 462 steps
- **Cause:** Model predictions occasionally lead to wall contact
- **Status:** Acceptable for current implementation
- **Solution:**
  - Add collision avoidance logic (brake when collision detected)
  - Fine-tune model with QLabs collision data
  - Add safety margins to route waypoints

**Model inference slow (~0.5s per frame):**
- **Expected:** This is normal for InternVL2-1B on GPU
- **Impact:** Control loop runs at ~2 Hz instead of 20 Hz
- **Solution:**
  - Use GPU if available (much faster than CPU)
  - Consider model quantization for faster inference
  - Async processing (run model in separate thread)

**Important Note on PID Tuning:**
- Custom QCar2-tuned parameters (Ki=0.75, Kd=1.20, continuous lookahead) performed **worse** than official SimLingo parameters
- **Current best:** Official SimLingo parameters (Ki=1.0, Kd=1.0, discrete lookahead)
- Success rate: 30% (official) vs. 15% (custom tuning)

---

## Key Parameters (`config.py`)

All parameters are defined in the `SimlingoQCar2Config` class.

### Model Parameters
- `model_name`: `"OpenGVLab/InternVL2-1B"` (Hugging Face model ID)
- `model_checkpoint`: `"models/simlingo/checkpoints/epoch=013.ckpt"` (LoRA weights)
- `device`: `"cuda"` or `"cpu"` (GPU recommended)
- `lora_r`: `32` (LoRA rank)
- `lora_alpha`: `64` (LoRA alpha)
- `lora_dropout`: `0.1` (LoRA dropout)

### Camera Parameters
- `camera_resolution`: `[1024, 512]` (width × height)
- `camera_fov`: `110` (degrees, horizontal field of view)
- `crop_bottom_percent`: `30` (removes hood/dashboard)
- `imagenet_mean`: `[0.485, 0.456, 0.406]` (normalization)
- `imagenet_std`: `[0.229, 0.224, 0.225]` (normalization)

### Control Parameters (Official SimLingo)
**Lateral PID (in `control_converter.py`):**
- `k_p`: `3.25` (proportional gain - from config.turn_kp)
- `k_i`: `1.0` (integral gain - from config.turn_ki)
- `k_d`: `1.0` (derivative gain - from config.turn_kd)
- `turn_n`: `20` (buffer size for integral/derivative)
- **Lookahead (discrete):**
  - `2.25m` when speed < 5.5 m/s
  - `3.0m` when 5.5 m/s ≤ speed < 15.0 m/s
  - `7.0m` when speed ≥ 15.0 m/s

**Longitudinal Linear Regression (in `control_converter.py`):**
- 7 regression coefficients (hardcoded from official SimLingo)
- `max_acceleration`: `1.89` m/tick
- `max_deceleration`: `-4.82` m/tick
- `minimum_target_speed`: `0.278` m/s

**Throttle Limits:**
- `clip_throttle`: `1.0` (maximum throttle)
- `max_throttle`: `1.0` (absolute maximum)

### Route Parameters
- `route_waypoints`: List of `[x, y, z]` waypoints (world frame)
- `qcar2_spawn_location`: `[x, y, z]` spawn position
- `qcar2_spawn_rotation`: `[roll, pitch, yaw]` spawn orientation

### Waypoint Parameters
- `num_route_waypoints`: `20` (model output)
- `num_speed_waypoints`: `10` (model output)
- `wp_dilation`: `1` (waypoint spacing multiplier)
- `carla_fps`: `20` (CARLA training framerate)

---

## Development Workflow

### Standard Development Cycle
1. **Make changes** to integration code in `src/`
2. **Test** with `python src/main.py`
3. **Visualize** results with `python src/visualize_trajectory.py`
4. **Analyze** metrics (success rate, lateral deviation, speed, collisions)
5. **Iterate** based on analysis
6. **Document** findings in analysis reports

### Tuning PID Parameters (Not Recommended)
**Note:** Custom PID tuning has been tested and performed worse than official SimLingo parameters.

If you still want to experiment:
1. **Identify issue** from trajectory visualization (oscillations, high deviation, etc.)
2. **Adjust parameters** in `src/control_converter.py`:
   - Lateral PID: `k_p`, `k_i`, `k_d` in `LateralPIDController.__init__`
3. **Test** with same route
4. **Compare** before/after metrics (success rate, lateral deviation, collisions)
5. **Revert if worse** - official parameters are currently best
6. **Document** changes and results

**Previous Tuning Attempt:**
- Custom: Ki=0.75, Kd=1.20, continuous lookahead → Success rate: 15%
- Official: Ki=1.0, Kd=1.0, discrete lookahead → Success rate: 30%
- **Conclusion:** Official parameters are better for QCar2

### Creating New Routes
1. **Choose nodes** from SDCSRoadMap (see `python/hal/products/mats.py`)
2. **Generate waypoints** using `src/generate_route_coordinates.py`
3. **Preview route** with `python src/visualize_route.py`
4. **Update config** in `src/config.py` → `route_waypoints` and spawn location
5. **Test** and iterate

---

## Important Notes

### DO NOT MODIFY
- **`python/`** - Quanser HAL library (original)
- **`pretrained/`** - Hugging Face model cache (DO NOT DELETE)
- **`models/`** - Model checkpoint files

### Implementation Details

**Control Architecture:**
- Uses official SimLingo control architecture (verified against paper)
- Lateral: Simple PID controller (official SimLingo parameters: Kp=3.25, Ki=1.0, Kd=1.0)
- Longitudinal: Linear Regression controller (official SimLingo default)
- See `SIMLINGO_PAPER_VERIFICATION.md` for detailed verification
- **Note:** Custom PID tuning performed worse than official parameters

**Coordinate Systems:**
- Model outputs waypoints in **ego frame** (vehicle-centric)
- Route waypoints stored in **world frame** (QLabs global)
- Conversion handled by `route_manager.py`

**Image Preprocessing:**
- Must match CARLA training: crop bottom 30%, split into 2 patches
- ImageNet normalization applied
- Camera resolution: 1024×512 (matches training)

**SDCSRoadMap Scaling:**
- SDCSRoadMap coordinates × 10 = QLabs coordinates
- Always scale when using `generate_path()`

**Domain Gap:**
- Model trained on CARLA (full-size vehicles, different dynamics)
- Running on QLabs QCar2 (1/10 scale, different acceleration)
- Using official SimLingo parameters (best performance observed)

**Current Performance (Official SimLingo Parameters):**
- Model inference: ~0.5s per frame on GPU
- Control loop: ~2 Hz (limited by model inference)
- Average speed: 0.7 m/s, max speed: 1.27 m/s
- Success rate: ~30% (within 1.0m of route)
- Mean lateral deviation: 1.5m
- Collision rate: ~4% of steps
- Route completion: Yes (reaches destination)

---

## Quick Reference

### Common Tasks

| Task | File | Location |
|------|------|----------|
| **Change route** | `config.py` | `SimlingoQCar2Config.route_waypoints` |
| **Change spawn location** | `config.py` | `SimlingoQCar2Config.qcar2_spawn_location` |
| **Tune lateral PID** | `control_converter.py` | `LateralPIDController.__init__` (k_p, k_i, k_d) - Not recommended |
| **Adjust lookahead** | `control_converter.py` | `LateralPIDController.__init__` (discrete thresholds) |
| **Modify throttle limits** | `config.py` | `SimlingoQCar2Config.clip_throttle` |
| **Generate new route** | `generate_route_coordinates.py` | Use SDCSRoadMap |
| **Preview route** | `visualize_route.py` | Run before testing |
| **Visualize trajectory** | `visualize_trajectory.py` | Run after testing |
| **Analyze results** | `debug_output/` | Check trajectory logs and visualizations |

### Key Files for Tuning

| Component | File | What to Tune |
|-----------|------|--------------|
| **Lateral Control** | `control_converter.py` | `LateralPIDController` (k_p, k_i, k_d, lookahead) |
| **Longitudinal Control** | `control_converter.py` | `LongitudinalLinearRegressionController` (coefficients) |
| **Route** | `config.py` | `route_waypoints`, spawn location |
| **Camera** | `config.py` | Resolution, FOV, crop percentage |
| **Model** | `config.py` | Model path, LoRA parameters |

### Analysis Documents

| Document | Purpose |
|----------|---------|
| `SIMLINGO_PAPER_VERIFICATION.md` | Verification against official SimLingo |
| `IMPLEMENTATION_CHANGES.md` | Recent implementation fixes |
| `POST_FIX_ANALYSIS.md` | Latest test run analysis |
| `QCAR2_PID_TUNING.md` | PID tuning documentation |
| `AUTONOMOUS_VEHICLE_ANALYSIS.md` | Comprehensive system analysis |

---

## References

- **Official SimLingo:** https://github.com/RenzKa/simlingo
- **SimLingo Paper:** https://arxiv.org/abs/2503.09594
- **InternVL2 Model:** https://huggingface.co/OpenGVLab/InternVL2-1B
- **Quanser QLabs:** https://www.quanser.com/products/qlabs/

---

**Last Updated:** October 8, 2025

