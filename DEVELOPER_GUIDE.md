# Developer Guide - Simlingo-QCar2 Integration

Quick reference guide for navigating and developing this project.

---

## Project Structure

```
Qcar2SimDev/
├── src/                    # Integration code (our code)
├── python/                 # Quanser HAL library (original, don't modify)
├── simlingo/              # Simlingo model code (original, don't modify)
├── models/                # Model checkpoints
├── debug_output/          # Test results and logs
└── README.md              # Project overview
```

---

## Core Integration Files (`src/`)

### Main Entry Point
- **`main.py`** - Main integration script
  - Initializes all components
  - Runs the control loop (20 Hz)
  - Logs trajectory data
  - Handles shutdown

### Configuration
- **`config.py`** - All configuration settings
  - Model paths and parameters
  - QCar2 spawn location and settings
  - Route waypoints (based on SDCSRoadMap)
  - Control parameters (PID gains, lookahead distance)
  - Camera settings

### Model & Interface
- **`simlingo_model.py`** - Simlingo model wrapper
  - Loads InternVL2-1B model with LoRA
  - Handles image preprocessing (cropping, patching)
  - Runs model inference
  - Returns waypoint predictions

- **`qcar2_interface.py`** - QCar2 vehicle interface
  - Connects to QLabs
  - Spawns vehicle
  - Sends control commands
  - Reads vehicle state (position, velocity, rotation)
  - Detects collisions

### Processing & Control
- **`camera_processor.py`** - Camera image processing
  - Captures images from QCar2 CSI camera
  - Crops bottom 30% (remove hood/dashboard)
  - Splits into patches for InternVL2
  - Applies ImageNet normalization

- **`control_converter.py`** - Control conversion
  - Converts Simlingo predictions to QCar2 commands
  - PID speed controller
  - Pure pursuit steering controller
  - Handles coordinate frame conversions

- **`route_manager.py`** - Route management
  - Manages route waypoints
  - Tracks current waypoint index
  - Finds target waypoint based on lookahead distance
  - Converts waypoints between world/ego frames
  - Checks route completion

- **`state_estimator.py`** - State estimation
  - Estimates vehicle state from QCar2 sensors
  - Calculates velocity from position changes
  - Provides filtered state estimates

### Analysis Tools
- **`visualize_trajectory.py`** - Trajectory visualization
  - Loads trajectory logs (JSON)
  - Creates 6-panel visualization:
    * Full route view (planned vs actual)
    * Zoomed start area
    * Lateral deviation over time
    * Speed profile
    * Steering profile
    * Statistics summary
  - Saves PNG visualizations

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

### Simlingo Model (`simlingo/`)
**DO NOT MODIFY** - Original Simlingo repository

Key files to reference:
- **`simlingo/simlingo_training/models/driving.py`** - Main model class
- **`simlingo/simlingo_training/models/encoder/internvl2_model.py`** - Vision encoder
- **`simlingo/simlingo_training/utils/custom_types.py`** - Data types
- **`simlingo/simlingo_training/utils/image_utils.py`** - Image preprocessing

---

## Model Checkpoint

**Location:** `models/simlingo/checkpoints/epoch=013.ckpt`

**Model Details:**
- Base: InternVL2-1B (vision-language model)
- Fine-tuning: LoRA (alpha=64, r=32, dropout=0.1)
- Training: CARLA simulator data
- Input: 2 image patches (448×448 each) + speed
- Output: 20 route waypoints + 10 speed waypoints

---

## Coordinate Systems

### World Frame (QLabs)
- X: East (+) / West (-)
- Y: North (+) / South (-)
- Z: Up (+) / Down (-)
- Heading: 0° = East, 90° = North, 180° = West, -90° = South

### Ego Frame (Vehicle)
- X: Forward
- Y: Left (opposite of world Y!)
- Origin: Vehicle position

### Conversion
```python
# World → Ego (from route_manager.py)
rotation_matrix = np.array([
    [np.cos(heading), np.sin(heading)],
    [-np.sin(heading), np.cos(heading)]
])
ego_point = rotation_matrix.T @ (world_point - vehicle_pos)
```

---

## Route Configuration

### Current Route (Simple Test Route)
- **Type:** Straight-line test route for visual verification
- **Start:** [0, -1.3] heading 90° (facing +Y/North)
- **End:** [0, 40.0]
- **Direction:** Straight north along X=0
- **Total length:** 41.3 meters
- **Total waypoints:** 22 (spaced every 2 meters)
- **Purpose:** Easy visual verification and bias diagnosis

### Spawn Location
- **Position:** [0.0, -1.3, 0.005]
- **Heading:** 90° (facing North/+Y)

### Modifying Routes
1. Choose node sequence from SDCSRoadMap (see `python/hal/products/mats.py`)
2. Generate path: `roadmap.generate_path([node1, node2, ...])`
3. Scale coordinates: `waypoints_scaled = waypoints * 10.0`
4. Downsample to ~5m spacing
5. Add lead-in from spawn to route start
6. Update `config.py` → `self.route_waypoints`

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

### Common Issues

**Vehicle veers left:**
- Model has systematic leftward bias (~0.34m in Y predictions)
- Solution: Apply bias correction in `control_converter.py`

**Route completion detected too early:**
- Check final waypoint distance from spawn
- Increase completion threshold in `route_manager.py`

**Steering oscillation:**
- Check waypoint filtering (remove waypoints < 0.05m from origin)
- Adjust PID gains in `config.py`

**Slow control loop:**
- Model inference takes ~0.5s
- Target: 20 Hz (50ms), actual: ~2 Hz
- Consider model optimization or async processing

---

## Key Parameters (`config.py`)

### Model
- `model_checkpoint`: Path to model weights
- `device`: 'cuda' or 'cpu'

### Camera
- `camera_resolution`: [820, 410] (width, height)
- `crop_bottom_percent`: 30% (remove hood)

### Control
- `target_point_lookahead`: 10.0 meters
- `speed_kp`, `speed_ki`, `speed_kd`: PID gains for speed
- `steering_k`: Pure pursuit gain

### Route
- `route_waypoints`: List of [x, y, z] waypoints
- `completion_threshold`: 2.0 meters

---

## Development Workflow

1. **Make changes** to integration code in `src/`
2. **Test** with `python src/main.py`
3. **Visualize** results with `python src/visualize_trajectory.py`
4. **Iterate** based on trajectory analysis
5. **Keep** only essential files, remove debug scripts

---

## Important Notes

- **Never modify** `python/` or `simlingo/` directories
  - The integration is designed to work WITHOUT modifying official Simlingo code
  - All fixes are in the wrapper (`src/simlingo_model.py`)

- **Placeholder values for image tokens:**
  - Image tokens (`<img>`, `</img>`, `<IMG_CONTEXT>`) are pre-existing in InternVL2 tokenizer
  - Must provide empty arrays for them in `placeholder_values` to avoid KeyError
  - These empty arrays satisfy the lookup but don't create waypoint embeddings
  - Vision embeddings replace them in a separate step

- **Always test** after configuration changes

- **Use trajectory visualization** to diagnose issues

- **SDCSRoadMap coordinates** need 10x scaling for QLabs

- **Steering convention:** QCar2 positive = right turn (opposite of CARLA)

- **Image preprocessing:** Must match CARLA training (crop bottom 30%, 2 patches)

---

## Quick Reference

| Task | File | Function/Class |
|------|------|----------------|
| Change route | `config.py` | `route_waypoints` |
| Adjust PID gains | `config.py` | `speed_kp/ki/kd` |
| Modify steering | `control_converter.py` | `calculate_steering()` |
| Change spawn | `config.py` | `qcar2_spawn_location` |
| Add bias correction | `control_converter.py` | `convert_to_qcar2_control()` |
| Visualize trajectory | `visualize_trajectory.py` | `visualize_trajectory()` |
| Generate new route | `python/hal/products/mats.py` | `SDCSRoadMap.generate_path()` |

---

**Last Updated:** October 8, 2025

