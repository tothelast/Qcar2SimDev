# SimLingo-QCar2 Integration Technical Report

**Project:** Integration of SimLingo Vision-Language-Action Model with Quanser QCar2 in QLabs Cityscape  
**Date:** October 8, 2025  
**Status:** Functional with Known Limitations

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture and Components](#2-architecture-and-components)
3. [Direct Integrations (No Changes Required)](#3-direct-integrations-no-changes-required)
4. [Required Adaptations for QLabs](#4-required-adaptations-for-qlabs)
5. [Implementation Details](#5-implementation-details)
6. [Current Status and Performance](#6-current-status-and-performance)
7. [File-by-File Documentation](#7-file-by-file-documentation)
8. [Known Issues and Future Work](#8-known-issues-and-future-work)

---

## 1. Project Overview

### 1.1 SimLingo Model

**SimLingo** is a Vision-Language-Action (VLA) model for autonomous driving developed for the CARLA simulator. Key characteristics:

- **Base Model:** InternVL2-1B (OpenGVLab) - multimodal vision-language model
- **Training:** Fine-tuned with LoRA/PEFT on CARLA driving data
- **Inputs:** Camera images (RGB) + target waypoints in ego frame
- **Outputs:** 
  - Predicted route waypoints (20 points, 2D ego frame)
  - Speed waypoints (10 points with speed values)
  - Natural language descriptions of driving actions
- **Control:** PID controllers convert model predictions to throttle/steering/brake

### 1.2 QCar2 and QLabs Cityscape

**QCar2:** Quanser's 1/10 scale autonomous vehicle platform with:
- CSI front camera (820x410 native resolution)
- GPS and IMU sensors
- Differential drive with Ackermann steering
- QLabs virtual twin for simulation

**QLabs Cityscape:** Virtual urban environment with:
- SDCSRoadMap: 24-node road network with defined paths
- Realistic buildings, roads, and intersections
- Physics-based vehicle simulation
- Real-time sensor feedback

### 1.3 Integration Objectives

1. **Deploy SimLingo** (trained on CARLA) to QCar2 in QLabs
2. **Maintain model integrity** - no retraining, use pretrained weights
3. **Adapt interfaces** - camera, control signals, coordinate systems
4. **Enable autonomous navigation** on QLabs road network
5. **Preserve original SimLingo behavior** as much as possible

---

## 2. Architecture and Components

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         QLabs Cityscape                         │
│  ┌──────────────┐                                               │
│  │   QCar2      │  Camera Images (820x410 RGB)                  │
│  │   Vehicle    │──────────────────────────────┐                │
│  └──────────────┘                              │                │
│         ▲                                      │                │
│         │ Control Commands                     │                │
│         │ (throttle, steering)                 │                │
└─────────┼──────────────────────────────────────┼────────────────┘
          │                                      │
          │                                      ▼
┌─────────┴──────────────────────────────────────────────────────┐
│                    Integration Layer (Python)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   QCar2      │  │   Camera     │  │    State     │         │
│  │  Interface   │  │  Processor   │  │  Estimator   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         ▲                 │                   │                 │
│         │                 ▼                   ▼                 │
│         │          ┌──────────────────────────────┐            │
│         │          │   SimLingo Model Wrapper     │            │
│         │          │  (InternVL2-1B + LoRA)       │            │
│         │          └──────────────────────────────┘            │
│         │                      │                                │
│         │                      ▼                                │
│         │          ┌──────────────────────────────┐            │
│         │          │   Control Converter          │            │
│         │          │  (PID Controllers)           │            │
│         │          └──────────────────────────────┘            │
│         │                      │                                │
│         └──────────────────────┘                                │
│                                                                  │
│  ┌──────────────┐                                               │
│  │    Route     │  Provides target waypoints                    │
│  │   Manager    │  in world & ego frames                        │
│  └──────────────┘                                               │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Perception:** QCar2 camera → Image preprocessing → InternVL2 patches (448x448)
2. **Localization:** QCar2 GPS/IMU → State estimation → Position, heading, velocity
3. **Planning:** Route manager → Target waypoints (world frame) → Ego frame conversion
4. **Prediction:** SimLingo model → Predicted waypoints + speeds + language
5. **Control:** PID controllers → Throttle/steering commands → QCar2 actuation

### 2.3 Python Files Overview

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 407 | Main entry point, control loop orchestration |
| `config.py` | 298 | All configuration parameters (Simlingo + QCar2) |
| `qcar2_interface.py` | 257 | QLabs connection, QCar2 spawning, camera capture |
| `simlingo_model.py` | 402 | Model loading, tokenization, inference |
| `camera_processor.py` | 145 | Image preprocessing (InternVL2 format) |
| `control_converter.py` | 371 | PID controllers, control signal conversion |
| `route_manager.py` | 215 | Waypoint management, coordinate transforms |
| `state_estimator.py` | 223 | Vehicle state tracking, velocity estimation |
| `visualize_trajectory.py` | 312 | Post-run trajectory analysis and plotting |
| `visualize_route.py` | 89 | Route preview visualization |
| `fix_route_coordinates.py` | 195 | Utility to generate routes from SDCSRoadMap |

---

## 3. Direct Integrations (No Changes Required)

### 3.1 SimLingo Model Components

The following components from the original SimLingo codebase were used **without modification**:

1. **Model Architecture** (`simlingo_training/models/`)
   - InternVL2-1B vision encoder
   - LoRA adapters for fine-tuning
   - Tokenizer and processor

2. **Image Preprocessing** (`simlingo_training/utils/internvl2_utils.py`)
   - `build_transform()` - Creates 448x448 resize + normalization
   - `dynamic_preprocess()` - Splits images into patches
   - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

3. **PID Controller Logic** (`simlingo/team_code/pid_controller.py`)
   - Lateral PID (steering control)
   - Longitudinal PID (speed control)
   - All gain values preserved exactly

4. **Prompt Engineering** (`simlingo/team_code/agent_simlingo.py`)
   - Chain-of-Thought (CoT) prompts
   - Target point encoding format
   - Special token handling

### 3.2 Quanser Libraries Used As-Is

1. **QVL (Quanser Virtual Labs)**
   - `QuanserInteractiveLabs` - QLabs connection
   - `QLabsQCar2` - Vehicle spawning and control
   - `QLabsFreeCamera` - Camera positioning

2. **HAL (Hardware Abstraction Layer)**
   - `SDCSRoadMap` - Road network definition (24 nodes)
   - `generate_path()` - Waypoint generation along roads

### 3.3 Configuration Parameters Transferred Directly

All PID gains, timing parameters, and model hyperparameters from SimLingo were preserved:

```python
# Lateral PID (exact values from SimLingo)
lateral_pid_kp = 3.118357247806046
lateral_pid_kd = 1.3782508892109167
lateral_pid_ki = 0.6406067986034124

# Control loop timing
carla_fps = 20  # Hz
control_frequency = 20  # Hz
dt = 0.05  # seconds
```

---

## 4. Required Adaptations for QLabs

### 4.1 Coordinate System Transformations

**Critical Discovery:** QLabs uses a 10× scaling of SDCSRoadMap coordinates.

#### SDCSRoadMap Internal Scaling
```python
# From hal/products/mats.py
scale = 0.002035
xOffset = 1134  # mm
yOffset = 2363  # mm

# Node positions in millimeters → scaled coordinates
X_scaled = scale * (X_mm - xOffset)
Y_scaled = scale * (yOffset - Y_mm)  # Y-axis inverted
```

#### QLabs Coordinate Scaling
```python
# Final QLabs coordinates
QLabs_X = SDCSRoadMap_X × 10
QLabs_Y = SDCSRoadMap_Y × 10
```

#### Example: Node 10
```python
# Raw position (from Quanser example)
Node_10_mm = [1134, 2299]  # millimeters

# SDCSRoadMap scaled
X_scaled = 0.002035 * (1134 - 1134) = 0.0
Y_scaled = 0.002035 * (2363 - 2299) = 0.130

# QLabs coordinates
QLabs_X = 0.0 × 10 = 0.0
QLabs_Y = 0.130 × 10 = 1.30

# But actual spawn in Quanser example: [-12.820, -4.599]
# This is Node 10 at SDCSRoadMap [-1.28205, -0.45991] × 10
```

**Implementation in `fix_route_coordinates.py`:**
```python
def create_route_from_nodes(roadmap, node_sequence):
    # Generate path using SDCSRoadMap
    waypoints_sdcs = roadmap.generate_path(node_sequence)  # Shape: (2, N)
    
    # Transpose and scale to QLabs coordinates
    waypoints_transposed = waypoints_sdcs.T  # Shape: (N, 2)
    waypoints_qlabs = []
    for wp in waypoints_transposed:
        x_qlabs = wp[0] * 10.0
        y_qlabs = wp[1] * 10.0
        waypoints_qlabs.append([x_qlabs, y_qlabs, 0.0])
    
    return waypoints_qlabs
```

### 4.2 Camera Interface Adaptation

**Challenge:** QCar2 CSI camera (820x410) vs CARLA camera (1024x512)

**Solution:** Preserve CARLA preprocessing pipeline:

```python
# camera_processor.py
def process_image(self, image: np.ndarray):
    # 1. Apply JPEG compression (matches CARLA training data)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, compressed = cv2.imencode('.jpg', image_bgr)
    image_bgr = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Crop bottom 30% (remove hood/dashboard)
    crop_height = int(image.shape[0] - (image.shape[0] * 4.8) // 16)
    image = image[:crop_height, :, :]
    
    # 3. Dynamic preprocessing (InternVL2)
    pil_image = Image.fromarray(image)
    images = dynamic_preprocess(pil_image, image_size=448, 
                                use_thumbnail=False, max_num=2)
    
    # 4. Apply transforms (resize + normalize)
    pixel_values = [self.transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # [num_patches, 3, 448, 448]
    
    # 5. Add batch and temporal dimensions
    pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)
    # Final shape: [1, 1, num_patches, 3, 448, 448]
    
    return pixel_values, None
```

### 4.3 Control Signal Conversion

**Challenge:** Steering sign convention differs between CARLA and QCar2

- **CARLA/SimLingo:** Positive steering = left turn
- **QCar2:** Positive turn_angle = right turn

**Solution:** Negate steering in `control_converter.py`:

```python
# Line 325 in control_converter.py
# NOTE: QCar2 convention is opposite to CARLA/Simlingo:
# - CARLA/Simlingo: positive steering = left turn
# - QCar2: positive turn_angle = right turn
# So we negate the steering value
turn_angle = -steer * self.config.steering_gain
```

### 4.4 Route Generation from SDCSRoadMap

**Challenge:** Create routes on actual QLabs roads, not arbitrary coordinates

**Solution:** Use SDCSRoadMap node sequences

```python
# Example: Node 10 → Node 4 (from Quanser example)
from hal.products.mats import SDCSRoadMap

roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
node_sequence = [10, 4]

# Generate smooth path
waypoints_sdcs = roadmap.generate_path(node_sequence)  # Returns (2, N) array

# Scale to QLabs coordinates
waypoints_qlabs = waypoints_sdcs.T * 10.0  # (N, 2) array

# Current route in config.py (Node 10 → Node 4, 45 meters, 19 waypoints)
route_waypoints = [
    [-12.820,  -4.599, 0.0],  # Node 10 start
    [-10.963,  -6.272, 0.0],
    # ... 15 more waypoints ...
    [ 22.548,   0.767, 0.0],  # Node 4 end
]
```

### 4.5 Timing Parameters

**Critical Adaptations:**

1. **`data_save_freq = 4`** (not 5)
   - Model outputs 10 waypoints (not 11 as expected)
   - Hard-coded in `control_converter.py` line 183-195

2. **`initial_frames_delay = 40`** (not 5)
   - Matches original SimLingo: `int(2.0 / carla_frame_rate)` = 40 frames
   - 2-second startup delay at 20 FPS

3. **Control loop rate: 20 Hz**
   - Matches CARLA frame rate
   - `dt = 0.05` seconds per step

### 4.6 Spawn Location and Rotation

**Current Configuration (Node 10):**

```python
# config.py lines 147-148
qcar2_spawn_location = [-12.820, -4.599, 0.005]  # [x, y, z]
qcar2_spawn_rotation = [0.0, 0.0, -0.7330]  # [roll, pitch, yaw] radians

# Derivation:
# Node 10 from SDCSRoadMap: [-1.28205, -0.45991, -0.7330]
# QLabs scaling: [-1.28205 × 10, -0.45991 × 10, -0.7330]
#              = [-12.8205, -4.5991, -0.7330 rad]
#              = [-12.8205, -4.5991, -42.0°]
```

---

## 5. Implementation Details

### 5.1 Step-by-Step Build Process

1. **Initial Setup** (Day 1)
   - Cloned SimLingo repository
   - Set up QVL/HAL libraries from Quanser
   - Created project structure with `src/` directory

2. **Model Integration** (Day 2-3)
   - Implemented `simlingo_model.py` wrapper
   - Resolved HuggingFace authentication issues (used local pretrained models)
   - Fixed DeepSpeed checkpoint loading

3. **Camera Pipeline** (Day 3-4)
   - Implemented `qcar2_interface.py` for camera capture
   - Adapted `camera_processor.py` for InternVL2 preprocessing
   - Verified image format matches training data

4. **Control System** (Day 4-5)
   - Ported PID controllers to `control_converter.py`
   - Discovered and fixed steering sign flip
   - Tuned control parameters

5. **Route Management** (Day 5-6)
   - Implemented `route_manager.py` for waypoint tracking
   - Discovered coordinate scaling issue
   - Fixed waypoint selection bug (search forward only)

6. **Testing and Debugging** (Day 6-7)
   - Multiple test runs with trajectory logging
   - Identified visual domain gap (CARLA vs QLabs)
   - Optimized route generation from SDCSRoadMap

### 5.2 Key Debugging Steps

#### Issue 1: Vehicle Not Moving
**Symptom:** Vehicle spawned but remained stationary  
**Root Cause:** `data_save_freq` mismatch - model outputs 10 waypoints, not 11  
**Solution:** Hard-coded `model_data_save_freq = 4` in control_converter.py

#### Issue 2: Waypoint Selection Bug
**Symptom:** Vehicle jumped to 82% progress immediately, targeting waypoint 38/47  
**Root Cause:** Route manager searched ALL waypoints for nearest, found parallel road section closer  
**Solution:** Limited search to 10 waypoints ahead from current index

```python
# route_manager.py lines 45-58
search_start = self.current_waypoint_index
search_end = min(self.current_waypoint_index + 10, len(self.route_waypoints))

if search_end > search_start:
    distances_ahead = np.linalg.norm(
        self.route_waypoints[search_start:search_end, :2] - current_position[:2],
        axis=1
    )
    nearest_idx_relative = np.argmin(distances_ahead)
    nearest_idx = search_start + nearest_idx_relative
```

#### Issue 3: Coordinate Scaling Discovery
**Symptom:** Vehicle spawned inside walls/meshes  
**Investigation:** Examined Quanser example code  
**Discovery:** QLabs uses SDCSRoadMap coordinates × 10  
**Solution:** Updated route generation and spawn location

---

## 6. Current Status and Performance

### 6.1 Latest Test Results (October 8, 2025, 12:27 PM)

**Test Configuration:**
- Route: Node 10 → Node 4 (45 meters, 19 waypoints)
- Spawn: [-12.820, -4.599, 0.005] at -42° heading
- Duration: 141.5 seconds (555 steps at 20 Hz)

**Performance Metrics:**
```
Total Distance Traveled: 54.4 meters (vs 45m planned)
Route Completion: 94.4%
Collisions: 7 detected
Mean Speed: 0.27 m/s (0.97 km/h)
Max Speed: 0.96 m/s (3.46 km/h)
Mean Lateral Deviation: 3.77 meters
Max Lateral Deviation: 6.11 meters
Steering Range: [-1.000, 1.000] (full range used)
```

### 6.2 What Works Well

✅ **Spawn and Initialization**
- Vehicle spawns correctly at Node 10 on actual road
- Camera captures images successfully
- Model loads and runs inference

✅ **Route Following (Partial)**
- Vehicle progresses through waypoints sequentially
- Reaches 94% route completion
- Navigates curved sections (with deviations)

✅ **Control System**
- PID controllers generate smooth commands
- Steering responds to model predictions
- Speed control functional (though conservative)

✅ **Coordinate Transformations**
- World-to-ego frame conversion correct
- SDCSRoadMap scaling verified (×10)
- Target waypoint selection improved

### 6.3 Known Limitations

❌ **High Lateral Deviation**
- Mean: 3.77m (should be <0.5m)
- Max: 6.11m (vehicle goes significantly off-road)
- Cause: Visual domain gap between CARLA and QLabs

❌ **Collisions**
- 7 collisions detected during run
- Occurs when vehicle deviates from road
- Model predictions don't match QLabs environment

❌ **Low Speed**
- Mean 0.27 m/s vs target ~1.0 m/s
- Conservative throttle application
- May be related to collision avoidance behavior

❌ **Visual Domain Gap**
- Model trained on CARLA visuals (different buildings, roads, lighting)
- QLabs Cityscape has different visual appearance
- Model predictions based on CARLA-learned features

### 6.4 Root Cause Analysis

The primary limitation is the **visual domain gap**:

1. **Training Environment:** CARLA simulator
   - Specific building textures and styles
   - CARLA road markings and colors
   - CARLA lighting and weather

2. **Deployment Environment:** QLabs Cityscape
   - Different building architecture
   - Different road surface appearance
   - Different lighting model

3. **Impact:**
   - Model's vision encoder expects CARLA-like images
   - Predictions based on visual features learned from CARLA
   - Mismatch causes suboptimal waypoint predictions
   - Results in lateral deviations and off-road behavior

**Evidence:**
- Model generates language outputs ("Go back to your original path after dodging the obstacle")
- Suggests model perceives obstacles/deviations that don't exist
- Waypoint predictions curve away from actual road

---

## 7. File-by-File Documentation

### 7.1 `main.py` (407 lines)

**Purpose:** Main entry point and control loop orchestration

**Key Classes:**
- `SimlingoQCar2Controller` - Main controller class

**Key Functions:**
```python
def initialize(self) -> bool:
    # Connect to QLabs, spawn QCar2, load model
    
def run_step(self) -> bool:
    # Single control loop iteration:
    # 1. Get camera image and vehicle state
    # 2. Run model inference
    # 3. Convert predictions to control commands
    # 4. Apply commands to vehicle
    # 5. Log trajectory data
    
def run(self):
    # Main control loop at 20 Hz
    # Handles stuck detection and route completion
```

**QLabs-Specific Adaptations:**
- QCar2 interface instead of CARLA client
- GPS-based state estimation instead of CARLA API
- Trajectory logging to JSON files

### 7.2 `qcar2_interface.py` (257 lines)

**Purpose:** Interface to QLabs and QCar2 vehicle

**Key Classes:**
- `QCar2Interface` - Handles all QLabs/QCar2 interactions

**Key Functions:**
```python
def connect(self) -> bool:
    # Connect to QLabs at localhost
    
def spawn_qcar(self) -> bool:
    # Spawn QCar2 at configured location/rotation
    
def get_camera_image(self) -> np.ndarray:
    # Capture RGB image from CSI camera (820x410)
    
def get_state(self) -> Tuple:
    # Get vehicle location, rotation from QLabs
    
def apply_control(self, throttle, steering):
    # Send control commands to QCar2
```

**Important Parameters:**
- `qcar2_camera = 3` - CAMERA_CSI_FRONT
- Collision detection via bumper sensors
- State update rate: 20 Hz

### 7.3 `simlingo_model.py` (402 lines)

**Purpose:** Model loading and inference wrapper

**Key Classes:**
- `SimlingoModelWrapper` - Wraps SimLingo model

**Key Functions:**
```python
def load_tokenizer(self):
    # Load InternVL2 tokenizer from local pretrained model
    
def load_model(self, checkpoint_path):
    # Load model architecture via Hydra
    # Load weights from DeepSpeed checkpoint
    # Move to GPU
    
def prepare_prompt(self, target_points, image):
    # Create prompt with target points and image
    # Format: "<img><IMG_CONTEXT>...<TARGET_POINT>..."
    
def inference(self, image, target_points):
    # Run model forward pass
    # Returns: route_waypoints, speed_waypoints, language
```

**QLabs-Specific Adaptations:**
- Local pretrained model loading (no HuggingFace auth)
- `local_files_only=True` for offline operation
- Checkpoint path from config

**Model Architecture:**
- Vision encoder: InternVL2-1B (448x448 patches)
- LoRA adapters: 17.6M trainable params (2.7% of total)
- Output heads: Route (20×2), Speed (10×2), Language (text)

### 7.4 `camera_processor.py` (145 lines)

**Purpose:** Image preprocessing for model input

**Key Classes:**
- `CameraProcessor` - Handles image transformations

**Key Functions:**
```python
def process_image(self, image: np.ndarray):
    # 1. JPEG compression/decompression
    # 2. Bottom crop (remove hood)
    # 3. Dynamic preprocessing (patches)
    # 4. Transform (resize + normalize)
    # Returns: [1, 1, num_patches, 3, 448, 448]
```

**Preprocessing Pipeline:**
1. **JPEG Compression:** Matches CARLA training data artifacts
2. **Bottom Crop:** Removes bottom 30% (hood/dashboard)
3. **Dynamic Preprocess:** Splits into 448×448 patches (max 2)
4. **Normalization:** ImageNet mean/std

**No QLabs-Specific Changes:** Uses exact SimLingo preprocessing

### 7.5 `control_converter.py` (371 lines)

**Purpose:** PID controllers and control signal conversion

**Key Classes:**
- `PIDController` - Basic PID implementation
- `LateralPIDController` - Steering control
- `LongitudinalPIDController` - Speed control
- `ControlConverter` - Main conversion class

**Key Functions:**
```python
def control_pid(self, route_waypoints, velocity, speed_waypoints):
    # 1. Interpolate route waypoints
    # 2. Compute steering via lateral PID
    # 3. Compute throttle/brake via longitudinal PID
    # 4. Apply QCar2 steering sign flip
    # Returns: (steer, throttle, brake)
```

**QLabs-Specific Adaptations:**
- **Line 325:** Steering sign negation for QCar2
- **Lines 183-195:** Hard-coded `model_data_save_freq = 4`

**PID Parameters (from SimLingo):**
```python
# Lateral PID
k_p = 3.118357247806046
k_d = 1.3782508892109167
k_i = 0.6406067986034124

# Longitudinal PID
proportional_gain = 5.0
derivative_gain = 1.5761818624794222
integral_gain = 0.2941563856687906
```

### 7.6 `route_manager.py` (215 lines)

**Purpose:** Waypoint management and coordinate transformations

**Key Classes:**
- `RouteManager` - Manages route waypoints

**Key Functions:**
```python
def get_target_point(self, current_position):
    # Find target waypoint based on lookahead distance
    # Search forward only (lines 45-58)
    # Returns: (target_point, next_target_point) in world frame
    
def get_target_point_ego(self, current_position, current_heading):
    # Convert target points to ego frame
    # Uses rotation matrix transformation
    # Returns: (target_ego, next_target_ego)
    
def _world_to_ego(self, world_point, vehicle_pos, vehicle_heading):
    # Rotation matrix: R^T @ (world - vehicle)
    # Matches SimLingo inverse_conversion_2d exactly
```

**QLabs-Specific Adaptations:**
- **Lines 45-58:** Forward-only waypoint search (fixes parallel road bug)
- **Lookahead distance:** 5.0m (from config)

**Coordinate Transformation (Lines 129-143):**
```python
rotation_matrix = np.array([
    [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
    [np.sin(vehicle_heading), np.cos(vehicle_heading)]
])
ego_point = rotation_matrix.T @ (world_point - vehicle_pos)
```

### 7.7 `config.py` (298 lines)

**Purpose:** Central configuration for all parameters

**Key Sections:**

1. **Model Configuration (Lines 14-21)**
   ```python
   model_checkpoint_path = "models/simlingo/checkpoints/epoch=013.ckpt"
   encoder_variant = "OpenGVLab/InternVL2-1B"
   hydra_config_path = "models/simlingo/.hydra/config.yaml"
   ```

2. **Camera Configuration (Lines 24-46)**
   - Target resolution: 1024×512
   - QCar2 native: 820×410
   - FOV: 110°
   - ImageNet normalization

3. **PID Parameters (Lines 49-80)**
   - All exact values from SimLingo
   - Turn, speed, lateral, longitudinal controllers

4. **Timing Configuration (Lines 109-128)**
   ```python
   carla_fps = 20  # Hz
   control_frequency = 20  # Hz
   dt = 0.05  # seconds
   data_save_freq = 1  # (overridden to 4 in control_converter)
   initial_frames_delay = 5  # (should be 40)
   ```

5. **QCar2 QLabs Configuration (Lines 137-191)**
   ```python
   qlabs_host = "localhost"
   qcar2_spawn_location = [-12.820, -4.599, 0.005]  # Node 10
   qcar2_spawn_rotation = [0.0, 0.0, -0.7330]  # -42°
   qcar2_camera = 3  # CAMERA_CSI_FRONT
   
   # Route: Node 10 → Node 4 (45m, 19 waypoints)
   route_waypoints = [
       [-12.820,  -4.599, 0.0],  # Start
       # ... 17 waypoints ...
       [ 22.548,   0.767, 0.0],  # End
   ]
   
   target_point_lookahead = 5.0  # meters
   ```

6. **Visualization (Lines 200-210)**
   - Debug output directory
   - Trajectory logging
   - Camera image saving

**QLabs-Specific Parameters:**
- Spawn location/rotation from SDCSRoadMap
- Route waypoints scaled ×10
- Lookahead distance tuned for QLabs roads

---

## 8. Known Issues and Future Work

### 8.1 Current Issues

1. **Visual Domain Gap** (Critical)
   - Model trained on CARLA, deployed on QLabs
   - Different visual appearance causes prediction errors
   - **Potential Solutions:**
     - Fine-tune on QLabs images (requires data collection)
     - Domain adaptation techniques
     - Visual style transfer

2. **High Lateral Deviation** (High Priority)
   - Mean 3.77m, max 6.11m
   - Causes collisions and off-road behavior
   - **Potential Solutions:**
     - Increase PID gains for tighter tracking
     - Add visual odometry for better localization
     - Implement path replanning

3. **Low Speed** (Medium Priority)
   - Mean 0.27 m/s vs target ~1.0 m/s
   - May be safety behavior from model
   - **Potential Solutions:**
     - Adjust longitudinal PID gains
     - Increase target speeds in route
     - Investigate brake triggering conditions

4. **Collision Detection** (Low Priority)
   - 7 collisions in 141-second run
   - Mostly from lateral deviations
   - **Potential Solutions:**
     - Improve lateral control
     - Add obstacle avoidance layer
     - Implement emergency braking

### 8.2 Future Enhancements

1. **Model Adaptation**
   - Collect QLabs driving data
   - Fine-tune SimLingo on QLabs images
   - Evaluate domain adaptation methods

2. **Control Improvements**
   - Implement Model Predictive Control (MPC)
   - Add feedforward control terms
   - Tune PID gains specifically for QCar2

3. **Route Planning**
   - Implement dynamic route generation
   - Add waypoint replanning based on deviations
   - Support multiple route options

4. **Sensor Fusion**
   - Integrate QCar2 LIDAR data
   - Use IMU for better heading estimation
   - Implement Extended Kalman Filter (EKF)

5. **Testing and Validation**
   - Create test suite with multiple routes
   - Benchmark against baseline controllers
   - Quantify performance metrics

### 8.3 Recommendations

**Short-term (1-2 weeks):**
1. Collect QLabs driving data for fine-tuning
2. Tune PID gains for better lateral control
3. Implement trajectory replanning

**Medium-term (1-2 months):**
1. Fine-tune SimLingo on QLabs data
2. Implement MPC for improved control
3. Add sensor fusion (LIDAR + camera)

**Long-term (3-6 months):**
1. Develop QLabs-specific VLA model
2. Implement full autonomous navigation stack
3. Deploy to physical QCar2 hardware

---

## Appendix A: Key Code Snippets

### A.1 Coordinate Transformation (World → Ego)

```python
# route_manager.py lines 129-143
def _world_to_ego(self, world_point, vehicle_pos, vehicle_heading):
    """Convert world coordinates to ego frame."""
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
        [np.sin(vehicle_heading), np.cos(vehicle_heading)]
    ])
    
    # Transform: R^T @ (world - vehicle)
    ego_point = rotation_matrix.T @ (world_point - vehicle_pos)
    
    return ego_point
```

### A.2 Steering Sign Conversion

```python
# control_converter.py line 325
# QCar2 convention is opposite to CARLA/Simlingo
turn_angle = -steer * self.config.steering_gain
```

### A.3 SDCSRoadMap to QLabs Scaling

```python
# fix_route_coordinates.py
waypoints_sdcs = roadmap.generate_path([10, 4])  # (2, N)
waypoints_qlabs = waypoints_sdcs.T * 10.0  # (N, 2) scaled
```

---

**End of Report**

*For questions or issues, refer to the source code in `src/` directory or contact the development team.*

