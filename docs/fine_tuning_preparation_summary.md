# SimLingo Fine-Tuning Preparation - Summary

This document provides a high-level overview of the fine-tuning preparation work completed for adapting SimLingo to the QLabs/QCar2 simulator.

## Completed Research Tasks

### ✅ Task 1: SimLingo Training Data Format Analysis

**Deliverable**: [`simlingo_training_data_format.md`](simlingo_training_data_format.md)

**Key Findings:**

1. **Data Structure**:
   - **Input**: `DrivingInput` with camera images, intrinsics/extrinsics, speed, target points, and language prompts
   - **Labels**: `DrivingLabel` with waypoints (11 points @ 0.2s), path (20 points @ 1m), and language answers

2. **Critical Insights**:
   - Waypoint special tokens (`<WAYPOINTS>`, etc.) are **NOT** in training labels
   - Training uses **two separate losses**: language (text) and waypoint (coordinates)
   - All waypoints are in **ego vehicle frame** (X=forward, Y=left, BEV)
   - Camera parameters: 1024×512 resolution, FOV=100°, position=[-1.5, 0.0, 2.0]

3. **Coordinate Frame**:
   - Origin: Vehicle center at ground level
   - Axes: X=forward, Y=left, Z=up
   - Transformation: `ego_point = R^T @ (world_point - translation)`

4. **Example Code**:
   - Complete sample construction code provided
   - Waypoint computation from measurements
   - Data augmentation (translation + rotation)

---

### ✅ Task 2: QLabs/QCar2 Data Collection Design

**Deliverable**: [`qlabs_data_collection_design.md`](qlabs_data_collection_design.md)

**Key Findings:**

1. **QLabs API Capabilities**:
   - Vehicle state: `get_world_transform()` → position, rotation
   - Camera images: `get_image(CAMERA_CSI_FRONT)` → 820×410 JPG
   - Collision detection: `set_velocity_and_request_state()` → frontHit, rearHit
   - LIDAR: `get_lidar()` → angles, distances (optional for obstacle detection)

2. **Coordinate Frame Differences**:
   - **CARLA**: X=forward, Y=right, Z=up
   - **QLabs**: X=forward, Y=left, Z=up
   - **Conversion**: Flip Y-axis when converting between frames

3. **Camera Configuration**:
   - Use `CAMERA_CSI_FRONT` (820×410 native, resize to 1024×512)
   - Extrinsics: [1.83, 0.0, 1.10] (ahead of vehicle center)
   - Intrinsics: Compute from FOV=160° and resolution

4. **Expert Controller Design**:
   - **Option 1 (Recommended)**: Reuse existing `RouteManager` + safety checks
   - **Option 2**: Implement rule-based pure pursuit controller
   - Predict future waypoints using kinematic bicycle model

5. **Data Recording Pipeline**:
   - Record at 4 Hz (every 0.25s)
   - Save images as JPG, measurements as JSON.gz
   - Structure: `session_XXX/measurements/` and `session_XXX/images/`

---

## Implementation Roadmap

### Phase 1: Basic Data Recorder (1-2 hours)

**Goal**: Create a working data recorder that saves images and measurements

**Tasks**:
1. Implement `QLabs DataRecorder` class
2. Record vehicle state (position, rotation, speed)
3. Capture camera images from `CAMERA_CSI_FRONT`
4. Save data in SimLingo-compatible format

**Code Location**: `src/data_recorder.py` (to be created)

**Dependencies**:
- `python/qvl/qcar2.py` (existing)
- `src/route_manager.py` (existing)
- `src/config.py` (existing)

---

### Phase 2: Expert Controller (2-3 hours)

**Goal**: Implement a simple controller that can follow routes and predict waypoints

**Tasks**:
1. Implement `QLabs ExpertController` class
2. Pure pursuit steering based on target points
3. Kinematic bicycle model for waypoint prediction
4. Speed control (maintain target speed)

**Code Location**: `src/expert_controller.py` (to be created)

**Key Parameters** (from SimLingo):
- Wheelbase: 1.4178275 m
- Steering gain: 0.36848336
- Max steering: ±30° (±π/6 rad)
- Target speed: 2.0 m/s (full-scale)

---

### Phase 3: Data Collection (2-3 hours)

**Goal**: Collect 50-100 high-quality training samples

**Scenarios** (10 samples each):
1. **Straight lane-keeping**: Follow straight road segments
2. **Gentle curves**: Navigate smooth turns
3. **Sharp turns**: Handle tight corners
4. **Obstacle avoidance**: Detect and avoid static obstacles
5. **Lane changes**: Change lanes smoothly

**Data Quality Checks**:
- ✅ No collisions during recording
- ✅ Smooth waypoint trajectories
- ✅ Consistent speed control
- ✅ Images properly captured and saved
- ✅ Measurements contain all required fields

---

### Phase 4: Dataset Preparation (1-2 hours)

**Goal**: Convert recorded data to SimLingo training format

**Tasks**:
1. Create custom dataset class (inherit from `dataset_base.py`)
2. Implement coordinate frame conversions (QLabs → CARLA)
3. Resize images (820×410 → 1024×512)
4. Generate language labels (optional: use simple templates)
5. Test data loading with SimLingo's dataloader

**Code Location**: `simlingo/simlingo_training/dataloader/dataset_qlabs.py` (to be created)

---

### Phase 5: Fine-Tuning (2-3 hours)

**Goal**: Fine-tune SimLingo on QLabs data

**Training Configuration**:
- **Freeze**: Vision encoder (InternVL2-300M)
- **Fine-tune**: Waypoint MLP heads (fully)
- **Optional**: LLM with LoRA (rank=16, alpha=32)

**Hyperparameters**:
- Learning rate: 1e-5 to 3e-5
- Batch size: 4-8
- Epochs: 10-20
- Loss weights: Waypoint loss ×2

**Expected Training Time**: 30-60 minutes on GPU

---

## Key Differences: CARLA vs QLabs

| Aspect | CARLA (SimLingo Training) | QLabs/QCar2 (Our Data) |
|--------|---------------------------|------------------------|
| **Coordinate Frame** | X=forward, Y=right, Z=up | X=forward, Y=left, Z=up |
| **Camera Position** | [-1.5, 0.0, 2.0] | [1.83, 0.0, 1.10] |
| **Camera Resolution** | 1024×512 | 820×410 (resize to 1024×512) |
| **Camera FOV** | 100° | 160° |
| **Scale** | 1:1 (full-scale) | 10:1 (QLabs is 10× physical) |
| **Speed Units** | m/s (full-scale) | m/s (full-scale, ×10 from physical) |

**Critical Conversion**:
```python
# QLabs ego frame → CARLA ego frame
def qlabs_to_carla_ego(qlabs_point):
    x, y = qlabs_point
    return np.array([x, -y], dtype=np.float32)  # Flip Y-axis
```

---

## Minimal Data Requirements

For each training sample:

**Input Data**:
- ✅ Camera image: 1024×512 RGB (resized from 820×410)
- ✅ Camera intrinsics: 3×3 matrix (computed from FOV=160°)
- ✅ Camera extrinsics: 4×4 matrix (fixed: [1.83, 0.0, 1.10])
- ✅ Vehicle speed: float (m/s)
- ✅ Target point: [x, y] in ego frame
- ✅ Language prompt: "Current speed: X.X m/s. Target point: <TARGET_POINT> <TARGET_POINT> Predict the waypoints."

**Label Data**:
- ✅ Waypoints: [11, 2] array (0.2s spacing, ego frame)
- ✅ Path: [20, 2] array (1m spacing, ego frame)
- ✅ Language answer: "Waypoints:" (or commentary if available)

**Total Samples Needed**: 50-100

---

## Reusable Components

### From Existing Codebase

1. **`src/route_manager.py`**:
   - ✅ Route waypoint management
   - ✅ Target point computation
   - ✅ World → Ego transformation
   - ✅ HLC computation (currently defaults to 4)

2. **`src/config.py`**:
   - ✅ Camera parameters (FOV, resolution)
   - ✅ QCar2 spawn location and rotation
   - ✅ Route waypoints (QLabs Cityscape Lite)
   - ✅ PID controller parameters

3. **`python/qvl/qcar2.py`**:
   - ✅ QLabs API wrapper
   - ✅ Camera image capture
   - ✅ Vehicle state queries
   - ✅ Control commands

### To Be Created

1. **`src/data_recorder.py`**:
   - Record driving sessions
   - Save images and measurements
   - Synchronize data at 4 Hz

2. **`src/expert_controller.py`**:
   - Pure pursuit steering
   - Waypoint prediction
   - Speed control

3. **`simlingo/simlingo_training/dataloader/dataset_qlabs.py`**:
   - Load QLabs data
   - Convert to SimLingo format
   - Handle coordinate transformations

---

## Expected Outcomes

### After Data Collection (Phase 1-3)

- ✅ 50-100 recorded driving sessions
- ✅ Images: 820×410 JPG files
- ✅ Measurements: JSON.gz with ego_matrix, speed, route, HLC
- ✅ Data organized in SimLingo-compatible structure

### After Fine-Tuning (Phase 4-5)

- ✅ Model adapted to QLabs visual domain
- ✅ Waypoint predictions calibrated for QCar2 physics
- ✅ Improved lane-keeping behavior
- ✅ Basic obstacle avoidance capability

### Performance Metrics

**Before Fine-Tuning** (current baseline):
- Lane-keeping: Poor (model trained on CARLA)
- Obstacle avoidance: Poor (domain mismatch)
- Waypoint accuracy: Low (physics mismatch)

**After Fine-Tuning** (expected):
- Lane-keeping: Good (adapted to QLabs)
- Obstacle avoidance: Moderate (depends on training data quality)
- Waypoint accuracy: High (calibrated for QCar2)

---

## Potential Challenges

### 1. Coordinate Frame Confusion

**Problem**: CARLA uses Y=right, QLabs uses Y=left

**Solution**: 
- Always convert to CARLA frame before training
- Use helper functions: `qlabs_to_carla_ego()` and `carla_to_qlabs_ego()`
- Test conversions with known waypoints

### 2. Camera Differences

**Problem**: Different FOV (100° vs 160°) and position

**Solution**:
- Fine-tune vision encoder (optional, but may help)
- Ensure camera intrinsics are correctly computed
- Consider data augmentation (crop, zoom)

### 3. Expert Controller Quality

**Problem**: Simple controller may not provide high-quality labels

**Solution**:
- Start with simple scenarios (straight roads, gentle curves)
- Manually verify waypoint quality before training
- Consider manual driving for complex scenarios

### 4. Limited Data Quantity

**Problem**: 50-100 samples may not be enough for full adaptation

**Solution**:
- Focus on critical scenarios (lane-keeping, obstacle avoidance)
- Use data augmentation (rotation, translation)
- Consider iterative data collection (collect → train → evaluate → collect more)

---

## Next Steps

1. **Review Documentation**:
   - Read `simlingo_training_data_format.md` for data format details
   - Read `qlabs_data_collection_design.md` for implementation design

2. **Implement Data Recorder** (Phase 1):
   - Create `src/data_recorder.py`
   - Test with manual driving
   - Verify data format

3. **Implement Expert Controller** (Phase 2):
   - Create `src/expert_controller.py`
   - Test on simple routes
   - Verify waypoint predictions

4. **Collect Data** (Phase 3):
   - Design 5-10 test scenarios
   - Record 50-100 samples
   - Verify data quality

5. **Prepare Dataset** (Phase 4):
   - Create `dataset_qlabs.py`
   - Test data loading
   - Verify compatibility with SimLingo

6. **Fine-Tune Model** (Phase 5):
   - Configure training script
   - Run fine-tuning
   - Evaluate on test scenarios

---

## Estimated Timeline

| Phase | Tasks | Time Estimate |
|-------|-------|---------------|
| Phase 1 | Data Recorder | 1-2 hours |
| Phase 2 | Expert Controller | 2-3 hours |
| Phase 3 | Data Collection | 2-3 hours |
| Phase 4 | Dataset Preparation | 1-2 hours |
| Phase 5 | Fine-Tuning | 2-3 hours |
| **Total** | | **8-13 hours** |

---

## References

- **SimLingo Paper**: CVPR 2025, 1st Place @ CARLA Challenge 2024
- **QLabs Documentation**: https://qlabs.quanserdocs.com/en/latest/
- **QCar2 API**: https://qlabs.quanserdocs.com/en/latest/Objects/qcar2_library.html
- **Existing Codebase**: `src/`, `python/qvl/`, `simlingo/`

---

**End of Summary**

