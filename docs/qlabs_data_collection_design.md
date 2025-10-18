# QLabs/QCar2 Data Collection Design

This document provides a complete design for collecting expert driving data from QLabs/QCar2 simulator in a format compatible with SimLingo's training pipeline.

## Table of Contents
1. [Overview](#overview)
2. [QLabs/QCar2 API Mapping](#qlabsqcar2-api-mapping)
3. [Expert Controller Architecture](#expert-controller-architecture)
4. [Data Recording Pipeline](#data-recording-pipeline)
5. [Implementation Plan](#implementation-plan)

---

## Overview

### Goal
Collect 50-100 driving samples from QLabs/QCar2 simulator focusing on:
1. **Lane-keeping**: Staying within lane boundaries
2. **Obstacle avoidance**: Detecting and avoiding obstacles

### Key Requirements
- **Compatibility**: Data must match SimLingo's expected format (see `simlingo_training_data_format.md`)
- **Simplicity**: Minimal implementation, reuse existing code where possible
- **Quality**: Expert-level driving behavior for high-quality labels

### Data Collection Frequency
- **Recording rate**: 4 Hz (every 0.25 seconds)
- **Rationale**: Matches SimLingo's training data frequency (CARLA runs at 20 Hz, but data is sampled at 4 Hz)

---

## QLabs/QCar2 API Mapping

### Available QCar2 State Information

Based on QLabs documentation and `python/qvl/qcar2.py`:

| SimLingo Required | QCar2 API Method | Return Format | Notes |
|-------------------|------------------|---------------|-------|
| **Vehicle Position** | `get_world_transform()` | `location [x, y, z]` | Full-scale coordinates (×10 from physical) |
| **Vehicle Rotation** | `get_world_transform()` | `rotation [roll, pitch, yaw]` | Radians |
| **Vehicle Speed** | `set_velocity_and_request_state()` | `location, rotation, frontHit, rearHit` | Speed must be tracked separately |
| **Camera Image** | `get_image(camera)` | `byte array (JPG)` | CSI cameras: 820×410, RGB: 640×480 |
| **Forward Vector** | `set_transform_and_request_state()` | `forward_vector [x, y, z]` | Unit vector |
| **Collision Detection** | `set_velocity_and_request_state()` | `frontHit, rearHit` | Boolean flags |

### Camera Configuration

**QCar2 Available Cameras:**
- `CAMERA_CSI_FRONT = 3`: 820×410 resolution (front-facing)
- `CAMERA_CSI_LEFT = 2`: 820×410 resolution
- `CAMERA_CSI_RIGHT = 0`: 820×410 resolution
- `CAMERA_CSI_BACK = 1`: 820×410 resolution
- `CAMERA_RGB = 4`: 640×480 resolution (RealSense)
- `CAMERA_DEPTH = 5`: 640×480 resolution (RealSense depth)

**Recommended for SimLingo:**
- Use `CAMERA_CSI_FRONT` (matches current implementation in `src/config.py`)
- Native resolution: 820×410
- Will be resized to 1024×512 for SimLingo

**Camera Extrinsics (from QLabs documentation):**

| Component | x (m) | y (m) | z (m) | Notes |
|-----------|-------|-------|-------|-------|
| Body frame | 0.0 | 0.0 | 0.0 | Between front/rear axles, ground level |
| CSI front | 1.83 | 0.0 | 1.10 | Front-facing camera |
| RealSense | 0.95 | 0.32 | 1.72 | RGB camera |

**Important**: QLabs coordinates are **10× larger** than physical QCar coordinates.

**Camera Intrinsics Calculation:**

For CSI front camera (820×410, estimated FOV ~160°):
```python
w, h = 820, 410
fov = 160.0  # degrees (from config.py)
focal = w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
K = np.array([
    [focal,   0.0,  w/2.0],
    [  0.0, focal,  h/2.0],
    [  0.0,   0.0,    1.0]
], dtype=np.float32)
```

**Camera Extrinsics Matrix:**

For CSI front camera at [1.83, 0.0, 1.10] with no rotation:
```python
extrinsics = np.array([
    [1.0, 0.0, 0.0, 1.83],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.10],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)
```

### Coordinate Frame Differences

**CARLA (SimLingo training data):**
- X: Forward
- Y: Right
- Z: Up
- Camera position: [-1.5, 0.0, 2.0] (behind vehicle center)

**QLabs/QCar2:**
- X: Forward
- Y: Left (opposite of CARLA!)
- Z: Up
- Camera position: [1.83, 0.0, 1.10] (ahead of vehicle center)

**Conversion Required:**
```python
def carla_to_qlabs_ego(carla_point):
    """Convert CARLA ego frame to QLabs ego frame."""
    x, y = carla_point
    return np.array([x, -y], dtype=np.float32)  # Flip Y-axis

def qlabs_to_carla_ego(qlabs_point):
    """Convert QLabs ego frame to CARLA ego frame."""
    x, y = qlabs_point
    return np.array([x, -y], dtype=np.float32)  # Flip Y-axis
```

---

## Expert Controller Architecture

### Option 1: Reuse Existing RouteManager (Recommended)

**Advantages:**
- ✅ Already implemented and tested
- ✅ Provides target points and HLC
- ✅ Minimal additional code

**Limitations:**
- ❌ No obstacle avoidance logic
- ❌ No lane boundary detection

**Solution**: Augment with simple safety checks.

### Option 2: Rule-Based Expert Controller

Implement a simple rule-based controller with:
1. **Lane-keeping**: Follow route waypoints with lateral error correction
2. **Obstacle avoidance**: Use LIDAR or vision-based detection
3. **Speed control**: Maintain target speed, slow down for turns

**Recommended Approach**: Start with Option 1, add safety features incrementally.

---

### Proposed Expert Controller Design

```python
class QLabs ExpertController:
    """
    Simple expert controller for QLabs data collection.
    Combines RouteManager with basic safety features.
    """
    
    def __init__(self, route_manager, config):
        self.route_manager = route_manager
        self.config = config
        self.target_speed = 2.0  # m/s (full-scale)
        
    def compute_control(self, vehicle_state):
        """
        Compute steering and throttle commands.
        
        Args:
            vehicle_state: Dict with 'position', 'rotation', 'speed'
        
        Returns:
            steering: float (radians)
            throttle: float (m/s)
        """
        # Get target point from route manager
        target_ego, _, hlc = self.route_manager.get_target_point_ego(
            vehicle_state['position'],
            vehicle_state['rotation'][2]  # yaw
        )
        
        # Simple pure pursuit steering
        lookahead = np.linalg.norm(target_ego)
        steering = np.arctan2(2 * self.wheelbase * target_ego[1], lookahead**2)
        steering = np.clip(steering, -np.pi/6, np.pi/6)  # ±30° max
        
        # Simple speed control
        current_speed = vehicle_state['speed']
        if current_speed < self.target_speed:
            throttle = self.target_speed
        else:
            throttle = 0.0
        
        return steering, throttle
    
    def predict_future_waypoints(self, vehicle_state, dt=0.2, num_points=11):
        """
        Predict future waypoints using kinematic bicycle model.
        
        Args:
            vehicle_state: Current vehicle state
            dt: Time step (0.2s for SimLingo)
            num_points: Number of waypoints (11 for SimLingo)
        
        Returns:
            waypoints: [num_points, 2] array in ego frame
        """
        waypoints = []
        
        # Start from current position (ego frame origin)
        x, y, yaw = 0.0, 0.0, 0.0
        speed = vehicle_state['speed']
        
        for i in range(num_points):
            # Get control at this timestep
            temp_state = {
                'position': vehicle_state['position'],
                'rotation': vehicle_state['rotation'],
                'speed': speed
            }
            steering, throttle = self.compute_control(temp_state)
            
            # Kinematic bicycle model (simplified)
            x += speed * np.cos(yaw) * dt
            y += speed * np.sin(yaw) * dt
            yaw += (speed / self.wheelbase) * np.tan(steering) * dt
            
            waypoints.append([x, y])
        
        return np.array(waypoints, dtype=np.float32)
```

---

## Data Recording Pipeline

### Recording Session Structure

```
data/
├── session_001/
│   ├── measurements/
│   │   ├── 0000.json.gz
│   │   ├── 0001.json.gz
│   │   └── ...
│   ├── images/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── metadata.json
├── session_002/
│   └── ...
```

### Measurement File Format

Each `XXXX.json.gz` contains:

```json
{
  "ego_matrix": [
    [1.0, 0.0, 0.0, x],
    [0.0, 1.0, 0.0, y],
    [0.0, 0.0, 1.0, z],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "speed": 5.2,
  "timestamp": 1234567890.123,
  "route": [[x1, y1, z1], [x2, y2, z2], ...],
  "route_original": [[x1, y1, z1], [x2, y2, z2], ...],
  "hlc": 4,
  "collision": false
}
```

### Data Recorder Implementation

```python
class QLabs DataRecorder:
    """
    Records driving data from QLabs in SimLingo-compatible format.
    """
    
    def __init__(self, qcar, route_manager, output_dir):
        self.qcar = qcar
        self.route_manager = route_manager
        self.output_dir = Path(output_dir)
        self.frame_count = 0
        self.recording = False
        
        # Create directories
        self.measurements_dir = self.output_dir / "measurements"
        self.images_dir = self.output_dir / "images"
        self.measurements_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def start_recording(self):
        """Start recording session."""
        self.recording = True
        self.frame_count = 0
        print(f"Recording started: {self.output_dir}")
    
    def stop_recording(self):
        """Stop recording session."""
        self.recording = False
        print(f"Recording stopped: {self.frame_count} frames saved")
    
    def record_frame(self, vehicle_state, control_command):
        """
        Record one frame of data.
        
        Args:
            vehicle_state: Dict with position, rotation, speed
            control_command: Dict with steering, throttle
        """
        if not self.recording:
            return
        
        # Get camera image
        success, image_data = self.qcar.get_image(self.qcar.CAMERA_CSI_FRONT)
        if not success:
            print(f"Warning: Failed to get image at frame {self.frame_count}")
            return
        
        # Save image
        image_path = self.images_dir / f"{self.frame_count:04d}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Build ego matrix
        pos = vehicle_state['position']
        rot = vehicle_state['rotation']
        ego_matrix = self._build_ego_matrix(pos, rot)
        
        # Get route waypoints
        route_waypoints = self.route_manager.route_waypoints.tolist()
        
        # Build measurement dict
        measurement = {
            'ego_matrix': ego_matrix.tolist(),
            'speed': vehicle_state['speed'],
            'timestamp': time.time(),
            'route': route_waypoints,
            'route_original': route_waypoints,
            'hlc': vehicle_state.get('hlc', 4),
            'collision': vehicle_state.get('collision', False),
            'steering': control_command['steering'],
            'throttle': control_command['throttle']
        }
        
        # Save measurement
        measurement_path = self.measurements_dir / f"{self.frame_count:04d}.json.gz"
        with gzip.open(measurement_path, 'wt') as f:
            json.dump(measurement, f)
        
        self.frame_count += 1
    
    def _build_ego_matrix(self, position, rotation):
        """
        Build 4x4 ego transformation matrix from position and rotation.
        
        Args:
            position: [x, y, z]
            rotation: [roll, pitch, yaw] in radians
        
        Returns:
            4x4 numpy array
        """
        roll, pitch, yaw = rotation
        
        # Rotation matrix (ZYX Euler angles)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        
        # Build 4x4 matrix
        ego_matrix = np.eye(4, dtype=np.float32)
        ego_matrix[:3, :3] = R
        ego_matrix[:3, 3] = position
        
        return ego_matrix
```

---

## Implementation Plan

### Phase 1: Basic Data Recorder (1-2 hours)

**Tasks:**
1. Create `QLabs DataRecorder` class
2. Implement frame recording (images + measurements)
3. Test with manual driving (keyboard control)

**Deliverable**: Working data recorder that saves images and measurements

### Phase 2: Expert Controller (2-3 hours)

**Tasks:**
1. Create `QLabs ExpertController` class
2. Implement pure pursuit steering
3. Implement waypoint prediction
4. Test on simple straight route

**Deliverable**: Controller that can follow route waypoints

### Phase 3: Data Collection (2-3 hours)

**Tasks:**
1. Design 5-10 test scenarios:
   - Straight lane-keeping (10 samples)
   - Gentle curves (10 samples)
   - Sharp turns (10 samples)
   - Obstacle avoidance (10 samples)
   - Lane changes (10 samples)
2. Run data collection sessions
3. Verify data quality

**Deliverable**: 50-100 high-quality training samples

### Phase 4: Data Conversion (1-2 hours)

**Tasks:**
1. Create dataset class compatible with SimLingo's `dataset_driving.py`
2. Implement coordinate frame conversions
3. Generate language labels (commentary)
4. Test data loading with SimLingo's dataloader

**Deliverable**: Dataset ready for fine-tuning

---

## Minimal Implementation Checklist

For each recorded frame, you need:

**From QLabs API:**
- ✅ Vehicle position: `get_world_transform()` → `location [x, y, z]`
- ✅ Vehicle rotation: `get_world_transform()` → `rotation [roll, pitch, yaw]`
- ✅ Vehicle speed: Track from control commands
- ✅ Camera image: `get_image(CAMERA_CSI_FRONT)` → JPG bytes
- ✅ Collision status: `set_velocity_and_request_state()` → `frontHit, rearHit`

**From Expert Controller:**
- ✅ Future waypoints: Predict using kinematic model (11 points, 0.2s apart)
- ✅ Path waypoints: Extract from route (20 points, 1m apart)
- ✅ HLC: From RouteManager (default: 4)

**From Configuration:**
- ✅ Camera intrinsics: Compute from FOV and resolution
- ✅ Camera extrinsics: Fixed mounting position [1.83, 0.0, 1.10]
- ✅ Target speed: 2.0 m/s (full-scale)

**Data Processing:**
- ✅ Coordinate conversion: QLabs → CARLA (flip Y-axis)
- ✅ Image resizing: 820×410 → 1024×512
- ✅ Ego frame transformation: World → Ego
- ✅ Language labels: Generate commentary (optional for minimal version)

---

## Next Steps

1. **Review this design** with the user
2. **Implement Phase 1** (Basic Data Recorder)
3. **Test data recording** with manual driving
4. **Implement Phase 2** (Expert Controller)
5. **Collect 50-100 samples** (Phase 3)
6. **Fine-tune SimLingo** on collected data

**Estimated Total Time**: 6-10 hours

---

**End of Design Document**

