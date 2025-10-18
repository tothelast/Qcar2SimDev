# SimLingo Training Data Format Specification

This document provides a complete specification of the data format required for fine-tuning the SimLingo model, based on analysis of the original codebase.

## Table of Contents
1. [Overview](#overview)
2. [Input Data Structure](#input-data-structure)
3. [Label Data Structure](#label-data-structure)
4. [Coordinate Frames and Transformations](#coordinate-frames-and-transformations)
5. [Example Code](#example-code)

---

## Overview

SimLingo training requires two main data structures:
- **`DrivingInput`**: Contains sensor data, vehicle state, and language prompts
- **`DrivingLabel`**: Contains ground truth waypoints and language answers

The model is trained with **two separate losses**:
1. **Language Loss**: Cross-entropy on text tokens (e.g., "Waypoints:")
2. **Waypoint Loss**: Smooth L1 loss on continuous waypoint coordinates

**Important**: Waypoint special tokens (`<WAYPOINTS>`, `<WAYPOINTS_DIFF>`, etc.) are **NOT** included in training labels. They are only generated during inference and immediately skipped.

---

## Input Data Structure

### DrivingInput (NamedTuple)

Defined in `simlingo/simlingo_training/utils/custom_types.py`:

```python
class DrivingInput(NamedTuple):
    camera_images: torch.Tensor      # [B, T, N, C, H, W]
    image_sizes: torch.Tensor        # [B, T, N, 2]
    camera_intrinsics: torch.Tensor  # [B, N, 3, 3]
    camera_extrinsics: torch.Tensor  # [B, N, 4, 4]
    vehicle_speed: torch.Tensor      # [B, S]
    target_point: torch.Tensor       # [B, 2]
    prompt: LanguageLabel            # User prompt
    prompt_inference: LanguageLabel  # Inference-only prompt
```

#### Field Specifications

| Field | Shape | Data Type | Range | Description |
|-------|-------|-----------|-------|-------------|
| `camera_images` | `[B, T, N, C, H, W]` | `uint8` | `[0, 255]` | Raw camera images |
| `image_sizes` | `[B, T, N, 2]` | `int64` | - | Original image sizes before padding |
| `camera_intrinsics` | `[B, N, 3, 3]` | `float32` | - | Camera intrinsic matrices (K) |
| `camera_extrinsics` | `[B, N, 4, 4]` | `float32` | - | Camera extrinsic matrices (world→camera) |
| `vehicle_speed` | `[B, S]` | `float32` | `[0, ∞)` m/s | Current vehicle speed |
| `target_point` | `[B, 2]` | `float32` | - | Target point in ego frame [x, y] |
| `prompt` | `LanguageLabel` | - | - | Training prompt with placeholders |
| `prompt_inference` | `LanguageLabel` | - | - | Inference prompt (optional) |

**Dimension Key:**
- `B`: Batch size
- `T`: Temporal history (default: 1 frame)
- `N`: Number of cameras (default: 1)
- `C`: Color channels (3 for RGB)
- `H`: Image height (512 pixels)
- `W`: Image width (1024 pixels)
- `S`: Speed dimension (1)

#### Camera Intrinsics

Camera intrinsic matrix `K` for a camera with width `w`, height `h`, and field of view `fov`:

```python
focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
K = np.array([
    [focal,   0.0,  w/2.0],
    [  0.0, focal,  h/2.0],
    [  0.0,   0.0,    1.0]
], dtype=np.float32)
```

**Default CARLA camera parameters:**
- Width: 1024 pixels
- Height: 512 pixels
- FOV: 100 degrees

#### Camera Extrinsics

Camera extrinsic matrix `[R | t]` in homogeneous form `[4, 4]`:

```python
extrinsics = np.array([
    [R00, R01, R02, tx],
    [R10, R11, R12, ty],
    [R20, R21, R22, tz],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)
```

**Default CARLA camera mounting:**
- Position: `[-1.5, 0.0, 2.0]` (x: forward, y: right, z: up)
- Rotation: `[0.0, 0.0, 0.0]` (roll, pitch, yaw in radians)

For identity rotation (no rotation):
```python
extrinsics = np.array([
    [1.0, 0.0, 0.0, -1.5],
    [0.0, 1.0, 0.0,  0.0],
    [0.0, 0.0, 1.0,  2.0],
    [0.0, 0.0, 0.0,  1.0]
], dtype=np.float32)
```

---

### LanguageLabel (NamedTuple)

```python
class LanguageLabel(NamedTuple):
    phrase_ids: Tensor           # [B, max_len] int64
    phrase_valid: Tensor         # [B, max_len] bool
    phrase_mask: Tensor          # [B, max_len] bool
    placeholder_values: list     # List of dicts mapping token_id → numpy array
    language_string: list        # List of strings
    loss_masking: Tensor         # [B, max_len] bool
```

#### Field Specifications

| Field | Shape | Data Type | Description |
|-------|-------|-----------|-------------|
| `phrase_ids` | `[B, max_len]` | `int64` | Tokenized text (padded to max length) |
| `phrase_valid` | `[B, max_len]` | `bool` | True for valid tokens, False for padding |
| `phrase_mask` | `[B, max_len]` | `bool` | Attention mask (True = attend) |
| `placeholder_values` | `list` | `dict` | Maps special token IDs to embeddings |
| `language_string` | `list` | `str` | Original text strings |
| `loss_masking` | `[B, max_len]` | `bool` | True = compute loss, False = ignore |

**Placeholder Values Format:**
```python
placeholder_values = [
    {
        token_id_1: np.array([[x1, y1], [x2, y2]], dtype=np.float32),  # Target points
        token_id_2: np.array([[x3, y3]], dtype=np.float32),            # Single point
    }
]
```

**Example Prompt:**
```
"Current speed: 5.2 m/s. Target point: <TARGET_POINT> <TARGET_POINT> What should the ego do next?"
```

Where `<TARGET_POINT>` tokens are replaced with actual waypoint embeddings via MLP encoder.

---

## Label Data Structure

### DrivingLabel (NamedTuple)

```python
class DrivingLabel(NamedTuple):
    waypoints: Tensor              # [B, F, 2]
    path: Tensor                   # [B, F, 2]
    answer: LanguageLabel          # Language answer
    image_ff_org: Tensor           # [B, T, N, C, H_org, W_org]
    eval_infos: Optional[Dict]     # Evaluation metadata
```

#### Field Specifications

| Field | Shape | Data Type | Range | Description |
|-------|-------|-----------|-------|-------------|
| `waypoints` | `[B, F, 2]` | `float32` | - | Future waypoints in ego frame [x, y] |
| `path` | `[B, F, 2]` | `float32` | - | Path waypoints in ego frame [x, y] |
| `answer` | `LanguageLabel` | - | - | Ground truth language answer |
| `image_ff_org` | `[B, T, N, C, H, W]` | `uint8` | `[0, 255]` | Original images (before preprocessing) |
| `eval_infos` | `Optional[Dict]` | - | - | Metadata for evaluation |

**Dimension Key:**
- `F`: Number of future waypoints (default: 11)

#### Waypoints Format

**Speed Waypoints** (`waypoints` field):
- **Number of points**: 11
- **Temporal spacing**: 0.2 seconds apart
- **Total horizon**: 2.2 seconds into the future
- **Coordinate frame**: Ego vehicle frame (origin at vehicle center)
- **Units**: Meters

**Path Waypoints** (`path` field):
- **Number of points**: 20
- **Spatial spacing**: 1.0 meter apart
- **Total distance**: 20 meters ahead
- **Coordinate frame**: Ego vehicle frame
- **Units**: Meters

**Example:**
```python
waypoints = np.array([
    [0.5, 0.0],   # 0.2s: 0.5m forward, 0m lateral
    [1.0, 0.0],   # 0.4s: 1.0m forward, 0m lateral
    [1.5, 0.1],   # 0.6s: 1.5m forward, 0.1m lateral (slight right turn)
    ...
], dtype=np.float32)  # Shape: [11, 2]
```

#### Language Answer Format

The `answer` field contains **plain text only** - NO waypoint tokens:

```python
# Commentary task
answer = "The vehicle should slow down because there is a red traffic light ahead. Waypoints:"

# Q&A task
answer = "A: The traffic light is red."

# Driving task (no commentary)
answer = "Waypoints:"
```

**Important**: The text "Waypoints:" is just a string - it does NOT include special tokens like `<WAYPOINTS>`.

---

## Coordinate Frames and Transformations

### Ego Vehicle Frame

**Origin**: Center of the vehicle at ground level (between front and rear axles)

**Axes**:
- **X-axis**: Forward (positive = ahead of vehicle)
- **Y-axis**: Left (positive = left of vehicle)
- **Z-axis**: Up (positive = above ground)

**Note**: SimLingo uses **Bird's Eye View (BEV)** for waypoints, so Z-coordinate is dropped.

### World to Ego Transformation

From `dataset_base.py` lines 785-811:

```python
def world_to_ego(world_point, ego_matrix):
    """
    Transform world coordinates to ego vehicle frame.
    
    Args:
        world_point: Point in world coordinates [x, y, z]
        ego_matrix: 4x4 transformation matrix [R | t; 0 1]
    
    Returns:
        Point in ego frame [x, y] (BEV, Z dropped)
    """
    # Extract rotation and translation from ego_matrix
    origin_matrix = ego_matrix[:3]  # [3, 4]
    origin_translation = origin_matrix[:, 3:4]  # [3, 1]
    origin_rotation = origin_matrix[:, :3]  # [3, 3]
    
    # Transform: R^T @ (point - translation)
    world_point_col = world_point.reshape(3, 1)
    ego_point_3d = origin_rotation.T @ (world_point_col - origin_translation)
    
    # Drop Z-coordinate for BEV
    ego_point_2d = ego_point_3d[:2, 0]  # [x, y]
    
    return ego_point_2d
```

### Data Augmentation

From `dataset_base.py` lines 799-811:

```python
def augment_waypoints(waypoints, y_translation=0.0, yaw_rotation_deg=0.0):
    """
    Apply data augmentation to waypoints.
    
    Args:
        waypoints: List of [x, y] waypoints in ego frame
        y_translation: Lateral offset in meters (positive = left)
        yaw_rotation_deg: Rotation in degrees (positive = counterclockwise)
    
    Returns:
        Augmented waypoints
    """
    yaw_rad = np.deg2rad(yaw_rotation_deg)
    rotation_matrix = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad),  np.cos(yaw_rad)]
    ])
    
    translation = np.array([[0.0], [y_translation]])
    
    waypoints_aug = []
    for waypoint in waypoints:
        pos = waypoint.reshape(2, 1)
        waypoint_aug = rotation_matrix.T @ (pos - translation)
        waypoints_aug.append(waypoint_aug.flatten())
    
    return np.array(waypoints_aug, dtype=np.float32)
```

**Typical augmentation ranges:**
- `y_translation`: [-0.5, 0.5] meters
- `yaw_rotation_deg`: [-5, 5] degrees

---

## Example Code

### Constructing a Single Training Sample

```python
import numpy as np
import torch
from PIL import Image

def create_training_sample():
    """
    Create one valid SimLingo training sample.
    
    Returns:
        Tuple of (DrivingInput, DrivingLabel)
    """
    batch_size = 1
    
    # ========== Load and preprocess image ==========
    # Load image (e.g., from QCar2 camera)
    image_path = "data/frame_0000.jpg"
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 512))  # Resize to SimLingo resolution
    image_array = np.array(image, dtype=np.uint8)  # [H, W, C]
    
    # Add batch and temporal dimensions: [B, T, N, C, H, W]
    camera_images = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    # Image sizes (original size before padding)
    image_sizes = torch.tensor([[[1024, 512]]], dtype=torch.int64)  # [B, T, N, 2]
    
    # ========== Camera parameters ==========
    # Intrinsics (CARLA default: 1024x512, FOV=100°)
    fov = 100.0
    w, h = 1024, 512
    focal = w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.array([
        [focal,   0.0,  w/2.0],
        [  0.0, focal,  h/2.0],
        [  0.0,   0.0,    1.0]
    ], dtype=np.float32)
    camera_intrinsics = torch.from_numpy(K).unsqueeze(0).unsqueeze(0)  # [B, N, 3, 3]
    
    # Extrinsics (CARLA default: [-1.5, 0.0, 2.0], no rotation)
    extrinsics = np.array([
        [1.0, 0.0, 0.0, -1.5],
        [0.0, 1.0, 0.0,  0.0],
        [0.0, 0.0, 1.0,  2.0],
        [0.0, 0.0, 0.0,  1.0]
    ], dtype=np.float32)
    camera_extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).unsqueeze(0)  # [B, N, 4, 4]
    
    # ========== Vehicle state ==========
    vehicle_speed = torch.tensor([[5.2]], dtype=torch.float32)  # [B, S] in m/s
    
    # ========== Target point (in ego frame) ==========
    target_point = torch.tensor([[10.0, 2.0]], dtype=torch.float32)  # [B, 2]
    
    # ========== Language prompt ==========
    # Tokenize prompt (simplified - use actual tokenizer in practice)
    prompt_text = "Current speed: 5.2 m/s. Target point: <TARGET_POINT> <TARGET_POINT> What should the ego do next?"
    # ... (tokenization code omitted for brevity)
    
    # ========== Ground truth waypoints ==========
    # Speed waypoints: 11 points, 0.2s apart
    waypoints = np.array([
        [1.0, 0.0],
        [2.0, 0.1],
        [3.0, 0.2],
        [4.0, 0.3],
        [5.0, 0.4],
        [6.0, 0.5],
        [7.0, 0.6],
        [8.0, 0.7],
        [9.0, 0.8],
        [10.0, 0.9],
        [11.0, 1.0]
    ], dtype=np.float32)
    waypoints_tensor = torch.from_numpy(waypoints).unsqueeze(0)  # [B, F, 2]
    
    # Path waypoints: 20 points, 1m apart
    path = np.array([
        [i, i * 0.1] for i in range(1, 21)
    ], dtype=np.float32)
    path_tensor = torch.from_numpy(path).unsqueeze(0)  # [B, F, 2]
    
    # ========== Language answer ==========
    answer_text = "The vehicle should continue straight. Waypoints:"
    # ... (tokenization code omitted for brevity)
    
    return driving_input, driving_label

# Usage
input_data, label_data = create_training_sample()
```

### Computing Waypoints from Measurements

```python
def compute_waypoints_from_trajectory(measurements, current_idx=0):
    """
    Compute waypoints from a sequence of vehicle measurements.
    
    Args:
        measurements: List of dicts with 'ego_matrix' field (4x4 transformation)
        current_idx: Index of current frame (default: 0)
    
    Returns:
        waypoints: [F, 2] array of waypoints in ego frame
    """
    # Extract current ego matrix
    origin = measurements[current_idx]
    origin_matrix = np.array(origin['ego_matrix'])[:3]  # [3, 4]
    origin_translation = origin_matrix[:, 3:4]  # [3, 1]
    origin_rotation = origin_matrix[:, :3]  # [3, 3]
    
    # Transform future positions to ego frame
    waypoints = []
    for i in range(current_idx + 1, len(measurements)):
        future_pos = np.array(measurements[i]['ego_matrix'])[:3, 3:4]  # [3, 1]
        ego_pos = origin_rotation.T @ (future_pos - origin_translation)
        waypoints.append(ego_pos[:2, 0])  # Drop Z, keep [x, y]
    
    return np.array(waypoints, dtype=np.float32)
```

---

## Summary Checklist

For each training sample, you need:

**Input Data:**
- ✅ Camera image: `[1, 1, 1, 3, 512, 1024]` uint8
- ✅ Image size: `[1, 1, 1, 2]` int64
- ✅ Camera intrinsics: `[1, 1, 3, 3]` float32
- ✅ Camera extrinsics: `[1, 1, 4, 4]` float32
- ✅ Vehicle speed: `[1, 1]` float32 (m/s)
- ✅ Target point: `[1, 2]` float32 (ego frame)
- ✅ Prompt: Tokenized text with placeholders

**Label Data:**
- ✅ Waypoints: `[1, 11, 2]` float32 (ego frame, 0.2s spacing)
- ✅ Path: `[1, 20, 2]` float32 (ego frame, 1m spacing)
- ✅ Answer: Plain text (e.g., "Waypoints:" or commentary)

**Coordinate Frame:**
- ✅ All waypoints in ego vehicle frame (X=forward, Y=left)
- ✅ Origin at vehicle center, ground level
- ✅ BEV representation (Z-coordinate dropped)

---

**End of Specification**

