# Route Manager Explained

## Overview

The `RouteManager` class manages route waypoints and provides target points to the SimLingo model. It handles three key responsibilities:
1. **Lookahead algorithm** - Selects target waypoints ahead of the vehicle
2. **Coordinate transformation** - Converts world coordinates to ego vehicle frame
3. **Progress tracking** - Monitors route completion

## 1. Lookahead Algorithm

The lookahead algorithm selects target waypoints at a fixed distance ahead of the vehicle to guide navigation.

### How It Works

**Step 1: Update Current Progress** (lines 42-58)
```python
# Find nearest waypoint AHEAD of current position
search_start = self.current_waypoint_index
search_end = min(self.current_waypoint_index + 10, len(self.route_waypoints))

distances_ahead = np.linalg.norm(
    self.route_waypoints[search_start:search_end, :2] - current_position[:2],
    axis=1
)
nearest_idx = search_start + np.argmin(distances_ahead)

# Update current waypoint index (only move forward)
if nearest_idx > self.current_waypoint_index:
    self.current_waypoint_index = nearest_idx
```
- Searches only forward (next 10 waypoints) to avoid jumping backwards
- Critical for routes with loops or parallel sections
- Updates progress marker (`current_waypoint_index`)

**Step 2: Find Target Based on Lookahead Distance** (lines 60-82)
```python
target_idx = self.current_waypoint_index
accumulated_distance = 0.0

# Start from current position to first waypoint
accumulated_distance = np.linalg.norm(
    self.route_waypoints[self.current_waypoint_index, :2] - current_position[:2]
)

# Accumulate distance along route segments
for i in range(self.current_waypoint_index, len(self.route_waypoints) - 1):
    if accumulated_distance >= self.lookahead_distance:
        target_idx = i
        break
    
    segment_distance = np.linalg.norm(
        self.route_waypoints[i + 1, :2] - self.route_waypoints[i, :2]
    )
    accumulated_distance += segment_distance
    target_idx = i + 1
```
- Accumulates distance along route segments (not straight-line distance)
- Stops when accumulated distance ≥ lookahead distance (default: 7.5m)
- Returns waypoint at that distance ahead

**Step 3: Get Next Target** (lines 87-89)
```python
next_target_idx = min(target_idx + 1, len(self.route_waypoints) - 1)
next_target_point = self.route_waypoints[next_target_idx]
```
- Provides second target point (one waypoint ahead of first)
- Used by model to understand route curvature

### Configuration
- `target_point_lookahead = 7.5` meters (from `config.py`)
- Matches SimLingo's default lookahead distance
- Balances between smooth following and sharp turns

## 2. World-to-Ego Frame Conversion

Converts global waypoints to vehicle-centric coordinates for the model.

### Coordinate Systems

**World Frame (QLabs)**
- Origin: QLabs world origin
- X-axis: East
- Y-axis: North
- Z-axis: Up

**Ego Frame (Vehicle)**
- Origin: Vehicle center
- X-axis: Forward (vehicle heading)
- Y-axis: Left
- Z-axis: Up

### Transformation Math (lines 129-152)

```python
def _world_to_ego(self, world_point, vehicle_pos, vehicle_heading):
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
        [np.sin(vehicle_heading), np.cos(vehicle_heading)]
    ])
    
    # Apply transformation: R^T @ (point - translation)
    ego_point = rotation_matrix.T @ (world_point - vehicle_pos)
    
    return ego_point
```

**Mathematical Steps:**
1. **Translation**: Subtract vehicle position from world point
   - `point - vehicle_pos` → Point relative to vehicle position
2. **Rotation**: Apply inverse rotation to align with vehicle heading
   - `R^T @ (...)` → Rotate to vehicle's coordinate frame
   - `R^T` is transpose (inverse for rotation matrices)

**Example:**
- Vehicle at world position `[10, 5]` with heading `90°` (facing North)
- Target waypoint at world position `[10, 15]`
- Translation: `[10, 15] - [10, 5] = [0, 10]` (10m North)
- Rotation by -90°: `[0, 10]` → `[10, 0]` (10m forward in ego frame)

### SimLingo Compatibility

This matches SimLingo's `inverse_conversion_2d` function from `transfuser_utils.py`:
```python
# Original SimLingo implementation
def inverse_conversion_2d(point, translation, yaw):
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    return rotation_matrix.T @ (point - translation)
```

Our implementation is identical to ensure model compatibility.

## 3. Pipeline Integration

### Where Route Manager is Used

**Main Control Loop** (`src/main.py`, lines 139-151)
```python
# Step 1: Get target points in ego frame
target_point, next_target_point = self.route_manager.get_target_point_ego(
    current_position, current_heading
)

# Step 2: Pass to model inference
speed_wps, route_wps, language = self.model_wrapper.inference(
    camera_images=camera_images,
    image_sizes=image_sizes,
    camera_intrinsics=camera_intrinsics,
    camera_extrinsics=camera_extrinsics,
    vehicle_speed=velocity,
    target_point=target_point,        # Ego frame [x, y]
    next_target_point=next_target_point  # Ego frame [x, y]
)
```

### Model Input Format

**Prompt Template** (`src/simlingo_model.py`, lines 356-363)
```python
# Default prompt
prompt = "Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"

# Target points array
target_points = np.array([target_point, next_target_point], dtype=np.float32)
# Shape: [2, 2] = [[x1, y1], [x2, y2]]
```

**Placeholder Replacement** (`src/simlingo_model.py`, lines 253-258)
```python
placeholder_values = {
    '<TARGET_POINT>': target_points,  # [2, 2] array
    # Each <TARGET_POINT> token in prompt is replaced with waypoint embedding
}
```

### How the Model Uses Target Points

**The `<TARGET_POINT>` string is NOT replaced with numbers before inference. Instead:**

1. **Tokenization**: Prompt string with `<TARGET_POINT>` placeholders is tokenized
   - Example: `"Target waypoint: <TARGET_POINT><TARGET_POINT>."`
   - Each `<TARGET_POINT>` becomes a special token ID (e.g., token ID 92547)
   - Result: Token sequence like `[..., 92547, 92547, ...]`

2. **Initial Embedding**: Tokens are converted to embeddings using language model's embedding layer
   - Each token ID → 2048-dimensional embedding vector
   - At this point, `<TARGET_POINT>` tokens have generic embeddings (not meaningful yet)

3. **Placeholder Replacement** (inside model during forward pass):
   - `replace_placeholder_tokens()` function finds all `<TARGET_POINT>` token positions
   - For each occurrence, it replaces the generic embedding with a learned waypoint embedding
   - Uses `WaypointInputAdaptor` (MLP) to convert [x, y] coordinates → 2048-dim embedding
   - Code: `wp_embeds = wp_encoder(coords)` where coords = `[[x1, y1], [x2, y2]]`
   - These waypoint embeddings are injected into the sequence at `<TARGET_POINT>` positions

4. **Attention**: Language model processes the sequence with waypoint embeddings
   - Attends to vision tokens, text tokens, AND waypoint embeddings together
   - Model learns to condition trajectory prediction on target waypoint locations

5. **Prediction**: Model outputs trajectory waypoints conditioned on target points

**Key Point**: The actual coordinate values `[x, y]` are passed separately in `placeholder_values` dictionary and converted to embeddings INSIDE the model, not before. The prompt string keeps the `<TARGET_POINT>` tokens as placeholders.

### Pipeline Stage Summary

```
Vehicle State → Route Manager → Model Input → Model Inference → Control Output
     ↓              ↓                ↓              ↓                ↓
[position,    [target_ego,    [DrivingInput    [waypoints,    [steer,
 heading]      next_ego]       with targets]     language]      throttle]
```

**Stage 1: Before Model** (Route Manager)
- Input: Vehicle position/heading (world frame)
- Output: Target points (ego frame)
- Purpose: Provide navigation goal to model

**Stage 2: Model Inference** (SimLingo)
- Input: Camera + target points (ego frame)
- Output: Predicted waypoints (ego frame)
- Purpose: Plan trajectory toward target

**Stage 3: Control** (PID Controllers)
- Input: Predicted waypoints (ego frame)
- Output: Steering/throttle commands
- Purpose: Follow predicted trajectory

## Key Design Decisions

1. **Forward-only search**: Prevents backtracking on complex routes
2. **Path distance vs straight-line**: Follows road curvature accurately
3. **Fixed lookahead**: Matches SimLingo training data (7.5m)
4. **Ego frame conversion**: Required by model (trained on ego-centric data)
5. **Two target points**: Helps model understand route direction and curvature

## Comparison to SimLingo Original

**SimLingo (CARLA)**
- Uses CARLA's `GlobalRoutePlanner` to get waypoints
- Route planner provides waypoints with turn commands
- Converts to ego frame using `inverse_conversion_2d`

**Our Implementation (QCar2)**
- Uses predefined route waypoints (no dynamic planner)
- Implements same lookahead logic manually
- Uses identical ego frame conversion math
- Same target point format and usage in model

The main difference is **route source** (CARLA planner vs predefined), but the **lookahead algorithm** and **coordinate transformation** are functionally equivalent.

