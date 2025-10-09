# SimLingo Model Integration with QLabs QCar2: Technical Analysis

**Document Version:** 1.0  
**Date:** 2025-10-09  
**Project:** Qcar2SimDev

---

## Executive Summary

This document provides a comprehensive technical analysis of the SimLingo vision-language-action (VLA) model integration with the Quanser QCar2 vehicle in QLabs simulation environment. The analysis traces the complete data flow from sensor acquisition through model inference to vehicle control, answering four critical questions about input formats, model architecture, output structure, and control integration.

---

## 1. Input Data Flow & Format

### 1.1 Camera Image Acquisition and Preprocessing

#### **Source: QCar2 CSI Front Camera**
- **Location:** `src/qcar2_interface.py:172-201`
- **Camera ID:** `CAMERA_CSI_FRONT` (camera=3)
- **Native Resolution:** 820×410 pixels
- **Raw Format:** BGR uint8 from QLabs API
- **Initial Conversion:** BGR → RGB (`cv2.cvtColor`, line 193)

#### **Preprocessing Pipeline** (`src/camera_processor.py:42-95`)

**Step 1: JPEG Compression/Decompression** (lines 54-62)
```python
# Simulates CARLA training data compression artifacts
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
_, compressed_image = cv2.imencode('.jpg', image_bgr)
image_bgr = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```
- **Purpose:** Match training data characteristics from CARLA simulator
- **Data Type:** uint8 [0, 255]

**Step 2: Bottom Crop** (lines 64-70)
```python
crop_height = int(image.shape[0] - (image.shape[0] * 1.6) // 16)  # 10% crop
image = image[:crop_height, :, :]
```
- **Crop Amount:** 10% of bottom removed (conservative vs. 30% in CARLA)
- **Rationale:** QCar2 camera doesn't show vehicle hood
- **Output Shape:** ~[369, 820, 3]

**Step 3: Dynamic Preprocessing** (lines 72-82)
```python
from simlingo_training.utils.internvl2_utils import dynamic_preprocess
images = dynamic_preprocess(
    pil_image,
    image_size=448,
    use_thumbnail=False,  # use_global_img=False
    max_num=2
)
```
- **Function:** Splits image into patches based on aspect ratio
- **Patch Size:** 448×448 pixels
- **Number of Patches:** Typically 2 for QCar2's aspect ratio
- **Output:** List of PIL Images

**Step 4: InternVL2 Transform** (lines 84-90)
```python
from simlingo_training.utils.internvl2_utils import build_transform
transform = build_transform(input_size=448)
pixel_values = [transform(img) for img in images]
pixel_values = torch.stack(pixel_values)
pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)
```
- **Transform Operations:**
  * Resize to 448×448
  * Convert to tensor
  * ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Final Shape:** `[1, 1, num_patches, 3, 448, 448]`
  * Dimension 0: Batch size (always 1)
  * Dimension 1: Temporal frames (always 1)
  * Dimension 2: Number of patches (typically 2)
  * Dimensions 3-5: Channels, Height, Width
- **Data Type:** `torch.float32` → converted to `bfloat16` before model input (line 349)

#### **Camera Intrinsics** (`src/config.py:217-251`)
- **Computation:** `src/camera_processor.py:97-107`
- **FOV:** 160° (matches QCar2 CSI camera specification)
- **Resolution:** 1024×512 (target after preprocessing)
- **Focal Length:** `f = width / (2 * tan(fov/2)) = 1024 / (2 * tan(80°)) ≈ 89.7`
- **Principal Point:** `(cx, cy) = (512, 256)` (image center)
- **Matrix Shape:** `[1, 3, 3]` float32
```
[[  89.7,   0.0, 512.0],
 [   0.0,  89.7, 256.0],
 [   0.0,   0.0,   1.0]]
```

#### **Camera Extrinsics** (`src/config.py:253-269`)
- **Computation:** `src/camera_processor.py:109-119`
- **Camera Position:** `[-1.5, 0.0, 2.0]` meters (x=forward, y=right, z=up)
- **Camera Rotation:** `[0.0, 0.0, 0.0]` radians (roll, pitch, yaw)
- **Matrix Shape:** `[1, 4, 4]` float32
```
[[ 1.0,  0.0,  0.0, -1.5],
 [ 0.0,  1.0,  0.0,  0.0],
 [ 0.0,  0.0,  1.0,  2.0],
 [ 0.0,  0.0,  0.0,  1.0]]
```

### 1.2 GPS Target Points (TP)

#### **Source: Route Manager** (`src/route_manager.py:93-127`)

**Step 1: World Frame Target Selection** (lines 31-91)
- **Route Waypoints:** 35 waypoints in QLabs global coordinates
- **Lookahead Distance:** 5.0 meters (configurable)
- **Selection Algorithm:**
  1. Find nearest waypoint ahead of vehicle (lines 42-58)
  2. Accumulate distance along route until lookahead reached (lines 60-79)
  3. Select target_point and next_target_point (lines 84-89)
- **Output:** Two waypoints `[x, y, z]` in world frame

**Step 2: World-to-Ego Transformation** (lines 129-152)
```python
def _world_to_ego(world_point, vehicle_pos, vehicle_heading):
    rotation_matrix = np.array([
        [np.cos(vehicle_heading), -np.sin(vehicle_heading)],
        [np.sin(vehicle_heading),  np.cos(vehicle_heading)]
    ])
    ego_point = rotation_matrix.T @ (world_point - vehicle_pos)
    return ego_point
```
- **Input:** World coordinates `[x, y]`, vehicle position `[x, y]`, heading (radians)
- **Output:** Ego frame coordinates `[x, y]`
- **Coordinate System:**
  * Ego X-axis: Forward direction
  * Ego Y-axis: Left direction (perpendicular to forward)
- **Data Type:** `np.float32`
- **Shape:** Two arrays of `[2]` each

#### **Integration into Model Input** (`src/simlingo_model.py:191-290`)

**Step 1: Create Target Points Array** (line 330)
```python
target_points = np.array([target_point, next_target_point], dtype=np.float32)
# Shape: [2, 2] - two points, each with [x, y]
```

**Step 2: Embed in Language Prompt** (lines 191-290)
- **Placeholder Token:** `<TARGET_POINT>`
- **Prompt Template:** `"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"`
- **Placeholder Values Dictionary:**
```python
placeholder_values = {
    '<TARGET_POINT>': target_points,  # [2, 2] array
    '<img>': np.array([]),
    '</img>': np.array([]),
    '<IMG_CONTEXT>': np.array([])
}
```

**Step 3: Waypoint Encoding** (`simlingo/simlingo_training/models/encoder/internvl2_model.py:60-91`)
- **Encoder:** `WaypointInputAdaptor` MLP
- **Architecture:** Linear(2→256) → ReLU → Linear(256→512) → ReLU → Linear(512→hidden_size)
- **Hidden Size:** 2048 (Qwen2-0.5B embedding dimension)
- **Process:**
  1. Extract target points from placeholder_values (line 80)
  2. Pass through MLP encoder (line 83)
  3. Replace `<TARGET_POINT>` tokens in embedding sequence (line 91)
- **Output:** Embedded waypoints integrated into language token sequence

### 1.3 Language Prompt (pglobal)

#### **Prompt Construction** (`src/config.py:271-290`)
```python
def get_prompt_template(speed, use_cot=True):
    if use_cot:
        prompt = f"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
    else:
        prompt = f"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints."
```
- **Chain-of-Thought (CoT):** Enabled by default (`use_cot=True`)
- **Speed Format:** Float with 2 decimal places (m/s)
- **Placeholder Count:** 2× `<TARGET_POINT>` (for target and next_target)

#### **Image Token Injection** (`src/simlingo_model.py:203-219`)
```python
# Calculate number of image tokens per patch
image_size = 448
patch_size = 14
downsample_ratio = 0.5
num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
# = (448/14)^2 * 0.5^2 = 32^2 * 0.25 = 256 tokens per patch

# Create image token string
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
# For 2 patches: '<img>' + 512× '<IMG_CONTEXT>' + '</img>'

# Prepend to prompt
prompt_with_image = f"<image>\n{prompt}"
prompt_with_image = prompt_with_image.replace('<image>', image_tokens, 1)
```

#### **Tokenization** (`src/simlingo_model.py:254-271`)
```python
tokenized = self.tokenizer(
    [prompt_with_image],
    padding=True,
    return_tensors="pt",
    add_special_tokens=False
)
language_label = LanguageLabel(
    phrase_ids=tokenized['input_ids'].to(device),        # [1, seq_len] int64
    phrase_valid=tokenized['attention_mask'].bool(),     # [1, seq_len] bool
    phrase_mask=tokenized['attention_mask'].bool(),      # [1, seq_len] bool
    placeholder_values=[placeholder_values_dict],        # List[Dict]
    language_string=[prompt_with_image],                 # List[str]
    loss_masking=None
)
```
- **Tokenizer:** Qwen2-0.5B tokenizer (from InternVL2-1B)
- **Sequence Length:** ~520-540 tokens (varies with speed value)
  * Image tokens: ~512
  * Text tokens: ~20-30
- **Data Type:** `torch.int64` for token IDs, `torch.bool` for masks

### 1.4 Vehicle Speed

#### **Source: State Estimator** (`src/state_estimator.py:42-98`)

**Velocity Calculation** (lines 66-87)
```python
# Calculate from position changes
pos_prev = self.position_history[-2]
pos_curr = self.position_history[-1]
time_prev = self.time_history[-2]
time_curr = self.time_history[-1]

displacement = pos_curr - pos_prev
dt = time_curr - time_prev

speed = np.linalg.norm(displacement[:2]) / dt  # Use only x, y
self.velocity_history.append(speed)

# Moving average filter
self.velocity = np.mean(self.velocity_history)
```
- **Method:** Finite difference of position
- **Filter:** 5-sample moving average
- **Units:** meters/second
- **Data Type:** `float` (Python native)

**Integration into Model** (`src/simlingo_model.py:340`)
```python
speed_tensor = torch.tensor([[vehicle_speed]], dtype=torch.float32).to(device)
# Shape: [1, 1]
```

### 1.5 High-Level Commands (HLC)

**Status:** NOT USED in current implementation
- **Configuration:** `eval_route_as = 'target_point'` (not 'command')
- **Rationale:** SimLingo model uses target points for navigation, not discrete commands
- **Alternative:** Language prompt provides context instead of HLC

---

## 2. Model Architecture & Processing Flow

### 2.1 Model Overview

**Architecture:** Vision-Language-Action (VLA) Model  
**Base Models:**
- **Vision Encoder:** InternVL2-1B (InternViT-300M)
- **Language Model:** Qwen2-0.5B with LoRA
- **Configuration:** `models/simlingo/.hydra/config.yaml`

### 2.2 Forward Pass Stages

#### **Stage 1: Input Preparation** (`src/simlingo_model.py:292-357`)

**DrivingInput Construction** (lines 348-357)
```python
driving_input = DrivingInput(
    camera_images=camera_images.to(device).bfloat16(),    # [1, 1, 2, 3, 448, 448]
    image_sizes=None,                                      # Not used by InternVL2
    camera_intrinsics=camera_intrinsics.to(device),        # [1, 3, 3]
    camera_extrinsics=camera_extrinsics.to(device),        # [1, 4, 4]
    vehicle_speed=speed_tensor,                            # [1, 1]
    target_point=target_point_tensor,                      # [1, 2]
    prompt=language_label,                                 # LanguageLabel
    prompt_inference=language_label                        # LanguageLabel
)
```

#### **Stage 2: Adaptor Processing** (`simlingo/simlingo_training/models/driving.py:104-125`)

**2a. Language Adaptor** (`simlingo/simlingo_training/models/adaptors/adaptors.py:238-257`)
```python
# Embed tokenized prompt
ids = label.phrase_ids.long()  # [1, seq_len]
inputs = self.embed_tokens(ids.clamp(min=0, max=num_embeddings - 1))
# Output: [1, seq_len, 2048] where 2048 is Qwen2-0.5B hidden_size
```
- **Embedding Layer:** Qwen2-0.5B's `model.embed_tokens`
- **Dimension:** 2048 (hidden_size)

**2b. Driving Adaptor** (`simlingo/simlingo_training/models/adaptors/adaptors.py:96-161`)
```python
# Create learnable query embeddings
self.query_embeds_wps = nn.Parameter(torch.randn(1, 20, 2048))     # Route queries
self.query_embeds_speed = nn.Parameter(torch.randn(1, 10, 2048))   # Speed queries

# Concatenate queries
inputs = torch.cat([
    self.query_embeds_wps.expand(batch_size, -1, -1),    # [1, 20, 2048]
    self.query_embeds_speed.expand(batch_size, -1, -1)   # [1, 10, 2048]
], dim=1)
# Output: [1, 30, 2048]
```
- **Route Queries:** 20 learnable embeddings for geometric path waypoints
- **Speed Queries:** 10 learnable embeddings for temporal speed waypoints
- **Total:** 30 query tokens

#### **Stage 3: Placeholder Token Replacement** (`simlingo/simlingo_training/models/encoder/internvl2_model.py:17-144`)

**3a. Target Point Embedding** (lines 60-91)
```python
# Find <TARGET_POINT> tokens in sequence
smallest_added_id = tokenizer.additional_special_tokens_ids[0]
special_ids = input_ids[input_ids >= smallest_added_id]

# Encode target points via WaypointInputAdaptor
coords = torch.tensor(placeholder_values[batch_id][special_id])  # [2, 2]
wp_embeds = wp_encoder(coords.unsqueeze(0)).squeeze(0)           # [2, 2048]

# Replace placeholder tokens with embeddings
inputs_embeds[batch_id, start:end] = wp_embeds
```
- **WaypointInputAdaptor:** MLP (2→256→512→2048)
- **Input:** Target points `[2, 2]` (two points, x-y coordinates)
- **Output:** Embedded waypoints `[2, 2048]`

**3b. Vision Feature Extraction** (lines 94-132)
```python
# Extract features from image patches
BS, T, NP, C, H, W = pixel_values.shape  # [1, 1, 2, 3, 448, 448]
pixel_values = pixel_values.view(BS, NP, C, H, W).reshape(BS*NP, C, H, W)  # [2, 3, 448, 448]

image_features = self.model.extract_feature(pixel_values)  # InternViT-300M
# Output: [2, 256, 2048] where 256 = num_image_tokens_per_patch

vit_embeds = image_features.reshape(-1, 2048)  # [512, 2048]

# Replace <IMG_CONTEXT> tokens with vision features
selected = (input_ids == img_context_token_id)
inputs_embeds[selected] = vit_embeds
```
- **Vision Encoder:** InternViT-300M (part of InternVL2-1B)
- **Patch Processing:** Each 448×448 patch → 256 tokens
- **Total Vision Tokens:** 2 patches × 256 = 512 tokens
- **Feature Dimension:** 2048

**3c. Token Sequence Assembly**
```
Final embedding sequence:
[<img>] [256× vision_tokens_patch1] [256× vision_tokens_patch2] [</img>] 
[text_tokens] [2× target_point_embeds] [text_tokens]
[20× route_queries] [10× speed_queries]

Total length: ~1 + 512 + 1 + 20 + 2 + 5 + 30 = ~571 tokens
```

#### **Stage 4: Language Model Forward Pass** (`simlingo/simlingo_training/models/driving.py:190-233`)

**4a. Qwen2-0.5B Processing** (lines 217-225)
```python
outputs = self.language_model.model(
    attention_mask=attention_mask,      # [1, 571]
    position_ids=None,
    inputs_embeds=input_embeds,         # [1, 571, 2048]
    output_hidden_states=True,
    return_dict=True,
)
features = outputs.hidden_states[-1]    # [1, 571, 2048]
logits = outputs[0]                     # [1, 571, vocab_size]
```
- **Model:** Qwen2-0.5B transformer with LoRA
- **LoRA Config:** rank=32, alpha=64, dropout=0.1
- **Layers:** 24 transformer blocks
- **Attention:** Multi-head self-attention (16 heads, 128 dim each)
- **FFN:** 2048 → 11008 → 2048 (SwiGLU activation)

**4b. Feature Extraction** (lines 227-232)
```python
# Split features by adaptor
vision_features, adaptor_features = features.split(
    [features.size(1) - adaptor_embeds.size(1), adaptor_embeds.size(1)], 
    dim=1
)
# adaptor_features: [1, 30, 2048] (last 30 tokens = driving queries)
```

#### **Stage 5: Prediction Heads** (`simlingo/simlingo_training/models/adaptors/adaptors.py:163-180`)

**5a. Route Waypoint Prediction** (lines 110-120)
```python
# Route head MLP
self.route_head = nn.Sequential(
    nn.Linear(2048, 512), nn.SiLU(),
    nn.Linear(512, 256), nn.SiLU(),
    nn.Linear(256, 2, bias=False)
)

# Apply to route query features
route_features = adaptor_features[:, 0:20]      # [1, 20, 2048]
route_deltas = self.route_head(route_features)  # [1, 20, 2]
route_wps = route_deltas.cumsum(dim=1)          # [1, 20, 2] cumulative sum
```
- **Input:** 20 route query features `[1, 20, 2048]`
- **MLP:** 2048 → 512 → 256 → 2
- **Output:** Cumulative waypoints `[1, 20, 2]` in ego frame

**5b. Speed Waypoint Prediction** (lines 122-136)
```python
# Speed head MLP
self.speed_wps_head = nn.Sequential(
    nn.Linear(2048, 256), nn.SiLU(),
    nn.Linear(256, 2, bias=False)
)

# Apply to speed query features
speed_features = adaptor_features[:, 20:30]      # [1, 10, 2048]
speed_deltas = self.speed_wps_head(speed_features)  # [1, 10, 2]
speed_wps = speed_deltas.cumsum(dim=1)           # [1, 10, 2] cumulative sum
```
- **Input:** 10 speed query features `[1, 10, 2048]`
- **MLP:** 2048 → 256 → 2
- **Output:** Cumulative waypoints `[1, 10, 2]` in ego frame

**5c. Language Generation** (lines 143-176)
```python
# Greedy sampling from language model
sampled_tokens, input_embeds = self.language_model.greedy_sample(
    input_embed,
    eos_token_id=eos,
    max_new_tokens=100,
    input_embed_matrix=self.embed_tokens.weight,
    logit_matrix=self.lm_head.weight,
    attention_mask=attention_mask,
)
language = self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=True)[0]
```
- **Method:** Greedy decoding (argmax at each step)
- **Max Length:** 100 new tokens
- **Output:** String describing driving action

### 2.3 Data Flow Summary

```
QCar2 Camera → [820×410 BGR] → JPEG → Crop → Dynamic Preprocess → [2×448×448 RGB]
                                                                         ↓
                                                              InternVL2 Transform
                                                                         ↓
                                                              [1,1,2,3,448,448] bfloat16
                                                                         ↓
Route Manager → World Waypoints → Ego Transform → [2,2] float32 → WaypointEncoder
                                                                         ↓
                                                              [2, 2048] embeddings
                                                                         ↓
Tokenizer → Prompt + Image Tokens → [1, ~540] token_ids → Embed → [1, ~540, 2048]
                                                                         ↓
                                                              Replace Placeholders
                                                                         ↓
                                    [Vision: 512 tokens] + [Text: ~30 tokens] + [Waypoints: 2 tokens]
                                                                         ↓
Driving Adaptor → [Route Queries: 20] + [Speed Queries: 10] → [1, 30, 2048]
                                                                         ↓
                                                    Concatenate All Embeddings
                                                                         ↓
                                                    [1, ~571, 2048] → Qwen2-0.5B
                                                                         ↓
                                                    [1, ~571, 2048] features
                                                                         ↓
                                    Extract Last 30 Tokens (Driving Queries)
                                                                         ↓
                                    [Route: 20 tokens] → Route MLP → [1, 20, 2]
                                    [Speed: 10 tokens] → Speed MLP → [1, 10, 2]
```

---

## 3. Model Output Format & Structure

### 3.1 Output Tensor Specifications

#### **Route Waypoints (Geometric Path)**
- **Variable Name:** `route_wps` (`src/simlingo_model.py:369`)
- **Shape:** `[1, 20, 2]`
  * Dimension 0: Batch size (always 1)
  * Dimension 1: Number of waypoints (20)
  * Dimension 2: Coordinates (x, y)
- **Data Type:** `torch.bfloat16` (model output) → `torch.float32` (converted at line 383)
- **Coordinate System:** Ego vehicle frame
  * X-axis: Forward (positive = ahead of vehicle)
  * Y-axis: Left (positive = left of vehicle centerline)
  * Origin: Vehicle center of mass
- **Semantic Meaning:** Cumulative lateral path for steering control
- **Waypoint Spacing:** Non-uniform, model-predicted
- **Typical Range:** X ∈ [0, 30] meters, Y ∈ [-5, 5] meters

#### **Speed Waypoints (Temporal Velocity)**
- **Variable Name:** `speed_wps` (`src/simlingo_model.py:369`)
- **Shape:** `[1, 10, 2]`
  * Dimension 0: Batch size (always 1)
  * Dimension 1: Number of waypoints (10)
  * Dimension 2: Coordinates (x, y)
- **Data Type:** `torch.bfloat16` (model output) → `torch.float32` (converted at line 381)
- **Coordinate System:** Ego vehicle frame (same as route waypoints)
- **Semantic Meaning:** Cumulative displacement over time for speed control
- **Temporal Spacing:** Represents future trajectory at ~0.2s intervals (from training)
- **Usage:** Distance between waypoints indicates desired speed
  * `desired_speed = ||speed_wps[i+1] - speed_wps[i]|| / dt`

#### **Language Predictions**
- **Variable Name:** `language` (`src/simlingo_model.py:369`)
- **Type:** `List[str]` (length 1 for batch_size=1)
- **Data Type:** String (UTF-8)
- **Generation Method:** Greedy sampling from Qwen2-0.5B
- **Max Length:** 100 tokens
- **Content:** Natural language description of driving action
- **Example:** "The ego vehicle should continue straight and maintain speed."
- **Usage:** Optional, for interpretability (not used in control loop)

### 3.2 Output Extraction and Post-Processing

#### **Extraction** (`src/main.py:144-160`)
```python
# Run model inference
speed_wps, route_wps, language = self.model_wrapper.inference(
    camera_images=camera_images,
    image_sizes=image_sizes,
    camera_intrinsics=camera_intrinsics,
    camera_extrinsics=camera_extrinsics,
    vehicle_speed=velocity,
    target_point=target_point,
    next_target_point=next_target_point
)

# Convert to numpy
route_waypoints = route_wps[0].cpu().numpy()  # [20, 2]
speed_waypoints = speed_wps[0].cpu().numpy()  # [10, 2]
```

#### **Post-Processing** (`src/control_converter.py:257-302`)

**Waypoint Interpolation** (for steering control)
```python
def interpolate_waypoints(waypoints):
    # Filter waypoints too close to origin (< 0.05m)
    distances = np.linalg.norm(waypoints, axis=1)
    valid_mask = distances >= 0.05
    waypoints = waypoints[valid_mask]
    
    # Add origin point
    waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
    
    # Calculate cumulative distances
    dists = np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
    
    # Interpolate to 0.1m spacing using PCHIP
    interp = PchipInterpolator(dists, waypoints, axis=0)
    x = np.arange(0.1, dists[-1], 0.1)
    interp_points = interp(x)
    
    return interp_points  # [N, 2] where N = route_length / 0.1
```

### 3.3 Output Validation

**Sanity Checks** (`src/main.py:168-174`)
```python
if self.step_count == 0:
    print(f"DEBUG: route_waypoints shape: {route_waypoints.shape}")
    print(f"DEBUG: speed_waypoints shape: {speed_waypoints.shape}")
    print(f"DEBUG: First 3 route waypoints:\n{route_waypoints[:3]}")
    print(f"DEBUG: First 3 speed waypoints:\n{speed_waypoints[:3]}")
```

**Expected Output Example:**
```
route_waypoints shape: (20, 2)
speed_waypoints shape: (10, 2)
First 3 route waypoints:
[[0.12, 0.03],
 [0.45, 0.08],
 [0.89, 0.15]]
First 3 speed waypoints:
[[0.08, 0.01],
 [0.21, 0.02],
 [0.38, 0.04]]
```

---

## 4. Control Integration with QCar2 in QLabs

### 4.1 Waypoint-to-Control Conversion

#### **Speed Calculation** (`src/control_converter.py:202-238`)

**Method:** Distance-based speed estimation
```python
# Model trained with data_save_freq=4, predicts 10 waypoints
model_data_save_freq = 4
one_second = int(carla_fps // (wp_dilation * model_data_save_freq))  # 20 // 4 = 5
half_second = one_second // 2  # 2.5 ≈ 2

# Calculate desired speed from waypoint displacement
if len(speed_waypoints) >= one_second:
    desired_speed = np.linalg.norm(
        speed_waypoints[half_second - 2] - speed_waypoints[one_second - 2]
    ) * 2.0
    # Indices: [0] and [3] → displacement over ~0.6 seconds → multiply by 2
else:
    # Fallback for insufficient waypoints
    desired_speed = np.linalg.norm(
        speed_waypoints[-1] - speed_waypoints[0]
    ) * 2.0 / len(speed_waypoints)
```
- **Waypoint Indices:** 3 and 8 (half_second-2 and one_second-2)
- **Time Interval:** ~0.5 seconds between waypoints
- **Units:** meters/second

#### **Throttle and Brake Control** (`src/control_converter.py:91-181`)

**Linear Regression Controller** (default, not PID)
```python
class LongitudinalLinearRegressionController:
    def __init__(self):
        # Pre-trained coefficients from CARLA data
        self.params = np.array([
            1.1990342347353184,   # current_speed
            -0.8057602384167799,  # current_speed²
            1.710818710950062,    # 100*speed_error_cl
            0.921890257450335,    # speed_error_cl²
            1.556497522998393,    # current_speed*speed_error_cl
            -0.7013479734904027,  # current_speed²*speed_error_cl
            1.031266635497984     # braking ratio threshold
        ])
    
    def get_throttle_and_brake(self, target_speed, current_speed):
        # Convert to km/h
        current_speed_kmh = current_speed * 3.6
        target_speed_kmh = target_speed * 3.6
        speed_error = target_speed_kmh - current_speed_kmh
        
        # Braking check
        if current_speed_kmh / target_speed_kmh > self.params[-1]:
            return 0.0, True
        
        # Normalize and construct features
        speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
        current_speed_norm = current_speed_kmh / 100.0
        
        features = np.array([
            current_speed_norm,
            current_speed_norm**2,
            100 * speed_error_cl,
            speed_error_cl**2,
            current_speed_norm * speed_error_cl,
            current_speed_norm**2 * speed_error_cl
        ])
        
        # Linear regression
        throttle = np.clip(features @ self.params[:-1], 0.0, 1.0)
        return throttle, False
```
- **Input:** `target_speed` (m/s), `current_speed` (m/s)
- **Output:** `throttle` ∈ [0, 1], `brake` ∈ {True, False}
- **Coefficients:** Optimized on CARLA driving data

#### **Steering Control** (`src/control_converter.py:13-88`)

**Lateral PID Controller**
```python
class LateralPIDController:
    def __init__(self):
        self.k_p = 3.25
        self.k_i = 1.0
        self.k_d = 1.0
        self.n = 20  # Buffer size for derivative/integral
        
        # Speed-dependent lookahead
        self.aim_distance_slow = 2.25      # m (< 5.5 m/s)
        self.aim_distance_fast = 3.0       # m (5.5-15 m/s)
        self.aim_distance_very_fast = 7.0  # m (> 15 m/s)
    
    def step(self, route_np, current_speed):
        # Determine lookahead distance
        current_speed_kmh = current_speed * 3.6
        if current_speed_kmh < 5.5:
            aim_distance = 2.25
        elif current_speed_kmh < 15.0:
            aim_distance = 3.0
        else:
            aim_distance = 7.0
        
        # Convert to waypoint index (0.1m spacing)
        n_lookahead = int(min(aim_distance * 10, len(route_np) - 1))
        
        # Calculate heading error
        desired_heading_vec = route_np[n_lookahead]
        yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
        heading_error = yaw_path % (2 * np.pi)
        heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi
        
        # Update buffer
        self._window.append(heading_error)
        
        # PID control
        integral = sum(self._window) / len(self._window)
        derivative = self._window[-1] - self._window[-2]
        steering = self.k_p * heading_error + self.k_i * integral + self.k_d * derivative
        
        return np.clip(steering, -1.0, 1.0)
```
- **Input:** Interpolated route waypoints (0.1m spacing), current speed
- **Output:** `steering` ∈ [-1, 1]
- **Gains:** Kp=3.25, Ki=1.0, Kd=1.0
- **Lookahead:** Speed-adaptive (2.25m to 7.0m)

### 4.2 QCar2 Control Conversion

#### **Kinematic Bicycle Model** (`src/control_converter.py:344-375`)
```python
def bicycle_model_step(speed, dt, steer, throttle, brake):
    # Calculate acceleration
    if brake:
        accel = -4.952399  # brake_acceleration
    else:
        accel = 0.5633837 * throttle  # throttle_acceleration
    
    # Update speed
    next_speed = speed + accel * dt
    next_speed = max(next_speed, 0.0)  # ReLU
    
    return next_speed
```
- **Parameters:** Calibrated from CARLA vehicle dynamics
- **Time Step:** dt = 0.05s (20 Hz control loop)

#### **QCar2 Command Conversion** (`src/control_converter.py:304-342`)
```python
def convert_to_qcar2_control(steer, throttle, brake, current_speed, dt):
    # Predict next speed using bicycle model
    next_speed = self.bicycle_model_step(current_speed, dt, steer, throttle, brake)
    
    # Convert steering to turn angle
    # NOTE: QCar2 convention is OPPOSITE to CARLA/SimLingo
    # CARLA: positive = left turn
    # QCar2: positive = right turn
    turn_angle = -steer * 0.36848336  # steering_gain
    
    # Forward velocity is predicted speed
    forward_velocity = next_speed
    
    return forward_velocity, turn_angle
```
- **Steering Gain:** 0.36848336 (calibrated)
- **Sign Convention:** Negated for QCar2 compatibility
- **Output:**
  * `forward_velocity`: m/s (target speed)
  * `turn_angle`: radians (positive = right turn)

### 4.3 QLabs API Integration

#### **Control Command Transmission** (`src/qcar2_interface.py:203-256`)
```python
def set_control(forward_velocity, turn_angle):
    success, location, rotation, front_hit, rear_hit = self.qcar.set_velocity_and_request_state(
        forward=forward_velocity,
        turn=turn_angle,
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=False,
        brakeSignal=False,
        reverseSignal=False
    )
    
    # Update state
    self.current_location = np.array(location, dtype=np.float32)
    self.current_rotation = np.array(rotation, dtype=np.float32)
    
    # Check collisions
    self.collision_detected = front_hit or rear_hit
    
    return success, self.current_location, self.current_rotation
```
- **API:** `QLabsQCar2.set_velocity_and_request_state()`
- **Returns:** State feedback (location, rotation, collision status)
- **Update Rate:** Synchronous with control loop (20 Hz)

### 4.4 Control Loop Timing

#### **Main Control Loop** (`src/main.py:326-361`)
```python
def run(self):
    while self.running:
        loop_start_time = time.time()
        
        # Execute one control step
        if not self.run_step():
            break
        
        # Maintain control frequency (20 Hz)
        elapsed = time.time() - loop_start_time
        sleep_time = self.config.dt - elapsed  # dt = 0.05s
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"WARNING: Control loop running slow ({elapsed:.3f}s > {self.config.dt:.3f}s)")
```
- **Target Frequency:** 20 Hz (50ms period)
- **Timing Method:** Sleep-based synchronization
- **Overrun Handling:** Warning printed if loop exceeds 50ms

### 4.5 Control Flow Summary

```
Model Outputs:
  route_wps [1, 20, 2] → route_waypoints [20, 2]
  speed_wps [1, 10, 2] → speed_waypoints [10, 2]
                ↓
Interpolate route_waypoints → [N, 2] @ 0.1m spacing
                ↓
Calculate desired_speed from speed_waypoints[3] and speed_waypoints[8]
                ↓
Lateral PID Controller:
  Input: route_waypoints, current_speed
  Output: steer ∈ [-1, 1]
                ↓
Longitudinal Linear Regression Controller:
  Input: desired_speed, current_speed
  Output: throttle ∈ [0, 1], brake ∈ {True, False}
                ↓
Bicycle Model:
  Input: steer, throttle, brake, current_speed, dt=0.05s
  Output: next_speed (m/s)
                ↓
QCar2 Conversion:
  forward_velocity = next_speed
  turn_angle = -steer * 0.36848336
                ↓
QLabs API:
  qcar.set_velocity_and_request_state(forward_velocity, turn_angle)
                ↓
State Feedback:
  location [x, y, z], rotation [roll, pitch, yaw], collision flags
                ↓
State Estimator:
  Update position, rotation, velocity (moving average filter)
                ↓
Next Control Cycle (20 Hz)
```

---

## 5. Key Findings and Observations

### 5.1 Input Processing
1. **Camera preprocessing** closely matches CARLA training data (JPEG compression, bottom crop)
2. **Target points** are encoded via learned MLP, not simple concatenation
3. **Language prompt** includes Chain-of-Thought to improve reasoning
4. **No HLC used** - navigation via target points only

### 5.2 Model Architecture
1. **Vision-language fusion** happens at embedding level before transformer
2. **Learnable queries** (30 total) extract driving-specific features
3. **Cumulative waypoint prediction** ensures smooth trajectories
4. **LoRA fine-tuning** on Qwen2-0.5B enables efficient adaptation

### 5.3 Control Integration
1. **Linear regression controller** (not PID) for longitudinal control
2. **Speed-adaptive lookahead** for lateral control improves stability
3. **Bicycle model** predicts next speed for smooth acceleration
4. **Sign convention** requires negation for QCar2 compatibility

### 5.4 Performance Considerations
1. **Control frequency:** 20 Hz (50ms budget)
2. **Model inference:** ~30-40ms on GPU (within budget)
3. **Waypoint interpolation:** PCHIP ensures C1 continuity
4. **Collision detection:** Real-time via QLabs API

---

## 6. References

### Source Files
- `src/main.py` - Main control loop
- `src/simlingo_model.py` - Model wrapper and inference
- `src/camera_processor.py` - Image preprocessing
- `src/control_converter.py` - PID controllers and QCar2 conversion
- `src/qcar2_interface.py` - QLabs API integration
- `src/route_manager.py` - Waypoint management
- `src/state_estimator.py` - Vehicle state estimation
- `src/config.py` - Configuration parameters
- `simlingo/simlingo_training/models/driving.py` - Model architecture
- `simlingo/simlingo_training/models/adaptors/adaptors.py` - Adaptors and heads
- `simlingo/simlingo_training/models/encoder/internvl2_model.py` - Vision encoder
- `models/simlingo/.hydra/config.yaml` - Model configuration

### Model Components
- **Vision Encoder:** InternVL2-1B (InternViT-300M)
- **Language Model:** Qwen2-0.5B with LoRA (rank=32, alpha=64)
- **Waypoint Encoder:** MLP (2→256→512→2048)
- **Route Head:** MLP (2048→512→256→2)
- **Speed Head:** MLP (2048→256→2)

---

**End of Document**

