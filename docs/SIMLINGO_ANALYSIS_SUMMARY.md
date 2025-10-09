# SimLingo Model Integration: Executive Summary

**Document Version:** 1.0  
**Date:** 2025-10-09  
**Related:** [Full Technical Analysis](SIMLINGO_TECHNICAL_ANALYSIS.md)

---

## Quick Reference

This document provides a concise summary of the SimLingo model integration with QCar2 in QLabs. For detailed technical specifications, see the full technical analysis document.

---

## 1. Input Data Flow - Quick Reference

### Camera Images (Front-View Tiles)
- **Source:** QCar2 CSI front camera (820×410 BGR)
- **Preprocessing:** JPEG compression → 10% bottom crop → dynamic split into 2 patches
- **Final Format:** `[1, 1, 2, 3, 448, 448]` bfloat16
- **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Code:** `src/camera_processor.py:42-95`

### GPS Target Points (TP)
- **Source:** Route manager with 5.0m lookahead
- **Format:** Two points `[2, 2]` float32 in ego frame (x=forward, y=left)
- **Encoding:** MLP (2→256→512→2048) replaces `<TARGET_POINT>` tokens
- **Code:** `src/route_manager.py:93-127`, `simlingo_model.py:191-290`

### Language Prompt (pglobal)
- **Template:** `"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"`
- **Image Tokens:** 512× `<IMG_CONTEXT>` (256 per patch)
- **Total Length:** ~540 tokens after tokenization
- **Code:** `src/config.py:271-290`, `simlingo_model.py:203-219`

### Vehicle Speed
- **Source:** State estimator (finite difference of position)
- **Format:** `[1, 1]` float32 tensor (m/s)
- **Filter:** 5-sample moving average
- **Code:** `src/state_estimator.py:66-87`

### High-Level Commands (HLC)
- **Status:** NOT USED (navigation via target points only)

---

## 2. Model Architecture - Quick Reference

### Model Components
- **Vision Encoder:** InternVL2-1B (InternViT-300M)
- **Language Model:** Qwen2-0.5B with LoRA (rank=32, alpha=64)
- **Hidden Size:** 2048
- **Total Parameters:** ~1.5B (vision) + ~0.5B (language)

### Forward Pass Stages
1. **Input Preparation:** Create DrivingInput with all modalities
2. **Adaptor Processing:**
   - Language: Embed prompt tokens → `[1, ~540, 2048]`
   - Driving: Create 30 learnable queries (20 route + 10 speed)
3. **Placeholder Replacement:**
   - Target points → MLP → replace `<TARGET_POINT>` tokens
   - Image patches → InternViT → replace `<IMG_CONTEXT>` tokens (512 total)
4. **Transformer Processing:** Qwen2-0.5B processes `[1, ~571, 2048]` embeddings
5. **Prediction Heads:**
   - Route: MLP (2048→512→256→2) on 20 queries → `[1, 20, 2]`
   - Speed: MLP (2048→256→2) on 10 queries → `[1, 10, 2]`
   - Both use cumsum for cumulative waypoints

### Key Files
- `simlingo/simlingo_training/models/driving.py` - Main model
- `simlingo/simlingo_training/models/adaptors/adaptors.py` - Prediction heads
- `simlingo/simlingo_training/models/encoder/internvl2_model.py` - Vision encoder

---

## 3. Model Outputs - Quick Reference

### Route Waypoints (Geometric Path)
- **Shape:** `[1, 20, 2]`
- **Data Type:** bfloat16 → float32
- **Coordinate System:** Ego frame (x=forward, y=left)
- **Semantic:** Cumulative lateral path for steering
- **Range:** X ∈ [0, 30]m, Y ∈ [-5, 5]m

### Speed Waypoints (Temporal Velocity)
- **Shape:** `[1, 10, 2]`
- **Data Type:** bfloat16 → float32
- **Coordinate System:** Ego frame
- **Semantic:** Cumulative displacement over time
- **Usage:** `desired_speed = ||speed_wps[3] - speed_wps[8]|| * 2.0`

### Language Predictions
- **Type:** `List[str]`
- **Generation:** Greedy sampling (max 100 tokens)
- **Usage:** Optional, for interpretability only

### Extraction Code
```python
# src/main.py:144-160
speed_wps, route_wps, language = model_wrapper.inference(...)
route_waypoints = route_wps[0].cpu().numpy()  # [20, 2]
speed_waypoints = speed_wps[0].cpu().numpy()  # [10, 2]
```

---

## 4. Control Integration - Quick Reference

### Waypoint-to-Control Conversion

**Speed Calculation** (`control_converter.py:220-238`)
```python
# Use waypoints at indices 3 and 8 (half_second and one_second)
desired_speed = ||speed_waypoints[3] - speed_waypoints[8]|| * 2.0
```

**Throttle/Brake** (`control_converter.py:125-180`)
- **Method:** Linear regression controller (NOT PID)
- **Coefficients:** Pre-trained on CARLA data
- **Output:** throttle ∈ [0, 1], brake ∈ {True, False}

**Steering** (`control_converter.py:36-84`)
- **Method:** Lateral PID controller
- **Gains:** Kp=3.25, Ki=1.0, Kd=1.0
- **Lookahead:** Speed-adaptive (2.25m to 7.0m)
- **Interpolation:** PCHIP to 0.1m spacing
- **Output:** steer ∈ [-1, 1]

### QCar2 Control Conversion

**Bicycle Model** (`control_converter.py:344-375`)
```python
# Predict next speed
accel = -4.952399 if brake else 0.5633837 * throttle
next_speed = max(speed + accel * dt, 0.0)
```

**QCar2 Commands** (`control_converter.py:304-342`)
```python
forward_velocity = next_speed
turn_angle = -steer * 0.36848336  # Note: negated for QCar2 convention
```

### Control Loop

**Frequency:** 20 Hz (50ms period)  
**API:** `qcar.set_velocity_and_request_state(forward_velocity, turn_angle)`  
**Timing:** Sleep-based synchronization  
**Code:** `src/main.py:326-361`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ QCar2 Sensors (QLabs)                                           │
├─────────────────────────────────────────────────────────────────┤
│ • Camera: 820×410 BGR                                           │
│ • Position: [x, y, z]                                           │
│ • Rotation: [roll, pitch, yaw]                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Preprocessing                                                    │
├─────────────────────────────────────────────────────────────────┤
│ • Camera: JPEG → Crop → Patches → [1,1,2,3,448,448] bfloat16   │
│ • Route: World → Ego → [2,2] float32                            │
│ • Speed: Position diff → Moving avg → float                     │
│ • Prompt: Template + Tokenize → [1,~540] int64                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ SimLingo Model                                                   │
├─────────────────────────────────────────────────────────────────┤
│ 1. Embed prompt → [1,~540,2048]                                 │
│ 2. Encode target points → [2,2048]                              │
│ 3. Extract vision features → [512,2048]                         │
│ 4. Create driving queries → [30,2048]                           │
│ 5. Concatenate → [1,~571,2048]                                  │
│ 6. Qwen2-0.5B transformer → [1,~571,2048]                       │
│ 7. Route head → [1,20,2]                                        │
│ 8. Speed head → [1,10,2]                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Control Conversion                                               │
├─────────────────────────────────────────────────────────────────┤
│ • Speed: ||wps[3]-wps[8]|| * 2.0 → desired_speed               │
│ • Throttle/Brake: Linear regression → [0,1] / {T,F}            │
│ • Steering: PID on interpolated route → [-1,1]                 │
│ • Bicycle model: Predict next_speed                             │
│ • QCar2: (next_speed, -steer*0.368)                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ QCar2 Actuation (QLabs)                                         │
├─────────────────────────────────────────────────────────────────┤
│ • set_velocity_and_request_state(forward, turn)                 │
│ • Returns: location, rotation, collision                        │
│ • Frequency: 20 Hz                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Implementation Details

### 1. Coordinate Systems
- **World Frame:** QLabs global coordinates
- **Ego Frame:** Vehicle-centric (x=forward, y=left, origin=vehicle center)
- **Transformation:** Rotation matrix transpose applied to translated points

### 2. Sign Conventions
- **CARLA/SimLingo:** Positive steering = left turn
- **QCar2:** Positive turn_angle = right turn
- **Solution:** Negate steering before sending to QCar2

### 3. Timing Constraints
- **Control Loop:** 20 Hz (50ms budget)
- **Model Inference:** ~30-40ms on GPU
- **Remaining:** ~10-20ms for preprocessing and control conversion

### 4. Waypoint Semantics
- **Route waypoints:** Cumulative lateral path (for steering)
- **Speed waypoints:** Cumulative displacement (for speed)
- **Both:** Predicted in ego frame, require interpolation for control

### 5. Model Configuration
- **Vision:** InternVL2-1B (use_global_img=False, 2 patches)
- **Language:** Qwen2-0.5B (LoRA: r=32, alpha=64)
- **Prediction:** 20 route waypoints, 10 speed waypoints (2D mode)

---

## Verification Against Architecture Diagram

The implementation matches the provided SimLingo architecture diagram:

✅ **Language Prompt (pglobal):** Tokenized and embedded via LLM tokenizer  
✅ **Front-view Tiles (iv):** 2 patches processed by InternVIT-300M  
✅ **GPS Target Points (TP):** Encoded via MLP, not LLM tokenizer  
✅ **Token Interleaver (IL):** Concatenates vision, text, and waypoint embeddings  
✅ **Pre-trained Qwen2-0.5B:** Processes interleaved tokens  
✅ **Path Waypoint Queries (qp):** 20 learnable queries → MLP → geometric path (p)  
✅ **Speed Waypoint Queries (qw):** 10 learnable queries → MLP → temporal speed (w)  
✅ **Language Predictions (l):** Greedy sampling from language model  

**Key Difference:** The diagram shows "Language Command (HLC)" as an alternative to TP, but the implementation uses TP exclusively (eval_route_as='target_point').

---

## File Reference Quick Index

| Component | Primary File | Line Range |
|-----------|-------------|------------|
| Camera Preprocessing | `src/camera_processor.py` | 42-95 |
| Target Point Selection | `src/route_manager.py` | 93-127 |
| Prompt Creation | `src/simlingo_model.py` | 191-290 |
| Model Forward Pass | `simlingo/models/driving.py` | 104-233 |
| Prediction Heads | `simlingo/models/adaptors/adaptors.py` | 96-180 |
| Speed Calculation | `src/control_converter.py` | 220-238 |
| Steering PID | `src/control_converter.py` | 36-84 |
| Throttle/Brake | `src/control_converter.py` | 125-180 |
| QCar2 Conversion | `src/control_converter.py` | 304-342 |
| Control Loop | `src/main.py` | 102-287 |

---

**For detailed technical specifications, tensor shapes, and code references, see [SIMLINGO_TECHNICAL_ANALYSIS.md](SIMLINGO_TECHNICAL_ANALYSIS.md).**

