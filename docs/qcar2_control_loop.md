# QCar2 Control Loop Implementation

This document explains the control loop implementation for the QCar2 in QLabs, detailing how model inputs are processed into control signals.

## Overview

The control loop operates at **4 Hz** (`carla_fps` in `config.py`). In each iteration, the system:
1.  Captures sensor data (Camera, Position, Velocity).
2.  Runs the **Simlingo** model to predict future waypoints.
3.  Converts these predictions into **Speed** and **Steering** commands using a PID controller.
4.  Sends actuation signals to the QCar2.

## 1. Inputs & State Estimation

Before inference, the system gathers the current state:

*   **Camera**: RGB Image from the front camera (`CAMERA_CSI_FRONT`).
*   **Velocity**: Calculated from position changes over time (single-frame delta).
    *   $v = \frac{||p_t - p_{t-1}||}{\Delta t}$
*   **Target Points**: Two future points on the global route, converted to the **Ego Frame** (relative to the car).
    *   **Lookahead**: The system searches ahead on the route for a point ~7.5m away (`target_point_lookahead`).

## 2. Model Inference

The **Simlingo** model takes the processed inputs and outputs two key sequences of **future positions** (not velocities):

1.  **Route Waypoints** ($W_{route}$): 20 predicted future positions for steering control.
2.  **Speed Waypoints** ($W_{speed}$): 10 predicted future positions for speed control.

Each waypoint represents where the car **should be** at a specific future time:
- Waypoint spacing: 0.25 seconds (based on `dt × data_save_freq`)
- $W_{speed}$ covers 2.5 seconds into the future (10 × 0.25s)
- $W_{route}$ covers 5.0 seconds into the future (20 × 0.25s)

**Input**:
- RGB Image (single frame)
- Current vehicle speed (scalar, m/s)
- Target point in ego frame (x, y)

**Output**:
- $W_{route} = [(x_0, y_0), (x_1, y_1), ..., (x_{19}, y_{19})]$ — 20 positions
- $W_{speed} = [(x_0, y_0), (x_1, y_1), ..., (x_9, y_9)]$ — 10 positions

**Important**: The model receives only a single frame. It cannot track elapsed time explicitly (e.g., how long stopped at a stop sign). Temporal behavior is encoded implicitly in the waypoint predictions based on learned visual patterns.

## 3. Control Logic

The `ControlConverter` class (`inference/control_converter.py`) translates model outputs into control signals.

### A. Speed Control

#### What the Model Predicts

The Simlingo model outputs **Speed Waypoints** — a sequence of 10 future (x, y) positions in the ego frame:

```
W_speed = [(x₀, y₀), (x₁, y₁), (x₂, y₂), ..., (x₉, y₉)]
```

Each waypoint represents where the car **should be** at a future time step:
- `W_speed[0]` = position at t + 0.25s
- `W_speed[1]` = position at t + 0.50s
- `W_speed[2]` = position at t + 0.75s
- ...
- `W_speed[9]` = position at t + 2.50s

**Important**: These are NOT velocities — they are **positions**. Speed is derived from the displacement between waypoints.

#### How Desired Speed is Calculated

Speed is computed from the **displacement between two predicted positions** divided by the time interval between them.

**The Equation:**

$$v_{desired} = \frac{||W_{speed}[k] - W_{speed}[0]||}{k \times \Delta t_{model}}$$

**What Each Variable Means:**

| Symbol | Value | Meaning |
|--------|-------|---------|
| $W_{speed}[0]$ | (x, y) meters | Predicted car position at **t + 0.25s** (first future position) |
| $W_{speed}[k]$ | (x, y) meters | Predicted car position at **t + (k+1) × 0.25s** |
| $k$ | 2 (in our implementation) | Index offset — we use waypoint 2 to span a 0.5s window |
| $\Delta t_{model}$ | 0.25s | Time between consecutive waypoints (`dt × data_save_freq`) |
| $k \times \Delta t_{model}$ | 0.5s | Total time span between $W_{speed}[0]$ and $W_{speed}[k]$ |
| $\|\|W_{speed}[k] - W_{speed}[0]\|\|$ | meters | Euclidean distance the car travels in that time span |

**Why k=2?** Using a 2-waypoint span (0.5 seconds) provides more stable speed estimates than using adjacent waypoints, which can be noisy.

**Implementation:**

```python
dt_model = config.dt * config.data_save_freq  # 0.25s * 1 = 0.25s per waypoint
k = 2  # Use 2-waypoint span for stability

# W_speed[0] = position at t+0.25s, W_speed[2] = position at t+0.75s
time_delta = k * dt_model  # 2 * 0.25s = 0.5 seconds
distance = ||W_speed[k] - W_speed[0]||  # How far the car moves in 0.5s
desired_speed = distance / time_delta   # speed = distance / time
```

**Concrete Examples:**

| Scenario | $W_{speed}[0]$ (t+0.25s) | $W_{speed}[2]$ (t+0.75s) | Distance | Time | Desired Speed |
|----------|--------------------------|--------------------------|----------|------|---------------|
| Stopped | (0.0, 0.0) | (0.0, 0.0) | 0.0m | 0.5s | **0.0 m/s** |
| Slow | (0.5, 0.0) | (1.5, 0.0) | 1.0m | 0.5s | **2.0 m/s** |
| Fast | (1.0, 0.0) | (3.0, 0.0) | 2.0m | 0.5s | **4.0 m/s** |

**Clamping**: The result is clamped to `qcar2_max_speed` (4.0 m/s).

#### Temporal Information in Waypoints

The waypoint structure implicitly encodes **when** to change speed:
- If the model predicts the car should stay stopped, all waypoints cluster near the origin.
- If the model predicts the car should accelerate, waypoints spread out progressively.
- The **spacing** between consecutive waypoints encodes the predicted speed at each future moment.

**Limitation**: The model receives only a single frame. For identical visual inputs (e.g., stopped at a static stop sign), the model cannot distinguish how long it has been stopped and will produce the same waypoint predictions.

### B. Steering Control (Lateral PID)

#### What the Model Predicts

The Simlingo model outputs **Route Waypoints** — a sequence of 20 future (x, y) positions representing the predicted path:

```
W_route = [(x₀, y₀), (x₁, y₁), ..., (x₁₉, y₁₉)]
```

These waypoints define the trajectory the car should follow, used by the steering controller to compute heading corrections.

#### Steering Computation Pipeline

**Step 1: Waypoint Interpolation**

The predicted route waypoints are interpolated to a fixed spacing of **0.1m** using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial):

```python
route_interp = interpolate_waypoints(W_route)  # ~100+ points at 0.1m spacing
```

**Step 2: Speed-Dependent Lookahead**

A lookahead index is calculated based on current speed to determine which point to "aim" at:

```python
speed_kmh = current_speed * 3.6
lookahead_index = clip(0.9755 * speed_kmh + 1.915, 24, 105)
aim_point = route_interp[lookahead_index]  # (x, y) in ego frame
```

| Speed | Lookahead Index | Lookahead Distance |
|-------|-----------------|-------------------|
| 0 m/s | 24 | 2.4m |
| 2 m/s | 31 | 3.1m |
| 4 m/s | 38 | 3.8m |

**Step 3: Heading Error Calculation**

The angle to the aim point is computed and normalized:

```python
yaw_path = atan2(aim_point.y, aim_point.x)  # Angle to aim point
heading_error = normalize_angle(yaw_path)    # Wrap to [-π, π]
heading_error_norm = heading_error * (180/π) / 90  # Normalize: 90° → 1.0
```

**Step 4: PID Control**

The normalized heading error drives a PID controller:

```python
steering = K_p * error + K_i * integral(error) + K_d * derivative(error)
steering = clip(steering, -1.0, 1.0)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `K_p` | 12.0 | Proportional gain |
| `K_i` | 0.0 | Integral gain (disabled) |
| `K_d` | 3.5 | Derivative gain |

**Note on Derivative Scaling**: The derivative term is scaled by 1/5 to account for the frequency difference between Simlingo's original 20Hz control and our 4Hz control loop.

### C. Actuation (QCar2 Interface)

The final step converts the abstract control signals into QCar2-specific commands.

#### Forward Velocity (Rate-Limited)

The desired speed is sent to QCar2 with rate limiting to ensure physically plausible acceleration/deceleration:

```python
speed_diff = target_speed_cmd - current_speed

# Rate limiting
max_accel = qcar2_max_acceleration * dt  # 0.2 * 0.25 = 0.05 m/s per step
max_decel = qcar2_max_deceleration * dt  # 4.0 * 0.25 = 1.0 m/s per step

if speed_diff > 0:
    speed_diff = min(speed_diff, max_accel)  # Limit acceleration
else:
    speed_diff = max(speed_diff, -max_decel)  # Limit deceleration

forward_velocity = current_speed + speed_diff
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `qcar2_max_acceleration` | 0.2 m/s² | Max speed increase: 0.05 m/s per step |
| `qcar2_max_deceleration` | 4.0 m/s² | Max speed decrease: 1.0 m/s per step |

**Note**: The model learns gradual deceleration profiles from expert data (~-0.85 m/s²). We removed hardcoded brake logic to let the model's learned behavior drive speed control directly.

#### Turn Angle

The normalized steering [-1, 1] is converted to QCar2's turn angle in radians:

```python
turn_angle = -steering * qcar2_max_steering
```

**Sign Inversion**: QCar2 uses opposite steering convention:
- CARLA/Simlingo: positive steering = left turn
- QCar2: positive turn_angle = right turn

| Parameter | Value | Description |
|-----------|-------|-------------|
| `qcar2_max_steering` | π/9 rad (~20°) | Maximum physical steering angle |

## Key Configuration Variables

These values are defined in `core/config.py`.

| Variable | Value | Description |
| :--- | :--- | :--- |
| `carla_fps` | 4 Hz | Main control loop frequency |
| `dt` | 0.25 s | Time step duration |
| `data_save_freq` | 1 | Save every frame (waypoint interval = dt × data_save_freq) |
| `turn_kp` | 12.0 | Proportional gain for steering PID |
| `turn_ki` | 0.0 | Integral gain for steering PID (disabled) |
| `turn_kd` | 3.5 | Derivative gain for steering PID |
| `qcar2_max_speed` | 4.0 m/s | Maximum allowed speed |
| `qcar2_max_acceleration` | 0.2 m/s² | Maximum acceleration rate |
| `qcar2_max_deceleration` | 4.0 m/s² | Maximum deceleration rate |
| `qcar2_max_steering` | π/9 rad | Maximum steering angle (~20°) |
| `target_point_lookahead` | 7.5 m | Distance to look ahead for global route target |
| `interpolation_spacing` | 0.1 m | Spacing for route waypoint interpolation |

## Summary Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL LOOP (4 Hz)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. STATE ESTIMATION                                                        │
│     ├── Camera Image (RGB)                                                  │
│     ├── Position → Velocity (single-frame delta: v = ||Δp|| / Δt)          │
│     └── Route → Target Points (ego frame, 7.5m lookahead)                   │
│                                                                             │
│  2. MODEL INFERENCE (Simlingo)                                              │
│     ├── Input:  (Image, Speed, Target Point)                                │
│     └── Output: W_route[20×2], W_speed[10×2]  (future positions)            │
│                                                                             │
│  3. CONTROL CONVERSION                                                      │
│     │                                                                       │
│     ├── SPEED:                                                              │
│     │   desired_speed = ||W_speed[2] - W_speed[0]|| / 0.5s                  │
│     │                                                                       │
│     └── STEERING:                                                           │
│         route_interp = interpolate(W_route, spacing=0.1m)                   │
│         aim_point = route_interp[lookahead_index(speed)]                    │
│         heading_error = atan2(aim.y, aim.x) * (180/π) / 90                  │
│         steering = PID(heading_error)  →  [-1, 1]                           │
│                                                                             │
│  4. ACTUATION (QCar2)                                                       │
│     ├── forward_velocity = rate_limit(desired_speed, current_speed)         │
│     └── turn_angle = -steering × (π/9)  [radians, sign inverted]            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
