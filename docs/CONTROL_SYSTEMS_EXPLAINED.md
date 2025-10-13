# SimLingo Control Systems - Simple Explanation

## Overview

SimLingo uses two controllers to drive the QCar2:
1. **PID Controller** → Controls **steering** (left/right)
2. **Linear Regression Controller** → Controls **speed** (throttle/brake)

---

## 1. PID Controller (Steering Control)

### What is PID?

PID stands for **P**roportional-**I**ntegral-**D**erivative. It's like a smart thermostat that:
- **P (Proportional)**: Reacts to current error → "How far off am I right now?"
- **I (Integral)**: Remembers past errors → "Have I been consistently off?"
- **D (Derivative)**: Predicts future errors → "Am I getting closer or further away?"

### How It Works

**Input**: 20 route waypoints from the model (positions in meters, ego frame)

**Process**:
```
Step 1: Interpolate 20 waypoints → Dense waypoints at 0.1m spacing
Step 2: Pick a target waypoint based on speed (lookahead distance)
Step 3: Calculate heading error (angle to target)
Step 4: Apply PID formula to get steering
```

**Output**: Steering angle [-1, 1] where -1 = full right, +1 = full left

### Concrete Example

**Given**:
- Current speed: 3 m/s (10.8 km/h)
- Route waypoints (first 5 of 20):
  ```
  [0]: (0.0, 0.0)   ← vehicle position
  [1]: (2.5, 0.1)
  [2]: (5.0, 0.3)
  [3]: (7.5, 0.6)
  [4]: (10.0, 1.0)
  ```

**Step 1: Determine Lookahead Distance**
```python
speed_kmh = 3.0 * 3.6 = 10.8 km/h

if speed_kmh < 5.5:
    aim_distance = 2.25 meters  ✓ (we're at 10.8 km/h, so use 3.0m)
elif speed_kmh < 15.0:
    aim_distance = 3.0 meters
else:
    aim_distance = 7.0 meters
```

**Step 2: After Interpolation to 0.1m Spacing**
```
Interpolated waypoints (every 0.1m):
[0]: (0.0, 0.0)
[1]: (0.1, 0.004)
[2]: (0.2, 0.008)
...
[30]: (3.0, 0.18)  ← Target waypoint at 3.0m
```

**Step 3: Calculate Heading Error**
```python
target_waypoint = (3.0, 0.18)  # 3.0m ahead

# Calculate angle to target
heading_error = arctan2(0.18, 3.0) = arctan2(y, x)
              = 0.0599 radians
              = 3.43 degrees

# Normalize to [-π, π]
if heading_error > π:
    heading_error -= 2π
# Result: 0.0599 radians (small left turn needed)
```

**Step 4: Apply PID Formula**
```python
# Our PID gains
kp = 0.9
ki = 0.75
kd = 0.3

# Assume error history (last 20 steps)
error_window = [0.05, 0.055, 0.058, 0.059, 0.0599, ...]

# Calculate components
P = kp * heading_error
  = 0.9 * 0.0599 = 0.0539

I = ki * mean(error_window)
  = 0.75 * 0.057 = 0.0428

D = kd * (current_error - previous_error)
  = 0.3 * (0.0599 - 0.059) = 0.00027

# Final steering
steering = P + I + D
         = 0.0539 + 0.0428 + 0.00027
         = 0.097

# Clip to [-1, 1]
steering = 0.097  ✓ (small left turn)
```

**Result**: Steering = **0.097** (gentle left turn to follow the curve)

---

## 2. Linear Regression Controller (Speed Control)

### What is Linear Regression?

Instead of using a simple formula, this controller uses a **learned model** (like a smart equation) that was trained on real driving data. It takes multiple features and combines them with learned weights to predict the best throttle.

### How It Works

**Input**: 10 speed waypoints from the model (positions in meters, ego frame)

**Process**:
```
Step 1: Extract waypoints[0] and waypoints[3] (0.75s apart)
Step 2: Calculate desired speed = distance * 2.0
Step 3: Calculate speed error = desired - current
Step 4: Build feature vector (6 features)
Step 5: Apply learned weights → throttle
```

**Output**: Throttle [0, 1] and Brake (true/false)

### Concrete Example

**Given**:
- Current speed: 2.0 m/s
- Speed waypoints (10 total):
  ```
  [0]: (0.0, 0.0)    ← current position
  [1]: (0.5, 0.0)    ← 0.25s ahead
  [2]: (1.0, 0.0)    ← 0.50s ahead
  [3]: (1.5, 0.0)    ← 0.75s ahead
  [4]: (2.0, 0.0)    ← 1.00s ahead
  ...
  [9]: (4.5, 0.0)    ← 2.25s ahead
  ```

**Step 1: Calculate Desired Speed**
```python
# Use waypoints[0] and waypoints[3] (0.75s apart)
wp0 = (0.0, 0.0)
wp3 = (1.5, 0.0)

# Calculate distance
distance = sqrt((1.5-0.0)² + (0.0-0.0)²)
         = 1.5 meters

# Calculate speed (formula assumes 0.5s, so underestimates by 33%)
desired_speed = distance * 2.0
              = 1.5 * 2.0
              = 3.0 m/s
```

**Step 2: Calculate Speed Error**
```python
current_speed = 2.0 m/s
desired_speed = 3.0 m/s

# Convert to km/h
current_speed_kmh = 2.0 * 3.6 = 7.2 km/h
desired_speed_kmh = 3.0 * 3.6 = 10.8 km/h

speed_error = 10.8 - 7.2 = 3.6 km/h
```

**Step 3: Check Special Cases**
```python
# Maximum acceleration check
if speed_error > 1.89:
    return throttle=1.0, brake=False  ✓ (3.6 > 1.89, so full throttle!)
```

**Result**: Throttle = **1.0**, Brake = **False** (accelerate hard!)

---

### Alternative: Normal Case (No Max Acceleration)

If speed_error was smaller (e.g., 1.0 km/h), we'd use the full linear regression:

**Step 4: Build Feature Vector**
```python
# Normalize values
current_speed_norm = 7.2 / 100 = 0.072
speed_error_cl = max(0, 1.0) / 100 = 0.01  # Clipped to positive

# Build 6 features
features = [
    current_speed_norm,                    # 0.072
    current_speed_norm²,                   # 0.00518
    100 * speed_error_cl,                  # 1.0
    speed_error_cl²,                       # 0.0001
    current_speed_norm * speed_error_cl,   # 0.00072
    current_speed_norm² * speed_error_cl   # 0.0000518
]
```

**Step 5: Apply Learned Weights**
```python
# Learned coefficients (from SimLingo training)
weights = [
    1.199,   # current_speed
    -0.806,  # current_speed²
    1.711,   # 100*speed_error
    0.922,   # speed_error²
    1.556,   # current_speed*speed_error
    -0.701   # current_speed²*speed_error
]

# Calculate throttle
throttle = features[0] * weights[0] +
           features[1] * weights[1] +
           features[2] * weights[2] +
           features[3] * weights[3] +
           features[4] * weights[4] +
           features[5] * weights[5]

        = 0.072*1.199 + 0.00518*(-0.806) + 1.0*1.711 +
          0.0001*0.922 + 0.00072*1.556 + 0.0000518*(-0.701)

        = 0.0863 - 0.0042 + 1.711 + 0.0001 + 0.0011 - 0.00004
        = 1.794

# Clip to [0, 1]
throttle = min(1.0, max(0.0, 1.794)) = 1.0
```

**Result**: Throttle = **1.0** (full acceleration to reach desired speed)

---

## Summary

| Controller | Input | Output | Key Idea |
|------------|-------|--------|----------|
| **PID (Steering)** | 20 route waypoints | Steering [-1, 1] | Look ahead, calculate angle error, apply PID formula |
| **Linear Regression (Speed)** | 10 speed waypoints | Throttle [0, 1] + Brake | Calculate desired speed, use learned model for throttle |

### Control Loop (20 Hz)
```
1. Model predicts → 20 route waypoints + 10 speed waypoints
2. PID controller → steering = 0.097 (gentle left)
3. Linear regression → throttle = 1.0 (accelerate)
4. Send to QCar2 → vehicle turns left while accelerating
5. Repeat every 0.05 seconds (20 times per second)
```

### Key Insight

- **PID** is reactive: It continuously corrects based on where you are vs. where you should be
- **Linear Regression** is predictive: It uses learned patterns from training data to choose the best throttle

Both work together to make the car follow the route smoothly at the right speed!

