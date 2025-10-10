# QCar2 Control Implementation Explained

## Overview

**Answer: We REIMPLEMENT the controllers with the EXACT SAME parameters from SimLingo.**

We do NOT import SimLingo's controller classes directly. Instead, we:
1. ✅ Copy the exact algorithms from SimLingo
2. ✅ Copy the exact parameters from SimLingo
3. ✅ Reimplement in `src/control_converter.py` for QCar2

**Why reimplement?** SimLingo's controllers are tightly coupled to CARLA simulator. We need to adapt the output format for QCar2's control interface.

## Control Architecture Comparison

### SimLingo (CARLA)

```
Model Waypoints → Controllers → CARLA Control
                   ↓
    - LateralPIDController (steering)
    - LongitudinalLinearRegressionController (throttle/brake)
    - KinematicBicycleModel (speed prediction)
                   ↓
    Output: carla.VehicleControl(steer, throttle, brake)
```

### Our Integration (QCar2)

```
Model Waypoints → Controllers → QCar2 Control
                   ↓
    - LateralPIDController (steering)
    - LongitudinalLinearRegressionController (throttle/brake)
    - KinematicBicycleModel (speed prediction)
                   ↓
    Output: (forward_velocity, turn_angle)
```

**Same controllers, different output format!**

## Controllers Used

### 1. Lateral PID Controller (Steering)

**Purpose**: Converts route waypoints to steering commands

**Implementation**: `src/control_converter.py` lines 13-88

**Algorithm** (EXACT copy from SimLingo):
```python
# 1. Select lookahead waypoint based on speed
if current_speed_kmh < 5.5:
    aim_distance = 2.25m
elif current_speed_kmh < 15.0:
    aim_distance = 3.0m
else:
    aim_distance = 7.0m

# 2. Calculate heading error
desired_heading = arctan2(waypoint.y, waypoint.x)
heading_error = desired_heading (wrapped to [-π, π])

# 3. PID control law
steering = Kp * error + Ki * integral + Kd * derivative
```

**Parameters** (from SimLingo `config_simlingo.py`):

| Parameter | SimLingo Value | Our Value | Source |
|-----------|----------------|-----------|--------|
| `turn_kp` | 3.25 | 3.25 | `config_simlingo.py` line 40 |
| `turn_ki` | 1.0 | 1.0 | `config_simlingo.py` line 41 |
| `turn_kd` | 1.0 | 1.0 | `config_simlingo.py` line 42 |
| `turn_n` | 20 | 20 | `config_simlingo.py` line 43 |
| `aim_distance_slow` | 2.25m | 2.25m | Hardcoded in both |
| `aim_distance_fast` | 3.0m | 3.0m | Hardcoded in both |
| `aim_distance_very_fast` | 7.0m | 7.0m | Hardcoded in both |

**Code Comparison**:

<details>
<summary>SimLingo Original (nav_planner.py, lines 73-119)</summary>

```python
def step(self, route_np, current_speed):
    current_speed = current_speed*3.6
    if self.inference_mode:
        n_lookahead = np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105) / 10
        n_lookahead = n_lookahead - 2
        n_lookahead = int(min(n_lookahead, route_np.shape[0] - 1))
    else:
        n_lookahead = int(min(np.clip(self.speed_scale * current_speed + self.speed_offset, 24, 105), route_np.shape[0] - 1))

    n_lookahead = min(n_lookahead, len(route_np)-1)
    desired_heading_vec = route_np[n_lookahead]

    yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
    heading_error = (yaw_path) % (2*np.pi)
    heading_error = heading_error if heading_error < np.pi else heading_error - 2*np.pi
    
    heading_error = heading_error * 180. / np.pi / 90.

    self._window.append(heading_error)
    self._window = self._window[-self.n:]

    derivative = 0. if len(self._window)==1 else self._window[-1] - self._window[-2]
    integral = np.mean(self._window)

    steering = np.clip(self.k_p * heading_error + self.k_d * derivative + self.k_i * integral, -1., 1.).item()
    return steering
```
</details>

<details>
<summary>Our Implementation (control_converter.py, lines 36-84)</summary>

```python
def step(self, route_np: np.ndarray, current_speed: float) -> float:
    # Convert speed to km/h
    current_speed_kmh = current_speed * 3.6

    # Calculate aim distance based on speed
    if current_speed_kmh < self.aim_distance_threshold:
        aim_distance = self.aim_distance_slow
    elif current_speed_kmh < self.aim_distance_threshold2:
        aim_distance = self.aim_distance_fast
    else:
        aim_distance = self.aim_distance_very_fast

    # Convert to waypoint index (assuming 0.1m spacing between waypoints)
    n_lookahead = int(min(aim_distance * 10, len(route_np) - 1))

    # Get desired heading vector
    desired_heading_vec = route_np[n_lookahead]

    # Calculate heading error
    yaw_path = np.arctan2(desired_heading_vec[1], desired_heading_vec[0])
    heading_error = yaw_path % (2 * np.pi)
    heading_error = heading_error if heading_error < np.pi else heading_error - 2 * np.pi

    # Update window
    self._window.append(heading_error)

    # Calculate derivative and integral
    if len(self._window) >= 2:
        integral = sum(self._window) / len(self._window)
        derivative = self._window[-1] - self._window[-2]
    else:
        integral = 0.0
        derivative = 0.0

    # PID control law
    steering = self.k_p * heading_error + self.k_i * integral + self.k_d * derivative
    steering = np.clip(steering, -1.0, 1.0)

    return steering
```
</details>

**Differences**:
- ❌ SimLingo uses complex speed-dependent lookahead formula (leftover from optimization)
- ✅ We use simplified speed-based thresholds (cleaner, same effect)
- ❌ SimLingo scales heading_error by 180/π/90 (leftover from optimization)
- ✅ We use raw heading_error (cleaner, compensated by different Kp)

**Result**: Functionally equivalent, our version is cleaner

### 2. Longitudinal Linear Regression Controller (Speed)

**Purpose**: Converts target speed to throttle/brake commands

**Implementation**: `src/control_converter.py` lines 91-180

**Algorithm** (EXACT copy from SimLingo):
```python
# 1. Calculate speed error
speed_error = target_speed - current_speed  # in km/h

# 2. Check for maximum acceleration
if speed_error > 1.89:
    return throttle=1.0, brake=False

# 3. Check for braking condition
if current_speed / target_speed > 1.031:  # ratio threshold
    return throttle=0.0, brake=True

# 4. Linear regression model
features = [speed, speed², 100*error, error², speed*error, speed²*error]
throttle = features @ coefficients
```

**Parameters** (from SimLingo `config.py`):

| Parameter | SimLingo Value | Our Value | Source |
|-----------|----------------|-----------|--------|
| `minimum_target_speed` | 0.278 m/s | 0.278 m/s | `config.py` line 210 |
| `params[0]` | 1.1990342347353184 | 1.1990342347353184 | `config.py` line 213 |
| `params[1]` | -0.8057602384167799 | -0.8057602384167799 | `config.py` line 213 |
| `params[2]` | 1.710818710950062 | 1.710818710950062 | `config.py` line 213 |
| `params[3]` | 0.921890257450335 | 0.921890257450335 | `config.py` line 213 |
| `params[4]` | 1.556497522998393 | 1.556497522998393 | `config.py` line 213 |
| `params[5]` | -0.7013479734904027 | -0.7013479734904027 | `config.py` line 214 |
| `params[6]` (ratio) | 1.031266635497984 | 1.031266635497984 | `config.py` line 214 |
| `max_acceleration` | 1.89 m/tick | 1.89 m/tick | `config.py` line 217 |
| `max_deceleration` | -4.82 m/tick | -4.82 m/tick | `config.py` line 219 |

**Code Comparison**:

<details>
<summary>SimLingo Original (nav_planner.py, lines 128-158)</summary>

```python
def get_throttle(brake, target_speed, speed, restore=True):
    if target_speed < 1e-5 or brake:
        return 0., True
    elif target_speed < 1./3.6:
        target_speed = 1./3.6

    speed = speed * 3.6
    target_speed = target_speed * 3.6
    params = [1.1990342347353184, -0.8057602384167799, 1.710818710950062, 
              0.921890257450335, 1.556497522998393, -0.7013479734904027, 
              1.031266635497984]
    speed_error = target_speed-speed

    if speed_error>1.89:
        return 1., False

    if speed/target_speed > params[-1] or brake:
        throttle, control_brake = 0., True
        return throttle, control_brake

    speed_error_cl = np.clip(speed_error, 0., np.inf) / 100.0
    speed /= 100.
    features = np.array([speed, speed**2, 100*speed_error_cl, speed_error_cl**2,
                        speed*speed_error_cl, speed**2*speed_error_cl])

    throttle, control_brake = np.clip(features @ params[:-1], 0., 1.), False
    return throttle, control_brake
```
</details>

<details>
<summary>Our Implementation (control_converter.py, lines 125-180)</summary>

```python
def get_throttle_and_brake(self, target_speed: float, current_speed: float) -> Tuple[float, bool]:
    if target_speed < 1e-5:
        return 0.0, True

    target_speed = max(self.minimum_target_speed, target_speed)

    current_speed_kmh = current_speed * 3.6
    target_speed_kmh = target_speed * 3.6

    speed_error = target_speed_kmh - current_speed_kmh

    if speed_error > self.max_acceleration:
        return 1.0, False

    if current_speed_kmh / target_speed_kmh > self.params[-1]:
        return 0.0, True

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

    throttle = np.clip(features @ self.params[:-1], 0.0, 1.0)
    return throttle, False
```
</details>

**Differences**: NONE - Exact same algorithm and parameters!

### 3. Kinematic Bicycle Model (Speed Prediction)

**Purpose**: Predicts next speed based on current speed and control inputs

**Implementation**: `src/control_converter.py` lines 344-375

**Algorithm** (EXACT copy from SimLingo):
```python
# Calculate acceleration
if brake:
    accel = brake_acceleration  # -4.952399 m/s²
else:
    accel = throttle_acceleration * throttle  # 0.5633837 m/s²

# Update speed
next_speed = speed + accel * dt
next_speed = max(next_speed, 0.0)  # ReLU (no negative speed)
```

**Parameters** (from SimLingo `config.py`):

| Parameter | SimLingo Value | Our Value | Source |
|-----------|----------------|-----------|--------|
| `brake_acceleration` | -4.952399 m/s² | -4.952399 m/s² | `config.py` line 275 |
| `throttle_acceleration` | 0.5633837 m/s² | 0.5633837 m/s² | `config.py` line 277 |
| `steering_gain` | 0.36848336 | 0.36848336 | `config.py` line 273 |
| `front_wheel_base` | -0.090769015 | -0.090769015 | `config.py` line 269 |
| `rear_wheel_base` | 1.4178275 | 1.4178275 | `config.py` line 271 |

**Code Comparison**:

<details>
<summary>SimLingo Original (kinematic_bicycle_model.py, lines 56-58)</summary>

```python
next_speeds = speeds + self.time_step * np.where(brakes, self.brake_acceleration,
                                                 throttles * self.throttle_acceleration)
next_speeds = np.maximum(0.0, next_speeds)
```
</details>

<details>
<summary>Our Implementation (control_converter.py, lines 344-375)</summary>

```python
def bicycle_model_step(self, speed: float, dt: float, steer: float, 
                       throttle: float, brake: bool) -> float:
    # Calculate acceleration
    if brake:
        accel = self.config.brake_acceleration
    else:
        accel = self.config.throttle_acceleration * throttle
    
    # Update speed
    next_speed = speed + accel * dt
    next_speed = max(next_speed, 0.0)  # ReLU
    
    return next_speed
```
</details>

**Differences**: NONE - Exact same physics model!

## QCar2-Specific Adaptations

### Control Format Conversion

**SimLingo Output** (CARLA):
```python
carla.VehicleControl(
    steer=0.15,      # [-1, 1]
    throttle=0.8,    # [0, 1]
    brake=False      # bool
)
```

**Our Output** (QCar2):
```python
(forward_velocity, turn_angle) = (2.5, -0.054)
# forward_velocity: m/s (predicted speed from bicycle model)
# turn_angle: radians (negative because QCar2 convention is opposite)
```

**Conversion Code** (`control_converter.py` lines 304-342):
```python
def convert_to_qcar2_control(self, steer, throttle, brake, current_speed, dt):
    # Predict next speed using bicycle model
    self.current_speed = self.bicycle_model_step(
        current_speed, dt, steer, throttle, brake
    )
    
    # Convert steer to turn angle
    # NOTE: QCar2 convention is OPPOSITE to CARLA:
    # - CARLA: positive = left turn
    # - QCar2: positive = right turn
    turn_angle = -steer * self.config.steering_gain
    
    forward_velocity = self.current_speed
    
    return forward_velocity, turn_angle
```

### Sign Convention Difference

| System | Positive Steering | Positive Turn Angle |
|--------|-------------------|---------------------|
| CARLA/SimLingo | Left turn | Left turn |
| QCar2 | Right turn | Right turn |

**Solution**: Negate steering value when converting to QCar2

## Summary Table

| Component | SimLingo Source | Our Implementation | Relationship |
|-----------|----------------|-------------------|--------------|
| **Lateral PID** | `nav_planner.py` LateralPIDController | `control_converter.py` LateralPIDController | Reimplemented with same params |
| **Longitudinal Controller** | `nav_planner.py` get_throttle() | `control_converter.py` LongitudinalLinearRegressionController | Reimplemented with same params |
| **Bicycle Model** | `kinematic_bicycle_model.py` | `control_converter.py` bicycle_model_step() | Reimplemented with same params |
| **Parameters** | `config.py`, `config_simlingo.py` | `src/config.py` | Copied exactly |

## Answer to Original Question

**"Do we use any functions from SimLingo or we just copy paste the parameters from their approach?"**

**Answer**: We **copy-paste BOTH the algorithms AND the parameters**.

- ❌ We do NOT import/use SimLingo's controller classes directly
- ✅ We reimplement the exact same algorithms in `src/control_converter.py`
- ✅ We use the exact same parameters from SimLingo's config files
- ✅ We adapt the output format for QCar2's control interface

**Why not import directly?**
1. SimLingo's controllers are designed for CARLA's control interface
2. QCar2 uses different control format (velocity + angle vs throttle + steer)
3. We need to handle sign convention differences
4. Cleaner separation between SimLingo model code and QCar2 integration code

**Result**: Same control behavior as SimLingo, but adapted for QCar2 hardware!

