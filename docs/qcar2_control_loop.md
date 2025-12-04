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

The **Simlingo** model takes the processed inputs and outputs two key sequences:

1.  **Route Waypoints** ($W_{route}$): Predicted path of the vehicle in the ego frame.
2.  **Speed Waypoints** ($W_{speed}$): Predicted path used specifically for speed estimation.

*   **Input**: Image, Speed, Target Points (Ego Frame).
*   **Output**: $W_{route} = [(x_0, y_0), (x_1, y_1), ...]$, $W_{speed} = [(x_0, y_0), (x_1, y_1), ...]$

## 3. Control Logic

The `ControlConverter` class (`inference/control_converter.py`) translates model outputs into control signals.

### A. Speed Control

The desired speed is calculated from the **Speed Waypoints**. The model predicts where the car *should be* in future time steps.

*   **Logic**: Calculate the distance between predicted points and divide by the time delta.
*   **Equation**:
    $$ v_{desired} = \frac{||W_{speed}[k] - W_{speed}[0]||}{k \cdot \Delta t_{model}} $$
    *   Where $\Delta t_{model} = 0.25s$ (based on 4Hz frequency).
    *   Typically uses the 1st or 2nd waypoint.
*   **Clamping**: The result is clamped to `qcar2_max_speed` (4.0 m/s).

### B. Steering Control (Lateral PID)

Steering is computed using a **PID Controller** that tracks the **Route Waypoints**.

1.  **Interpolation**: The predicted route waypoints are interpolated to have a fixed spacing of **0.1m**.
2.  **Lookahead Calculation**: A dynamic lookahead distance is calculated based on current speed.
    *   $$ Index_{lookahead} = \text{clip}(0.9755 \cdot v_{km/h} + 1.915, 24, 105) $$
    *   This selects a point on the interpolated path to "aim" at.
3.  **Heading Error**: The angle to the lookahead point is calculated.
    *   $\theta_{error} = \text{atan2}(y_{aim}, x_{aim})$
    *   **Normalization**: The error is normalized so that 90 degrees $\approx$ 1.0.
    *   $\theta_{norm} = \theta_{error} \cdot \frac{2}{\pi}$
4.  **PID Equation**:
    $$ Steering = K_p \cdot \theta_{norm} + K_i \cdot \int \theta_{norm} + K_d \cdot \frac{d\theta_{norm}}{dt} $$
    *   **Output**: A value between [-1.0, 1.0].

### C. Actuation (QCar2 Interface)

The final step converts the abstract control signals into QCar2-specific units.

*   **Forward Velocity**:
    *   The `desired_speed` is rate-limited by `qcar2_max_acceleration` (0.2 m/s²) and `qcar2_max_deceleration` (4.0 m/s²) to ensure smooth transitions.
    *   **Braking**: If the model predicts a stop (or explicit brake flag), velocity is forced to 0.0.
*   **Turn Angle**:
    *   QCar2 steering is in **radians**, and the direction is inverted compared to the model (Right is positive).
    *   $$ \phi_{turn} = -Steering \cdot \phi_{max} $$
    *   Where $\phi_{max} = \frac{\pi}{9}$ (~20 degrees).

## Key Configuration Variables

These values are defined in `core/config.py`.

| Variable | Value | Description |
| :--- | :--- | :--- |
| `carla_fps` | 4 Hz | Main control loop frequency. |
| `dt` | 0.25 s | Time step duration. |
| `turn_kp` | 12.0 | Proportional gain for steering PID. |
| `turn_ki` | 0.0 | Integral gain for steering PID. |
| `turn_kd` | 3.5 | Derivative gain for steering PID. |
| `qcar2_max_speed` | 4.0 m/s | Maximum allowed speed. |
| `qcar2_max_steering` | $\pi/9$ rad | Maximum steering angle (~20°). |
| `target_point_lookahead` | 7.5 m | Distance to look ahead for global route target. |

## Summary Flow

1.  **State** $\rightarrow$ `(Image, Speed, Target)`
2.  **Model** $\rightarrow$ `(Image, Speed, Target)` $\rightarrow$ `Waypoints`
3.  **Converter** $\rightarrow$ `Waypoints` $\rightarrow$ `(Desired Speed, Steering [-1,1])`
4.  **Interface** $\rightarrow$ `(Desired Speed, Steering)` $\rightarrow$ `(Velocity m/s, Angle rad)` $\rightarrow$ **QCar2**
