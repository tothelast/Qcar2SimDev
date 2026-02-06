# Simlingo Model Testing Framework

## Overview

Tests the Simlingo model on roundabout navigation with obstacle variations. Measures safety and route completion.

## Running commands

```bash
# Full test (15 runs)
python results/test_simlingo_roundabout.py --checkpoint <path_to_checkpoint.pt>

# Single scenario
python results/test_simlingo_roundabout.py --checkpoint <path> --scenario obstacle_var1
```

**Prerequisite**: QLabs must be running with SDCS RoadMap loaded.

---

## How Results Are Calculated

### 1. Collision Detection

```
collision_detected = collision_count > 0
```

The QCar2 interface checks for collisions at each control step. If any collision occurs during the run, `collision_detected` is `True`.

---

### 2. Stopped Before Obstacle

The system loops through the trajectory and checks each point:

```
For each trajectory point:
    distance_to_obstacle = distance(vehicle_position, obstacle_position)

    If speed < 0.05 m/s AND no collision:
        stopped_before_obstacle = True
        stopping_distance = distance_to_obstacle
        Stop checking (use first stop point)
```

**Key threshold**: `STOPPED_SPEED_THRESHOLD = 0.05 m/s`

If the vehicle ever drops below 0.05 m/s before hitting the obstacle, it counts as "stopped".

---

### 3. Stopping Distance

```
stopping_distance = Euclidean distance from vehicle to obstacle when speed first drops below 0.05 m/s
```

Calculated as:
```
stopping_distance = sqrt((vehicle_x - obstacle_x)² + (vehicle_y - obstacle_y)²)
```

Returns `-1.0` if:
- No obstacle in scenario (baseline)
- Collision occurred
- Vehicle never stopped

---

### 4. Route Coverage

```
For each waypoint in route (85 total):
    Find minimum distance from any trajectory point to this waypoint
    If min_distance < 1.5 meters:
        waypoint is "reached"

route_coverage_percent = (waypoints_reached / total_waypoints) × 100
```

**Example**: If vehicle passed within 1.5m of 77 out of 85 waypoints:
```
route_coverage = (77 / 85) × 100 = 90.6%
```

---

### 5. Distance Traveled

```
total_distance = 0
For each consecutive pair of trajectory points:
    total_distance += distance(point[i], point[i+1])
```

Sums the Euclidean distance between each consecutive position in the trajectory.

---

### 6. Lateral Deviation

Measures how far the vehicle strayed from the planned route.

```
For each trajectory point:
    Find distance to nearest route waypoint
    Add to deviations list

avg_lateral_deviation = mean(deviations)
max_lateral_deviation = max(deviations)
```

Lower values = vehicle followed the route more closely.

---

### 7. Pass/Fail Determination

#### Baseline (no obstacle)
```
PASS if:
    route_coverage >= 90% AND
    no timeout
```

#### Obstacle Scenarios
```
PASS if:
    no collision AND (
        (stopped_before_obstacle AND stopping_distance > 0.3m) OR
        route_coverage >= 80%
    )
```

The 0.3m minimum stopping distance prevents false positives from stopping too close or on top of the obstacle.

---

## Early Termination for Obstacle Scenarios

To avoid waiting for timeout when the vehicle has clearly stopped:

```
If obstacle scenario AND trajectory has 20+ entries:
    avg_speed_last_20_steps = average of last 20 speed values

    If avg_speed < 0.05 m/s:
        distance_to_obstacle = distance(current_pos, obstacle_pos)
        distance_from_start = distance(current_pos, route_start)

        If distance_to_obstacle < 15m AND distance_from_start > 5m:
            Terminate test (vehicle stopped near obstacle)
```

This distinguishes between:
- **Slow start** (ignored): distance from start < 5m
- **Stopped at obstacle** (terminates): near obstacle AND past start
- **Route complete** (handled separately): reaches final waypoint

---

## Test Scenarios

| Scenario | Obstacle Location | Runs |
|----------|-------------------|------|
| baseline | None | 5 |
| obstacle_var1 | Early roundabout [21.01, 33.90] | 2 |
| obstacle_var2 | Mid roundabout [18.85, 44.23] | 2 |
| obstacle_var3 | Roundabout exit [6.07, 44.97] | 2 |
| obstacle_var4 | Straight section [-10.60, 44.97] | 2 |
| obstacle_var5 | Late route [-18.73, 40.37] | 2 |

---

## Output Files

```
results/
├── runs/
│   ├── baseline_run_1/trajectory_log.json
│   ├── obstacle_var1_run_1/trajectory_log.json
│   └── ...
├── test_results_<timestamp>.json
└── test_results_<timestamp>.csv
```

### CSV Columns
```
scenario, run, pass, collision, stopped, stopping_distance_m, route_coverage_pct, avg_lateral_dev_m, total_steps, total_time_s
```

---

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| STOPPED_SPEED_THRESHOLD | 0.05 m/s | Speed below this = "stopped" |
| STOPPED_WINDOW_SIZE | 20 steps | Steps to confirm sustained stop |
| MIN_DISTANCE_FROM_START | 5.0 m | Ignore stops in start zone |
| MAX_DISTANCE_TO_OBSTACLE | 15.0 m | Must be within this to trigger early termination |
| Route waypoint threshold | 1.5 m | Distance to count waypoint as "reached" |
| Minimum stopping distance | 0.3 m | Must stop at least this far from obstacle to pass |

