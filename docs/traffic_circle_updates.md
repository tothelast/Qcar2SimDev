# Traffic Circle Scene Updates

## Summary of Changes

### 1. Extended Route (✓ Completed)
- **Previous route**: Nodes 14 → 16 → 17 → 15
- **New route**: Nodes 14 → 16 → 17 → 15 → 5
- **Improvements**:
  - Route now starts 3 meters before node 14 (spawn at Y=26.67 instead of Y=29.67)
  - Extended to node 5, adding approximately 20 meters to the route
  - Total route length: 75.4m (was 54.7m)
  - Total waypoints: 72 (was 53)

### 2. Stop Sign (✓ Completed)
**Actor**: `traffic_circle_stop_sign`
- **Location**: [24.0, 35.0, 0.001]
- **Rotation**: [0.0, 0.0, 90.0] (facing east)
- **Actor Number**: 201
- **Position**: Right curbside before the roundabout entrance (before node 16)

### 3. East Crosswalk (✓ Completed)
**Actor**: `east_crosswalk`
- **Location**: [20.849, 15.008, 0.01] (middle of pedestrian crossing)
- **Rotation**: [0.0, 0.0, 0.0] (horizontal orientation)
- **Scale**: [0.04, 2.5, 1.0] (same as other crosswalks)
- **Actor Number**: 422
- **Position**: Across the east pedestrian crossing path

### 4. Traffic Light (✓ Completed)
**Actor**: `east_crosswalk_light`
- **Location**: [19.84, 17.5, 0.0] (on ego vehicle's path before crosswalk)
- **Rotation**: [0.0, 0.0, -90.0] (facing south, toward traffic)
- **Actor Number**: 423
- **Configuration**: Traffic light with red/green cycle
- **Cycle Settings**:
  - Red: 10.0 seconds
  - Green: 5.0 seconds
  - Yellow: 0.0 seconds
  - Starts on: Red (color_index: 1)

### 5. Scene Configuration (✓ Completed)
**File**: `config/scenes/training/05_traffic_circle.json`

Updated actors list:
```json
"actors": [
  "roundabout_car",
  "east_pedestrian",
  "traffic_circle_stop_sign",
  "east_crosswalk",
  "east_crosswalk_light"
]
```

## Files Created/Modified

### Created Files:
1. `config/actors/static/traffic_circle_stop_sign.json`
2. `config/actors/static/east_crosswalk.json`
3. `config/actors/static/east_crosswalk_light.json`
4. `tools/generate_traffic_circle_extended.py`

### Modified Files:
1. `config/routes/traffic_circle.json` (backup created)
2. `config/scenes/training/05_traffic_circle.json`

## Testing Notes

The scene should now feature:
- Ego vehicle starting earlier (3m before node 14) and ending at node 5
- A stop sign on the right side before entering the roundabout
- A crosswalk marking where the east pedestrian crosses
- A traffic light controlling the crossing (cycles between red and green)
- All existing actors (roundabout car and east pedestrian) remain active

## Coordinate Reference

**Key Nodes**:
- Node 14: (22.55, 29.67) - heading 90° (north)
- Node 16: (9.08, 37.10) - heading -80.6°
- Node 17: (14.66, 31.51) - heading -9.4°
- Node 15: (19.84, 18.50) - heading -90° (south)
- Node 5: (19.84, 0.81) - heading -90° (south)

**East Pedestrian**:
- Curb 1: (16.921, 15.008)
- Curb 2: (24.777, 15.008)
- Crossing: Horizontal (east-west)
