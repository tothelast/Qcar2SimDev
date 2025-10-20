# Scene Actor Control System

This document describes the scene actor control system for managing which actors spawn in the QLabs Cityscape Lite environment.

## Overview

The scene actor control system allows you to selectively enable/disable different actors in the simulation environment:
- **Autonomous vehicles** (circular route car, roundabout route car)
- **Pedestrians** (4 total: south, west, north, east)
- **Static actors** (parked vehicles, stop signs)

**By default, ALL actors are DISABLED** to provide a clean testing environment with just the ego vehicle and the route.

**This system is shared between data collection and inference** - both use the same `SceneManager` from `core/scene_manager.py`.

## Architecture

The system consists of three main components:

### 1. `SceneConfig` Class (`core/config.py`)

A reusable configuration class that holds flags for each actor type:

```python
from core.config import SceneConfig

scene = SceneConfig()
scene.spawn_circular_qcar = True
scene.spawn_pedestrians = True
scene.spawn_parked_vehicles = True
```

**Available flags:**
- `spawn_circular_qcar` - Autonomous car on circular route (Node 0→2→4→6→0)
- `spawn_roundabout_qcar` - Autonomous car on roundabout route (Node 16→17→16)
- `spawn_pedestrians` - All pedestrians (enables all 4)
- `spawn_south_pedestrian` - South pedestrian only
- `spawn_west_pedestrian` - West pedestrian only
- `spawn_north_pedestrian` - North pedestrian only
- `spawn_east_pedestrian` - East pedestrian only
- `spawn_parked_vehicles` - All parked vehicles (4 total)
- `spawn_stop_signs` - Stop signs (1 total)

**Helper methods:**
- `enable_all()` - Enable all actors
- `disable_all()` - Disable all actors (default state)
- `__str__()` - Get human-readable summary of enabled actors

### 2. Command-Line Arguments

Both `data_collection/collect_data.py` and `inference/main.py` support the same command-line flags for enabling actors:

**Data Collection:**
```bash
# Empty scene (just ego vehicle and route)
python data_collection/collect_data.py --route simple_straight

# Scene with circular car and all pedestrians
python data_collection/collect_data.py --route roundabout_navigation --circular-car --pedestrians

# Scene with specific pedestrians only
python data_collection/collect_data.py --route long_route --south-pedestrian --west-pedestrian

# Full scene with all actors
python data_collection/collect_data.py --route complex_route --all-actors
```

**Inference:**
```bash
# Empty scene (just ego vehicle and route)
python inference/main.py --route simple_straight

# Scene with circular car and all pedestrians
python inference/main.py --route roundabout_navigation --circular-car --pedestrians

# Full scene with all actors
python inference/main.py --route complex_route --all-actors
```

**Available flags:**
- `--all-actors` - Enable all scene actors
- `--circular-car` - Enable autonomous car on circular route
- `--roundabout-car` - Enable autonomous car on roundabout route
- `--pedestrians` - Enable all pedestrians
- `--south-pedestrian` - Enable south pedestrian only
- `--west-pedestrian` - Enable west pedestrian only
- `--north-pedestrian` - Enable north pedestrian only
- `--east-pedestrian` - Enable east pedestrian only
- `--parked-vehicles` - Enable parked vehicles
- `--stop-signs` - Enable stop signs

### 3. `SceneManager` (`core/scene_manager.py`)

The `SceneManager` is now in `core/` and shared between data collection and inference.

The `spawn_all_pedestrians()` method accepts individual pedestrian flags:

```python
pedestrians = scene_manager.spawn_all_pedestrians(
    spawn_south=True,
    spawn_west=False,
    spawn_north=True,
    spawn_east=False
)
```

## Usage Examples

### Data Collection

#### Example 1: Empty Scene (Default)
```bash
python data_collection/collect_data.py --route simple_straight
```
- Spawns: Ego vehicle only
- Use case: Clean route following without distractions

#### Example 2: Light Traffic
```bash
python data_collection/collect_data.py --route roundabout_navigation --circular-car
```
- Spawns: Ego vehicle + 1 autonomous car
- Use case: Basic interaction scenarios

#### Example 3: Pedestrian Crossings
```bash
python data_collection/collect_data.py --route long_route --pedestrians
```
- Spawns: Ego vehicle + 4 pedestrians
- Use case: Pedestrian detection and yielding

#### Example 4: Specific Pedestrians
```bash
python data_collection/collect_data.py --route complex_route --south-pedestrian --north-pedestrian
```
- Spawns: Ego vehicle + 2 specific pedestrians
- Use case: Testing specific crossing scenarios

#### Example 5: Full Scene
```bash
python data_collection/collect_data.py --route full_circuit --all-actors
```
- Spawns: All actors (2 cars, 4 pedestrians, 4 parked vehicles, 1 stop sign)
- Use case: Complex, realistic driving scenarios

### Programmatic Usage

You can also configure scenes programmatically:

```python
from core.config import SimlingoQCar2Config

# Create config
config = SimlingoQCar2Config()

# Configure scene
config.scene.spawn_circular_qcar = True
config.scene.spawn_south_pedestrian = True
config.scene.spawn_parked_vehicles = True

# Check configuration
print(config.scene)  # Output: SceneConfig: circular_qcar, pedestrians(south), parked_vehicles

# Use in your code
if config.scene.spawn_circular_qcar:
    circular_qcar = scene_manager.spawn_circular_qcar()
```

## Future Extensions

This system is designed to be reusable for inference code. Future enhancements could include:

### 1. Scenario Files (JSON)
Define complete scenes in JSON files:

```json
{
  "name": "heavy_traffic",
  "description": "Heavy traffic scenario with all actors",
  "actors": {
    "circular_qcar": true,
    "roundabout_qcar": true,
    "pedestrians": ["south", "west", "north", "east"],
    "parked_vehicles": true,
    "stop_signs": true
  }
}
```

### 2. Dynamic Actor Placement
Allow custom positions for actors:

```python
config.scene.add_custom_vehicle(
    location=[10.0, 20.0, 0.0],
    rotation=[0.0, 0.0, 90.0],
    route=[node1, node2, node3]
)
```

### 3. Inference Integration
Use the same system in inference code:

```python
# inference/main.py
controller = SimlingoQCar2Controller(
    route_name='roundabout_navigation',
    scene_config='heavy_traffic'  # Load from scenario file
)
```

## Implementation Details

### Default Behavior
- All actors are **disabled by default**
- Ego vehicle (teleop QCar2) **always spawns**
- Route visualization (green line) **always shows**

### Actor Spawning Logic
The `collect_data.py` script checks `config.scene` flags before spawning:

```python
# Only spawn if enabled
if config.scene.spawn_circular_qcar:
    circular_qcar = scene_manager.spawn_circular_qcar()

if config.scene.spawn_pedestrians or config.scene.spawn_south_pedestrian:
    pedestrians = scene_manager.spawn_all_pedestrians(
        spawn_south=config.scene.spawn_south_pedestrian or config.scene.spawn_pedestrians,
        # ... other flags
    )
```

### Pedestrian Control
Pedestrians can be controlled:
1. **All at once**: `--pedestrians` enables all 4
2. **Individually**: `--south-pedestrian`, `--west-pedestrian`, etc.
3. **Combination**: `--pedestrians` overrides individual flags

## Benefits

1. **Clean Testing**: Start with empty scenes, add complexity as needed
2. **Reproducibility**: Exact same scene configuration via command-line flags
3. **Flexibility**: Mix and match actors for different scenarios
4. **Reusability**: Same system works for data collection and inference
5. **Simplicity**: Easy-to-use command-line interface
6. **Extensibility**: Easy to add new actor types or scenarios

## Actor Locations

### Autonomous Vehicles
- **Circular QCar**: Spawns at Node 0 [0.000, 1.302], follows route 0→2→4→6→0 (infinite loop)
- **Roundabout QCar**: Spawns at Node 16 [9.076, 37.098], follows route 16→17→16 (infinite loop)

### Pedestrians
- **South**: Crosses at [−2.5, 18.5] ↔ [5.2, 18.4] (Edges 12→7 ↔ 6→13)
- **West**: Crosses at [−21.9, 14.0] ↔ [−14.5, 16.2] (Edges 22→9 ↔ 8→23)
- **North**: Crosses at [−0.02, 39.8] ↔ [−0.02, 47.5] (Edges 23→21 ↔ 20→22)
- **East**: Crosses at [16.9, 15.0] ↔ [24.8, 15.0] (Edges 15→6 ↔ 7→14)

### Static Actors
- **Parked Vehicles**: 4 vehicles at strategic parking spots
- **Stop Signs**: 1 sign at roundabout approach (Node 19 area)

