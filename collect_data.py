#!/usr/bin/env python3
"""
QLabs Data Collection Script

This script:
1. Connects to QLabs (Cityscape Lite workspace)
2. Spawns a QCar2 at a predefined location
3. Spawns obstacles (traffic cones and basic shapes) in the scene
4. Spawns a pedestrian crossing the road
5. Will be extended to collect driving data for SimLingo fine-tuning

Coordinate System (Cityscape Lite):
- World size: 500m x 500m (±250m from origin)
- Navigation area: 450m x 450m (±225m from origin)
- Origin: [0, 0, 0]
- Ground elevation: z = 0
- Actor ground offset: z = 0.005 (for vehicles)
- Pedestrian offset: z = 1.0 (origin at body center)
"""

import sys
import time
import math
import threading
from python.qvl.qlabs import QuanserInteractiveLabs
from python.qvl.qcar2 import QLabsQCar2
from python.qvl.system import QLabsSystem
from python.qvl.traffic_cone import QLabsTrafficCone
from python.qvl.basic_shape import QLabsBasicShape
from python.qvl.person import QLabsPerson


def spawn_obstacles(qlabs):
    """
    Spawn obstacles in the scene for testing lane-keeping and obstacle avoidance.

    Uses precise coordinates based on Cityscape Lite coordinate system:
    - Origin: [0, 0, 0]
    - QCar2 spawn: [0, -1.300, 0.005] facing 90° (along +Y axis)
    - Route follows existing waypoints from config.py

    Args:
        qlabs: QuanserInteractiveLabs instance

    Returns:
        List of spawned obstacle actors
    """
    obstacles = []

    print("\nSpawning obstacles...")

    # =========================================================================
    # TRAFFIC CONES - Placed along the route from config.py
    # =========================================================================
    # Route goes from [2.686, 18.498] northward and curves around
    # Place cones to test obstacle avoidance

    cone_locations = [
        # Along the straight section (Y: 18-25)
        [3.5, 22.0, 0.0],    # Cone 1: Right side of road
        [2.0, 24.0, 0.0],    # Cone 2: Left side of road

        # Along the curve (Y: 25-31, X: 4-14)
        [6.0, 27.0, 0.0],    # Cone 3: On the curve
        [10.0, 30.0, 0.0],   # Cone 4: Mid-curve

        # Along the straight section (X: 14-22, Y: ~31-40)
        [18.0, 32.5, 0.0],   # Cone 5: Right side
        [20.5, 34.0, 0.0],   # Cone 6: Right side
    ]

    for i, location in enumerate(cone_locations):
        cone = QLabsTrafficCone(qlabs)
        status = cone.spawn_id(
            actorNumber=100 + i,
            location=location,
            rotation=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            configuration=0,
            waitForConfirmation=True
        )
        if status == 0:
            print(f"  ✓ Traffic cone {i+1} spawned at [{location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f}]")
            obstacles.append(cone)
        else:
            print(f"  ✗ Failed to spawn traffic cone {i+1}")

    # =========================================================================
    # CUBE OBSTACLES - Simulating parked cars or barriers
    # =========================================================================
    # Using official parking spot coordinates from Cityscape Lite docs

    cube_obstacles = [
        # Parking Spot 1 (from official docs)
        {
            'location': [-5.987, 14.643, 0.5],
            'rotation': [0.0, 0.0, math.pi/2],  # 90 degrees
            'scale': [2.0, 1.0, 1.0],
            'color': [0.8, 0.2, 0.2],  # Dark red
            'description': 'Parked car at Parking Spot 1'
        },
        # Near the route - obstacle to avoid
        {
            'location': [8.0, 28.5, 0.5],
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.5, 1.5, 1.0],
            'color': [1.0, 0.5, 0.0],  # Orange barrier
            'description': 'Barrier on curve'
        },
        # Road Parking 1 (from official docs)
        {
            'location': [-13.093, -7.572, 0.5],
            'rotation': [0.0, 0.0, -42*math.pi/180],  # -42 degrees
            'scale': [2.0, 1.0, 1.0],
            'color': [0.2, 0.2, 0.8],  # Blue car
            'description': 'Parked car at Road Parking 1'
        },
    ]

    shape = QLabsBasicShape(qlabs)
    for i, obstacle in enumerate(cube_obstacles):
        status = shape.spawn_id(
            actorNumber=200 + i,
            location=obstacle['location'],
            rotation=obstacle['rotation'],
            scale=obstacle['scale'],
            configuration=shape.SHAPE_CUBE,
            waitForConfirmation=True
        )
        if status == 0:
            # Set material properties
            shape.set_material_properties(
                color=obstacle['color'],
                roughness=0.5,
                metallic=False,
                waitForConfirmation=True
            )
            loc = obstacle['location']
            print(f"  ✓ {obstacle['description']} at [{loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}]")
            obstacles.append(shape)
        else:
            print(f"  ✗ Failed to spawn {obstacle['description']}")

    print(f"Total obstacles spawned: {len(obstacles)}")
    return obstacles


def pedestrian_movement_loop(pedestrian, curb_1, curb_2, wait_offset_1, wait_offset_2):
    """
    Infinite loop for pedestrian movement.

    The pedestrian will continuously:
    1. Pace on curb 1 (back and forth)
    2. Cross to curb 2
    3. Wait on curb 2
    4. Pace on curb 2 (back and forth)
    5. Cross back to curb 1
    6. Wait on curb 1
    7. Repeat forever

    This function runs in a separate thread.

    Args:
        pedestrian: QLabsPerson instance
        curb_1: [x, y, z] coordinates of first curb
        curb_2: [x, y, z] coordinates of second curb
        wait_offset_1: [dx, dy, 0] offset for first waiting position
        wait_offset_2: [dx, dy, 0] offset for second waiting position
    """
    # Calculate waiting positions by adding offsets to curb positions
    curb_1_wait_1 = [curb_1[0] + wait_offset_1[0], curb_1[1] + wait_offset_1[1], curb_1[2]]
    curb_1_wait_2 = [curb_1[0] + wait_offset_2[0], curb_1[1] + wait_offset_2[1], curb_1[2]]
    curb_2_wait_1 = [curb_2[0] + wait_offset_1[0], curb_2[1] + wait_offset_1[1], curb_2[2]]
    curb_2_wait_2 = [curb_2[0] + wait_offset_2[0], curb_2[1] + wait_offset_2[1], curb_2[2]]

    cycle_count = 0
    while True:
        try:
            cycle_count += 1

            # Phase 1: Pace on curb 1
            pedestrian.move_to(location=curb_1_wait_2, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.5)
            pedestrian.move_to(location=curb_1_wait_1, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.5)
            pedestrian.move_to(location=curb_1, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.0)

            # Phase 2: Cross to curb 2
            pedestrian.move_to(location=curb_2, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(9.0)  # Crossing time: 10m / 1.2m/s ≈ 8.3s + buffer

            # Phase 3: Pace on curb 2
            pedestrian.move_to(location=curb_2_wait_2, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.5)
            pedestrian.move_to(location=curb_2_wait_1, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.5)
            pedestrian.move_to(location=curb_2, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(2.0)

            # Phase 4: Cross back to curb 1
            pedestrian.move_to(location=curb_1, speed=pedestrian.WALK, waitForConfirmation=False)
            time.sleep(9.0)  # Crossing time: 10m / 1.2m/s ≈ 8.3s + buffer

        except Exception as e:
            print(f"  ✗ Pedestrian movement error: {e}")
            time.sleep(1.0)  # Brief pause before retrying


def spawn_pedestrian_generic(qlabs, actor_number, curb_1, curb_2, crossing_direction, description):
    """
    Spawn a pedestrian that crosses between two curbs.

    The pedestrian will walk back and forth on each curb, then cross the road
    repeatedly in an infinite loop to test dynamic obstacle detection and avoidance.

    Args:
        qlabs: QuanserInteractiveLabs instance
        actor_number: Unique actor number for this pedestrian
        curb_1: [x, y, z] coordinates of first curb
        curb_2: [x, y, z] coordinates of second curb
        crossing_direction: 'horizontal' (varies X) or 'vertical' (varies Y)
        description: Human-readable description of the crossing location

    Returns:
        QLabsPerson instance or None if failed
    """
    print(f"\nSpawning pedestrian {actor_number}: {description}...")

    # Calculate crossing distance
    crossing_distance = ((curb_2[0] - curb_1[0])**2 +
                        (curb_2[1] - curb_1[1])**2)**0.5

    # Calculate waiting offsets based on crossing direction
    if crossing_direction == 'horizontal':
        # Road runs East-West, pedestrian crosses North-South
        # Pace along Y axis (north-south)
        wait_offset_1 = [0.0, -1.0, 0.0]  # 1m south
        wait_offset_2 = [0.0, 1.0, 0.0]   # 1m north
        rotation = 0.0 if curb_2[1] > curb_1[1] else 180.0  # Face crossing direction
    else:  # vertical
        # Road runs North-South, pedestrian crosses East-West
        # Pace along X axis (east-west)
        wait_offset_1 = [-1.0, 0.0, 0.0]  # 1m west
        wait_offset_2 = [1.0, 0.0, 0.0]   # 1m east
        rotation = 90.0 if curb_2[0] > curb_1[0] else 270.0  # Face crossing direction

    # Spawn pedestrian at first curb
    pedestrian = QLabsPerson(qlabs)
    status = pedestrian.spawn_id_degrees(
        actorNumber=actor_number,
        location=curb_1,
        rotation=[0.0, 0.0, rotation],
        scale=[1.0, 1.0, 1.0],
        configuration=(actor_number % 12),  # Cycle through 12 available configurations
        waitForConfirmation=True
    )

    if status == 0:
        print(f"  ✓ Spawned at curb 1: [{curb_1[0]:.1f}, {curb_1[1]:.1f}, {curb_1[2]:.1f}]")
        print(f"  → Crossing to curb 2: [{curb_2[0]:.1f}, {curb_2[1]:.1f}, {curb_2[2]:.1f}]")
        print(f"  → Distance: {crossing_distance:.1f}m ({crossing_direction} crossing)")

        # Start the pedestrian movement loop in a separate daemon thread
        movement_thread = threading.Thread(
            target=pedestrian_movement_loop,
            args=(pedestrian, curb_1, curb_2, wait_offset_1, wait_offset_2),
            daemon=True
        )
        movement_thread.start()

        print(f"  ✓ Movement thread started (running indefinitely)")
        return pedestrian
    else:
        print(f"  ✗ Failed to spawn. Status code: {status}")
        return None


def spawn_all_pedestrians(qlabs):
    """
    Spawn all pedestrians across the map at strategic crossing locations.

    Based on the Cityscape Lite map:
    - Node 13 is at approximately (0.13, 1.85) on the map (scaled coordinates)
    - Vehicle spawns at Node 13 pointing toward the roundabout (north)
    - Pedestrians are placed on STRAIGHT road sections only (not roundabouts)
    - All crossings are perpendicular to the road direction

    Pedestrian placement strategy:
    1. Near spawn (before roundabout) - straight section
    2. After roundabout exit - straight section going west
    3. Far west section - straight section
    4. South section - straight section
    5. East section - straight section

    Args:
        qlabs: QuanserInteractiveLabs instance

    Returns:
        List of spawned QLabsPerson instances
    """
    print("\n" + "-"*70)
    print("Spawning Pedestrians Across the Map")
    print("-"*70)

    pedestrians = []

    # Pedestrian 1: Near spawn (before roundabout) - Straight North-South road
    # Node 13 area, vehicle heading north toward roundabout
    # This is the straight section from Node 13 toward Node 19
    # Map shows this is around X≈0.13, Y≈1.85 (map coords) → X≈1.3, Y≈18.5 (QLabs)
    # Road width: ~10m (conservative for straight section)
    ped1 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=300,
        curb_1=[-2.0, 23.5, 1.0],   # West curb
        curb_2=[8.0, 23.5, 1.0],    # East curb
        crossing_direction='vertical',
        description="Before roundabout (Node 13→19) - Straight section"
    )
    if ped1:
        pedestrians.append(ped1)

    # Pedestrian 2: West straight section (Node 22→23 area)
    # Long straight horizontal road on the west side (runs East-West)
    # Map shows this is around X≈-2.0, Y≈3.0 (map coords) → X≈-20, Y≈30 (QLabs)
    # This is the straight section going west from the roundabout
    # Road runs HORIZONTAL (East-West), so pedestrian crosses NORTH-SOUTH (perpendicular)
    # Curbs at SAME X, DIFFERENT Y
    # crossing_direction='horizontal' because the ROAD is horizontal
    ped2 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=301,
        curb_1=[-19.841, 26.0, 1.0],   # South curb
        curb_2=[-19.841, 36.0, 1.0],   # North curb
        crossing_direction='horizontal',
        description="West section (Node 22→23) - Crossing horizontal road"
    )
    if ped2:
        pedestrians.append(ped2)

    # Pedestrian 3: Long horizontal road (waypoint 45-55 area)
    # Straight East-West road in the middle-north section
    # Map shows this is around X≈0.0, Y≈4.5 (map coords) → X≈0, Y≈45 (QLabs)
    # Road runs EAST-WEST (horizontal), so pedestrian crosses NORTH-SOUTH (perpendicular)
    # Curbs should be at SAME X, DIFFERENT Y (north and south sides of the road)
    # crossing_direction='horizontal' because the ROAD is horizontal
    ped3 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=302,
        curb_1=[0.0, 42.0, 1.0],   # South curb
        curb_2=[0.0, 48.0, 1.0],   # North curb
        crossing_direction='horizontal',
        description="North horizontal road (waypoint 45-55) - Crossing perpendicular"
    )
    if ped3:
        pedestrians.append(ped3)

    # Pedestrian 4: South section (Node 0→1 area)
    # Straight vertical road on the south side
    # Map shows this is around X≈0.0, Y≈0.0 (map coords) → X≈0, Y≈0 (QLabs)
    ped4 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=303,
        curb_1=[-5.0, 5.0, 1.0],   # West curb
        curb_2=[5.0, 5.0, 1.0],    # East curb
        crossing_direction='vertical',
        description="South section (Node 0→1) - Straight vertical road"
    )
    if ped4:
        pedestrians.append(ped4)

    # Pedestrian 5: East section (Node 4→5 area)
    # Straight vertical road on the east side
    # Map shows this is around X≈2.0, Y≈1.5 (map coords) → X≈20, Y≈15 (QLabs)
    # Road runs NORTH-SOUTH (vertical), so pedestrian crosses EAST-WEST (perpendicular)
    # Curbs should be at SAME Y, DIFFERENT X (east and west sides of the road)
    # crossing_direction='vertical' because the ROAD is vertical
    ped5 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=304,
        curb_1=[17.0, 12.0, 1.0],  # West curb
        curb_2=[23.0, 12.0, 1.0],  # East curb
        crossing_direction='vertical',
        description="East section (Node 4→5) - Crossing perpendicular"
    )
    if ped5:
        pedestrians.append(ped5)

    print("-"*70)
    print(f"Total pedestrians spawned: {len(pedestrians)}/5")

    return pedestrians


def main():
    """Main function to setup QLabs environment and spawn QCar2 with obstacles."""

    # Create connection to QLabs
    print("="*70)
    print("QLabs Data Collection - Cityscape Lite")
    print("="*70)
    print("\nConnecting to QLabs...")
    qlabs = QuanserInteractiveLabs()

    # Connect to QLabs on localhost
    if not qlabs.open("localhost"):
        print("ERROR: Unable to connect to QLabs.")
        print("Make sure QLabs is running with Cityscape Lite workspace.")
        return False

    print("✓ Connected to QLabs successfully!")

    # Create system object to set title
    system = QLabsSystem(qlabs)
    system.set_title_string("QCar2 Data Collection - Cityscape Lite")

    # Destroy any existing actors
    print("\nCleaning up existing actors...")
    num_destroyed = qlabs.destroy_all_spawned_actors()
    print(f"✓ Destroyed {num_destroyed} existing actors")

    # =========================================================================
    # SPAWN QCAR2
    # =========================================================================
    # Using spawn location from config.py (Node 13 - roundabout route start)
    # Location: [2.686, 18.498, 0.005]
    # Rotation: 90° (1.5708 radians) - facing along +Y axis (north)

    print("\n" + "-"*70)
    print("Spawning QCar2...")
    print("-"*70)
    qcar = QLabsQCar2(qlabs)

    spawn_location = [2.686, 18.498, 0.005]  # Node 13 (matches config.py)
    spawn_rotation = [0.0, 0.0, 90.0]  # 90 degrees yaw (facing along +Y axis)

    status = qcar.spawn_id_degrees(
        actorNumber=0,
        location=spawn_location,
        rotation=spawn_rotation,
        scale=[1.0, 1.0, 1.0],
        configuration=0,
        waitForConfirmation=True
    )

    if status == 0:
        print(f"✓ QCar2 spawned at [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}, {spawn_location[2]:.3f}]")
        print(f"  Rotation: {spawn_rotation[2]}° (facing north along +Y axis)")
    else:
        print(f"✗ ERROR: Failed to spawn QCar2. Status code: {status}")
        print("  Status codes: 0=success, 1=class not available, 2=actor in use, 3=unknown")
        qlabs.close()
        return False

    # =========================================================================
    # SPAWN OBSTACLES
    # =========================================================================
    print("\n" + "-"*70)
    obstacles = spawn_obstacles(qlabs)
    print("-"*70)

    # =========================================================================
    # SPAWN PEDESTRIANS
    # =========================================================================
    pedestrians = spawn_all_pedestrians(qlabs)

    # =========================================================================
    # CAMERA VIEW
    # =========================================================================
    print("\nSetting camera view...")
    qcar.possess(qcar.CAMERA_TRAILING)
    print("✓ Viewing from QCar2 trailing camera")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SCENE SETUP COMPLETE")
    print("="*70)
    print(f"QCar2:        Spawned at Node 13 [2.686, 18.498]")
    print(f"Obstacles:    {len(obstacles)} objects (cones + cubes)")
    print(f"Pedestrians:  {len(pedestrians)}/5 active")
    print("\nPedestrian Locations (on straight roads only):")
    print("  1. Before roundabout (Y=23.5)  - Near spawn")
    print("  2. West section (X=-19.8)      - After roundabout")
    print("  3. North horizontal (X=5.0)    - Long straight road")
    print("  4. South section (Y=5.0)       - Bottom of map")
    print("  5. East section (Y=12.0)       - Right side of map")
    print("\nCoordinate System:")
    print("  - World size: 500m × 500m (±250m from origin)")
    print("  - Origin: [0, 0, 0]")
    print("  - QCar2 facing: +Y axis (north)")
    print("\nPress Ctrl+C to exit...")
    print("="*70)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\nShutting down...")

    # Clean up
    qlabs.close()
    print("✓ Connection closed. Done!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

