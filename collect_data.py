#!/usr/bin/env python3
"""
QLabs Data Collection Script

This script:
1. Connects to QLabs (Cityscape Lite workspace)
2. Spawns a QCar2 at a predefined location
3. Spawns pedestrians crossing the roads
4. Will be extended to collect driving data for SimLingo fine-tuning

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
import threading
from python.qvl.qlabs import QuanserInteractiveLabs
from python.qvl.qcar2 import QLabsQCar2
from python.qvl.system import QLabsSystem
from python.qvl.person import QLabsPerson


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
    - Vehicle spawns at Node 1 pointing north
    - Pedestrians are placed on STRAIGHT road sections only (not roundabouts)
    - All crossings are perpendicular to the road direction
    - Each pedestrian crosses both parallel roads (from outer curb to outer curb)

    Pedestrian placement strategy:
    1. Near Node 13 - crossing edges 12→7 and 6→13
    2. West section - crossing edges 22→9 and 8→23
    3. North section - crossing edges 23→21 and 20→22
    4. East section - crossing edges 15→6 and 7→14

    Args:
        qlabs: QuanserInteractiveLabs instance

    Returns:
        List of spawned QLabsPerson instances
    """
    print("\n" + "-"*70)
    print("Spawning Pedestrians Across the Map")
    print("-"*70)

    pedestrians = []

    # Pedestrian 1: Near Node 13 (Edges 6→13 and 12→7 - parallel roads)
    # Two parallel road edges running North-South near Node 13
    # Pedestrian crosses BOTH roads East-West (perpendicular)
    # Crosses from outer curb of edge 12→7 to outer curb of edge 6→13
    # Total crossing: ~7.7m (sidewalk + road + yellow divider + road + sidewalk)
    # crossing_direction='vertical' because the ROADS run North-South
    ped1 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=300,
        curb_1=[-2.500, 18.498, 1.0],   # West curb (outer edge of 12→7)
        curb_2=[5.186, 18.426, 1.0],    # East curb (outer edge of 6→13)
        crossing_direction='vertical',
        description="Near spawn (Edges 12→7 ↔ 6→13) - Crossing both parallel roads"
    )
    if ped1:
        pedestrians.append(ped1)

    # Pedestrian 2: West section (Edges 22→9 and 8→23 - parallel roads)
    # Two parallel road edges running mostly North-South
    # Pedestrian crosses BOTH roads East-West (perpendicular)
    # Crosses from outer curb of edge 22→9 to outer curb of edge 8→23
    # Total crossing: ~7.7m (sidewalk + road + yellow divider + road + sidewalk)
    # crossing_direction='vertical' because the ROADS run North-South
    ped2 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=301,
        curb_1=[-21.891, 14.043, 1.0],   # West curb (outer edge of 22→9)
        curb_2=[-14.508, 16.190, 1.0],   # East curb (outer edge of 8→23)
        crossing_direction='vertical',
        description="West section (Edges 22→9 ↔ 8→23) - Crossing both parallel roads"
    )
    if ped2:
        pedestrians.append(ped2)

    # Pedestrian 3: North section (Edges 23→21 and 20→22 - parallel roads)
    # Two parallel road edges running East-West near the top of the map
    # Pedestrian crosses BOTH roads North-South (perpendicular, vertical crossing)
    # Crosses from outer curb of edge 23→21 to outer curb of edge 20→22
    # Total crossing: ~7.7m (sidewalk + road + yellow divider + road + sidewalk)
    # crossing_direction='horizontal' because the ROADS run East-West
    ped3 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=302,
        curb_1=[-0.019, 39.767, 1.0],   # South curb (outer edge of 23→21)
        curb_2=[-0.019, 47.474, 1.0],   # North curb (outer edge of 20→22)
        crossing_direction='horizontal',
        description="North section (Edges 23→21 ↔ 20→22) - Crossing both parallel roads"
    )
    if ped3:
        pedestrians.append(ped3)

    # Pedestrian 4: East section (Edges 15→6 and 7→14 - parallel roads)
    # Two parallel road edges running mostly North-South near Node 15
    # Pedestrian crosses BOTH roads East-West (perpendicular, horizontal crossing)
    # Crosses from outer curb of edge 15→6 to outer curb of edge 7→14
    # Total crossing: ~7.9m (sidewalk + road + yellow divider + road + sidewalk)
    # crossing_direction='vertical' because the ROADS run North-South
    ped4 = spawn_pedestrian_generic(
        qlabs=qlabs,
        actor_number=303,
        curb_1=[16.921, 15.008, 1.0],  # West curb (outer edge of 15→6)
        curb_2=[24.777, 15.008, 1.0],  # East curb (outer edge of 7→14)
        crossing_direction='vertical',
        description="East section (Edges 15→6 ↔ 7→14) - Crossing both parallel roads"
    )
    if ped4:
        pedestrians.append(ped4)

    print("-"*70)
    print(f"Total pedestrians spawned: {len(pedestrians)}/4")

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
    # Spawn at Node 1
    # Location: [2.686, 0.814, 0.005]
    # Rotation: 90° - facing along +Y axis (north)

    print("\n" + "-"*70)
    print("Spawning QCar2...")
    print("-"*70)
    qcar = QLabsQCar2(qlabs)

    spawn_location = [2.686, 0.814, 0.005]  # Node 1
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
    print(f"QCar2:        Spawned at Node 1 [2.686, 0.814]")
    print(f"Pedestrians:  {len(pedestrians)}/4 active")
    print("\nPedestrian Crossing Locations:")
    print("  1. Near spawn (Edges 12→7 ↔ 6→13)   - Crossing both parallel roads")
    print("  2. West section (Edges 22→9 ↔ 8→23) - Crossing both parallel roads")
    print("  3. North section (Edges 23→21 ↔ 20→22) - Crossing both parallel roads")
    print("  4. East section (Edges 15→6 ↔ 7→14) - Crossing both parallel roads")
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

