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
import math
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from python.qvl.qlabs import QuanserInteractiveLabs
from python.qvl.qcar2 import QLabsQCar2
from python.qvl.system import QLabsSystem
from python.qvl.person import QLabsPerson
from hal.products.mats import SDCSRoadMap


def circular_qcar_loop(qcar, route_nodes):
    """
    Infinite loop for QCar2 to follow a circular route.

    Uses Stanley controller for steering (similar to QCarDriveController in HAL library).

    Args:
        qcar: QLabsQCar2 instance
        route_nodes: List of node IDs forming a circular route (e.g., [0, 2, 4, 6])
    """
    print(f"  Starting circular route loop for QCar2...")

    # Initialize roadmap
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

    # Generate full route waypoints
    all_waypoints = []
    for i in range(len(route_nodes)):
        from_node = route_nodes[i]
        to_node = route_nodes[(i + 1) % len(route_nodes)]  # Wrap around for circular route

        # Generate path for this edge
        path = roadmap.generate_path([from_node, to_node])
        x_coords = path[0, :] * 10.0  # Scale to QLabs coordinates
        y_coords = path[1, :] * 10.0

        # Add waypoints (skip first point if not the first edge to avoid duplicates)
        start_idx = 1 if i > 0 else 0
        for j in range(start_idx, len(x_coords)):
            all_waypoints.append([x_coords[j], y_coords[j]])

    all_waypoints = np.array(all_waypoints)
    print(f"  Generated {len(all_waypoints)} waypoints for circular route")

    # Control parameters
    # NOTE: QCar2 in QLabs is 10x scale, so speeds are in full-scale m/s
    target_speed = 2.5  # m/s (full-scale speed) - increased for smoother motion
    k_stanley = 0.3  # Stanley controller gain (lower for smoother control)
    max_steering_angle = np.pi / 6  # 30 degrees max
    lookahead_distance = 3.0  # meters - look ahead for smoother path following
    current_waypoint_idx = 0

    print(f"  Target speed: {target_speed} m/s (full-scale)")
    print(f"  Stanley gain: {k_stanley}")
    print(f"  Lookahead: {lookahead_distance}m")
    print(f"  Max steering: {np.degrees(max_steering_angle):.1f}°")

    # Wait before starting
    time.sleep(2.0)
    print(f"  Circular QCar2 starting movement...")

    # Error tracking
    consecutive_errors = 0
    max_consecutive_errors = 10

    # Control loop timing
    dt = 0.1  # 10 Hz control loop
    iteration = 0

    # State variables
    current_pos = None
    current_yaw = None
    steering_angle = 0.0

    while True:
        try:
            iteration += 1

            # Query state every iteration (needed for accurate control)
            try:
                success, location, rotation, _, _ = qcar.set_velocity_and_request_state(
                    forward=target_speed if current_pos is not None else 0.0,
                    turn=steering_angle,
                    headlights=True,
                    leftTurnSignal=False,
                    rightTurnSignal=False,
                    brakeSignal=False,
                    reverseSignal=False
                )
            except Exception:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print(f"  ✗ Too many consecutive errors, stopping thread")
                    break
                time.sleep(0.2)
                continue

            if not success:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print(f"  ✗ Too many consecutive failures, stopping thread")
                    break
                time.sleep(0.2)
                continue

            # Reset error counter on success
            consecutive_errors = 0
            current_pos = np.array([location[0], location[1]])
            current_yaw = rotation[2]

            # Find nearest waypoint on path
            distances = np.linalg.norm(all_waypoints - current_pos, axis=1)
            nearest_idx = np.argmin(distances)
            current_waypoint_idx = nearest_idx

            # Find lookahead waypoint (for smoother path following)
            lookahead_idx = current_waypoint_idx
            accumulated_dist = 0.0

            for i in range(1, min(100, len(all_waypoints))):
                idx = (current_waypoint_idx + i) % len(all_waypoints)
                prev_idx = (current_waypoint_idx + i - 1) % len(all_waypoints)

                segment_dist = np.linalg.norm(all_waypoints[idx] - all_waypoints[prev_idx])
                accumulated_dist += segment_dist

                if accumulated_dist >= lookahead_distance:
                    lookahead_idx = idx
                    break

            # Get target waypoint (lookahead point)
            target_wp = all_waypoints[lookahead_idx]

            # Calculate cross-track error (perpendicular distance to path)
            # Vector from current position to target waypoint
            path_vector = target_wp - current_pos
            path_distance = np.linalg.norm(path_vector)

            if path_distance < 0.01:  # Too close, skip
                time.sleep(dt)
                continue

            # Path heading
            path_heading = math.atan2(path_vector[1], path_vector[0])

            # Heading error
            heading_error = path_heading - current_yaw

            # Normalize to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            # Cross-track error (simplified - distance to target)
            cross_track_error = path_distance * math.sin(heading_error)

            # Stanley controller: steering = heading_error + atan(k * cross_track_error / velocity)
            # For low speeds, use simplified version
            if target_speed > 0.01:
                steering_correction = math.atan(k_stanley * cross_track_error / target_speed)
            else:
                steering_correction = 0.0

            steering_angle = heading_error + steering_correction

            # NEGATE steering angle - flip left/right convention
            # (Testing if QCar2 convention is opposite to what we expect)
            steering_angle = -steering_angle

            # Clip steering angle
            steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

            # Control loop rate - 10 Hz
            time.sleep(dt)

        except Exception as e:
            print(f"  ✗ Circular QCar2 error: {e}")
            time.sleep(1.0)


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
    1. South section - crossing edges 12→7 and 6→13
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

    # Pedestrian 1: South section (Edges 6→13 and 12→7 - parallel roads)
    # Two parallel road edges running North-South
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
    # SPAWN CIRCULAR ROUTE QCAR2
    # =========================================================================
    print("\n" + "-"*70)
    print("Spawning Circular Route QCar2...")
    print("-"*70)

    circular_qcar = QLabsQCar2(qlabs)
    circular_spawn_location = [0.000, 1.302, 0.005]  # Node 0
    circular_spawn_rotation = [0.0, 0.0, -90.0]  # -90 degrees (facing south)

    status = circular_qcar.spawn_id_degrees(
        actorNumber=1,  # Different actor number from main QCar2
        location=circular_spawn_location,
        rotation=circular_spawn_rotation,
        scale=[1.0, 1.0, 1.0],
        configuration=0,
        waitForConfirmation=True
    )

    if status == 0:
        print(f"✓ Circular QCar2 spawned at Node 0 [{circular_spawn_location[0]:.3f}, {circular_spawn_location[1]:.3f}]")
        print(f"  Route: 0 → 2 → 4 → 6 → 0 (infinite loop)")
        print(f"  Movement will start after pedestrians are spawned")
    else:
        print(f"✗ Failed to spawn circular QCar2. Status code: {status}")
        circular_qcar = None

    print("-"*70)

    # =========================================================================
    # SPAWN ROUNDABOUT QCAR2
    # =========================================================================
    print("\n" + "-"*70)
    print("Spawning Roundabout QCar2...")
    print("-"*70)

    roundabout_qcar = QLabsQCar2(qlabs)
    roundabout_spawn_location = [9.076, 37.098, 0.005]  # Node 16
    roundabout_spawn_rotation = [0.0, 0.0, -80.6]  # -80.6 degrees

    status = roundabout_qcar.spawn_id_degrees(
        actorNumber=2,  # Different actor number from other QCar2s
        location=roundabout_spawn_location,
        rotation=roundabout_spawn_rotation,
        scale=[1.0, 1.0, 1.0],
        configuration=0,
        waitForConfirmation=True
    )

    if status == 0:
        print(f"✓ Roundabout QCar2 spawned at Node 16 [{roundabout_spawn_location[0]:.3f}, {roundabout_spawn_location[1]:.3f}]")
        print(f"  Route: 16 → 17 → 16 (infinite loop)")
        print(f"  Movement will start after pedestrians are spawned")
    else:
        print(f"✗ Failed to spawn roundabout QCar2. Status code: {status}")
        roundabout_qcar = None

    print("-"*70)

    # =========================================================================
    # SPAWN PEDESTRIANS
    # =========================================================================
    pedestrians = spawn_all_pedestrians(qlabs)

    # =========================================================================
    # START CIRCULAR QCAR2 MOVEMENT
    # =========================================================================
    if circular_qcar:
        print("\n" + "-"*70)
        print("Starting Circular QCar2 Movement...")
        print("-"*70)

        route_nodes = [0, 2, 4, 6]
        circular_thread = threading.Thread(
            target=circular_qcar_loop,
            args=(circular_qcar, route_nodes),
            daemon=True
        )
        circular_thread.start()
        print(f"✓ Circular route thread started")
        print("-"*70)

    # =========================================================================
    # START ROUNDABOUT QCAR2 MOVEMENT
    # =========================================================================
    if roundabout_qcar:
        # Small delay to avoid simultaneous communication with circular car
        time.sleep(1.0)

        print("\n" + "-"*70)
        print("Starting Roundabout QCar2 Movement...")
        print("-"*70)

        route_nodes = [16, 17]
        roundabout_thread = threading.Thread(
            target=circular_qcar_loop,
            args=(roundabout_qcar, route_nodes),
            daemon=True
        )
        roundabout_thread.start()
        print(f"✓ Roundabout route thread started")
        print("-"*70)

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
    print(f"QCar2 (Main):       Spawned at Node 1 [2.686, 0.814]")
    print(f"QCar2 (Circular):   {'Spawned at Node 0 [0.000, 1.302]' if circular_qcar else 'Failed to spawn'}")
    if circular_qcar:
        print(f"                    Route: 0 → 2 → 4 → 6 → 0 (infinite loop)")
    print(f"QCar2 (Roundabout): {'Spawned at Node 16 [9.076, 37.098]' if roundabout_qcar else 'Failed to spawn'}")
    if roundabout_qcar:
        print(f"                    Route: 16 → 17 → 16 (infinite loop)")
    print(f"Pedestrians:        {len(pedestrians)}/4 active")
    print("\nPedestrian Crossing Locations:")
    print("  1. South section (Edges 12→7 ↔ 6→13)   - Crossing both parallel roads")
    print("  2. West section (Edges 22→9 ↔ 8→23)    - Crossing both parallel roads")
    print("  3. North section (Edges 23→21 ↔ 20→22) - Crossing both parallel roads")
    print("  4. East section (Edges 15→6 ↔ 7→14)    - Crossing both parallel roads")
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

