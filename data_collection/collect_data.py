#!/usr/bin/env python3
"""
QLabs Data Collection Script with Teleop Control

This script sets up the QLabs Cityscape Lite environment for expert data collection:
1. Connects to QLabs (Cityscape Lite workspace)
2. Spawns a teleop-controlled QCar2 at Node 1 for expert data collection
3. Spawns autonomous QCar2s on circular and roundabout routes
4. Spawns pedestrians crossing the roads
5. Spawns parked vehicles and stop signs
6. Enables keyboard-based teleop control for collecting driving demonstrations

Teleop Controls:
- Arrow Up / W: Accelerate forward
- Arrow Down / S: Brake / Reverse
- Arrow Left / A: Steer left
- Arrow Right / D: Steer right
- Space: Emergency stop
- Q / ESC: Quit

Coordinate System (Cityscape Lite):
- World size: 500m x 500m (±250m from origin)
- Navigation area: 450m x 450m (±225m from origin)
- Origin: [0, 0, 0]
- Ground elevation: z = 0
- Actor ground offset: z = 0.005 (for vehicles)
- Pedestrian offset: z = 1.0 (origin at body center)
"""

import sys
from pathlib import Path

# Add python directory to path for QLabs SDK
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.system import QLabsSystem

from teleop_controller import TeleopController, teleop_control_loop
from scene_manager import SceneManager


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
    # SPAWN MAIN TELEOP QCAR2
    # =========================================================================
    # Spawn at Node 1
    # Location: [2.686, 0.814, 0.005]
    # Rotation: 90° - facing along +Y axis (north)

    print("\n" + "-"*70)
    print("Spawning Main QCar2 (Teleop Control)...")
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
        print(f"✓ Main QCar2 spawned at [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}, {spawn_location[2]:.3f}]")
        print(f"  Rotation: {spawn_rotation[2]}° (facing north along +Y axis)")
        print(f"  Mode: TELEOP CONTROL")
    else:
        print(f"✗ ERROR: Failed to spawn main QCar2. Status code: {status}")
        print("  Status codes: 0=success, 1=class not available, 2=actor in use, 3=unknown")
        qlabs.close()
        return False

    print("-"*70)

    # =========================================================================
    # SETUP SCENE WITH SCENE MANAGER
    # =========================================================================
    print("\n" + "="*70)
    print("SETTING UP SCENE ACTORS")
    print("="*70)

    scene_manager = SceneManager(qlabs)

    # Spawn autonomous QCar2 vehicles
    circular_qcar = scene_manager.spawn_circular_qcar()
    roundabout_qcar = scene_manager.spawn_roundabout_qcar()

    # Spawn static actors
    parked_vehicles = scene_manager.spawn_parked_vehicles()
    stop_signs = scene_manager.spawn_stop_signs()

    # Spawn pedestrians
    pedestrians = scene_manager.spawn_all_pedestrians()

    # Start autonomous vehicle movements
    if circular_qcar:
        scene_manager.start_circular_qcar()

    if roundabout_qcar:
        scene_manager.start_roundabout_qcar()

    # =========================================================================
    # CAMERA VIEW
    # =========================================================================
    print("\nSetting camera view...")
    qcar.possess(qcar.CAMERA_TRAILING)
    print("✓ Viewing from QCar2 trailing camera")

    # =========================================================================
    # TELEOP CONTROL SETUP
    # =========================================================================
    print("\n" + "-"*70)
    print("Setting up Teleop Control...")
    print("-"*70)

    teleop_controller = TeleopController()
    print("✓ Teleop controller initialized")
    print("\nKeyboard Controls:")
    print("  Arrow Up / W:    Accelerate forward")
    print("  Arrow Down / S:  Brake / Reverse")
    print("  Arrow Left / A:  Steer left")
    print("  Arrow Right / D: Steer right")
    print("  Space:           Emergency stop")
    print("  Q / ESC:         Quit")
    print("-"*70)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SCENE SETUP COMPLETE")
    print("="*70)
    print(f"QCar2 (Main):       Spawned at Node 1 [2.686, 0.814] - TELEOP CONTROL")
    print(f"QCar2 (Circular):   {'Spawned at Node 0 [0.000, 1.302]' if circular_qcar else 'Failed to spawn'}")
    if circular_qcar:
        print(f"                    Route: 0 → 2 → 4 → 6 → 0 (infinite loop)")
    print(f"QCar2 (Roundabout): {'Spawned at Node 16 [9.076, 37.098]' if roundabout_qcar else 'Failed to spawn'}")
    if roundabout_qcar:
        print(f"                    Route: 16 → 17 → 16 (infinite loop)")
    print(f"Pedestrians:        {len(pedestrians)}/4 active")
    print(f"Parked Vehicles:    {len(parked_vehicles)}/4 spawned")
    print(f"Stop Signs:         {len(stop_signs)}/1 spawned")
    print("\nPedestrian Crossing Locations:")
    print("  1. South section (Edges 12→7 ↔ 6→13)   - Crossing both parallel roads")
    print("  2. West section (Edges 22→9 ↔ 8→23)    - Crossing both parallel roads")
    print("  3. North section (Edges 14→11 ↔ 10→15) - Crossing both parallel roads")
    print("  4. East section (Edges 18→5 ↔ 4→19)    - Crossing both parallel roads")
    print("\nCoordinate System:")
    print("  - World size: 500m × 500m (±250m from origin)")
    print("  - Origin: [0, 0, 0]")
    print("  - QCar2 facing: +Y axis (north)")
    print("\nStarting teleop control... (Press Q or ESC to exit)")
    print("="*70)

    # =========================================================================
    # START TELEOP CONTROL LOOP
    # =========================================================================
    try:
        teleop_control_loop(qcar, teleop_controller)
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received...")

    # Clean up
    print("\nShutting down...")
    teleop_controller.stop()
    qlabs.close()
    print("✓ Connection closed. Done!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

