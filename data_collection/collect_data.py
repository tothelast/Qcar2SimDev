#!/usr/bin/env python3
"""QLabs data collection with teleop control for expert demonstrations."""

import sys
from pathlib import Path

# Add parent directory to path so we can import core and data_collection modules
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.system import QLabsSystem
from qvl.spline_line import QLabsSplineLine

from data_collection.teleop_controller import TeleopController, teleop_control_loop
from core.scene_loader import SceneLoader
from core.scene_spawner import SceneSpawner
from core.config import SimlingoQCar2Config


def initialize_planned_route_tracer(qlabs, config):
    """Initialize route visualization spline line."""
    print("Initializing route tracer...")
    tracer = QLabsSplineLine(qlabs)

    tracer.spawn_id(
        actorNumber=101,
        location=[0, 0, 0.02],
        rotation=[0, 0, 0],
        configuration=1,
        waitForConfirmation=True
    )

    # Convert waypoints to spline format [x, y, z, width]
    route_points = [[wp[0], wp[1], 0.02, 0.05] for wp in config.route_waypoints]

    if len(route_points) >= 2:
        tracer.set_points(
            color=[0.0, 1.0, 0.0],
            pointList=route_points,
            alignEndPointTangents=False,
            waitForConfirmation=True
        )
        print(f"✓ Route tracer initialized ({len(route_points)} waypoints)")
    else:
        print("WARNING: Not enough waypoints for route visualization")

    return tracer


def main(scene_name=None):
    """Setup QLabs environment and spawn QCar2 for data collection."""
    print("QLabs Data Collection - Cityscape Lite")

    config = SimlingoQCar2Config()
    scene_definition = None

    # Load scene or use default route
    if scene_name:
        print(f"Loading scene: {scene_name}")
        scene_definition = SceneLoader().load_scene(scene_name)
        if not scene_definition:
            print(f"ERROR: Failed to load scene '{scene_name}'")
            return False
        route_name = scene_definition.ego_route
    else:
        route_name = 'simple_straight'
        print(f"Using default route: {route_name}")

    if not config.load_route(route_name):
        print(f"ERROR: Failed to load route '{route_name}'")
        return False

    # Connect to QLabs
    print("Connecting to QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("ERROR: Unable to connect to QLabs")
        return False
    print("✓ Connected to QLabs")

    QLabsSystem(qlabs).set_title_string("QCar2 Data Collection")

    print(f"Cleaning up existing actors...")
    num_destroyed = qlabs.destroy_all_spawned_actors()
    print(f"✓ Destroyed {num_destroyed} actors")

    # Spawn main QCar2 for teleop control
    print("Spawning QCar2 (teleop control)...")
    qcar = QLabsQCar2(qlabs)

    spawn_loc = config.qcar2_spawn_location
    spawn_rot_rad = config.qcar2_spawn_rotation
    spawn_rot_deg = [r * 180.0 / 3.14159265359 for r in spawn_rot_rad]

    status = qcar.spawn_id_degrees(
        actorNumber=0,
        location=spawn_loc,
        rotation=spawn_rot_deg,
        scale=[1.0, 1.0, 1.0],
        configuration=0,
        waitForConfirmation=True
    )

    if status == 0:
        print(f"✓ QCar2 spawned at [{spawn_loc[0]:.1f}, {spawn_loc[1]:.1f}]")
    else:
        print(f"✗ Failed to spawn QCar2 (status: {status})")
        qlabs.close()
        return False

    print("-"*70)

    # =========================================================================
    # INITIALIZE PLANNED ROUTE TRACER (GREEN LINE)
    # =========================================================================
    planned_route_tracer = initialize_planned_route_tracer(qlabs, config)

    # Small delay to ensure ego vehicle is fully initialized before spawning scene actors
    import time
    time.sleep(0.5)

    # =========================================================================
    # SETUP SCENE WITH SCENE SPAWNER
    # =========================================================================
    if scene_definition:
        print("\n" + "="*70)
        print("SETTING UP SCENE ACTORS")
        print("="*70)

        # Create scene spawner with separate QLabs connection
        scene_spawner = SceneSpawner(scene_definition)

        # Connect to QLabs (separate connection for actors)
        if not scene_spawner.connect():
            print("WARNING: Failed to connect scene actors to QLabs")
            scene_spawner = None
        else:
            # Spawn all actors defined in the scene
            if not scene_spawner.spawn_all_actors():
                print("WARNING: Some actors failed to spawn")

            # Start control threads for dynamic actors
            scene_spawner.start_actor_control()
    else:
        print("\nNo scene defined - skipping actor setup")

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

    teleop_controller = TeleopController(config)
    print("✓ Teleop controller initialized")
    print("\nKeyboard Controls:")
    print("  Arrow Up / W:    Accelerate forward")
    print("  Arrow Down / S:  Brake / Reverse")
    print("  Arrow Left / A:  Steer left")
    print("  Arrow Right / D: Steer right")
    print("  B:               Emergency brake (full stop)")
    print("  Q / ESC:         Quit")
    print("-"*70)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SCENE SETUP COMPLETE")
    print("="*70)
    print(f"QCar2 (Main):       Spawned at [{spawn_loc[0]:.3f}, {spawn_loc[1]:.3f}] - TELEOP CONTROL")
    print(f"                    Route: {route_name}")

    if scene_spawner:
        print(f"Autonomous Vehicles: {len(scene_spawner.autonomous_vehicles)}")
        for qcar_obj, vehicle_def in scene_spawner.autonomous_vehicles:
            spawn_loc = vehicle_def.data['spawn_location']
            print(f"  - {vehicle_def.name}: Spawned at [{spawn_loc[0]:.3f}, {spawn_loc[1]:.3f}]")
            if vehicle_def.data.get('route_nodes'):
                route_nodes = vehicle_def.data['route_nodes']
                print(f"    Route: {' → '.join(map(str, route_nodes))} ({vehicle_def.data.get('route_type', 'unknown')})")

        print(f"Pedestrians:        {len(scene_spawner.pedestrians)}")
        for ped_obj, ped_def in scene_spawner.pedestrians:
            curb_1 = ped_def.data['curb_1']
            print(f"  - {ped_def.name}: Spawned at [{curb_1[0]:.3f}, {curb_1[1]:.3f}]")

        print(f"Parked Vehicles:    {len(scene_spawner.parked_vehicles)}")
        print(f"Stop Signs:         {len(scene_spawner.stop_signs)}")
    else:
        print("Scene actors: None (no scene loaded)")

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

    # Cleanup scene actors
    if scene_spawner:
        print("Cleaning up scene actors...")
        scene_spawner.cleanup()

    qlabs.close()
    print("✓ Connection closed. Done!")
    return True


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='QLabs Data Collection with Teleop Control and Scene System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scene System:
  Scenes are predefined configurations that include the ego route and all actors.
  Scenes are organized in:
    - scenes/training/  (7 training scenes)
    - scenes/testing/   (3 testing scenes)

  Use --scene to load a complete scene configuration.
  If no scene is specified, only the ego vehicle will be spawned.

Available Training Scenes:
  01_empty_road              Empty road (simple_straight route)
  02_light_traffic           Light traffic (short_route + 1 circular car)
  03_roundabout_north        Roundabout with north pedestrian
  04_kink_navigation         Kink street with south pedestrian
  05_roundabout_exit_east    Traffic circle with roundabout car + east pedestrian
  06_urban_parking           One-way street with parked vehicle
  07_mixed_traffic           Long route with circular car + east pedestrian

Available Testing Scenes:
  08_full_circuit            Full circuit with 2 cars + east pedestrian
  09_roundabout_exit_dual    Roundabout exit with 2 pedestrians
  10_heavy_traffic           Complex route with 2 cars + pedestrian + stop sign

Examples:
  # Empty scene (just ego vehicle, no actors)
  python data_collection/collect_data.py

  # Load a training scene
  python data_collection/collect_data.py --scene 01_empty_road
  python data_collection/collect_data.py --scene light_traffic

  # Load a testing scene
  python data_collection/collect_data.py --scene full_circuit
        """
    )

    # Scene selection (NEW - replaces individual actor flags)
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene name to load from scenes/ directory (e.g., "empty_road", "01_empty_road", "light_traffic")')

    args = parser.parse_args()

    # Run main with selected scene
    success = main(scene_name=args.scene)
    sys.exit(0 if success else 1)

