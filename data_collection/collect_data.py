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
- B: Emergency brake (full stop)
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
import argparse
from pathlib import Path

# Add python directory to path for QLabs SDK
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))
# Add parent directory to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.system import QLabsSystem
from qvl.spline_line import QLabsSplineLine

from teleop_controller import TeleopController, teleop_control_loop
from core.scene_loader import SceneLoader
from core.scene_spawner import SceneSpawner
from core.config import SimlingoQCar2Config


def initialize_planned_route_tracer(qlabs, config):
    """
    Initialize QLabs planned route tracer (green spline line) to visualize the route.

    Args:
        qlabs: QuanserInteractiveLabs instance
        config: SimlingoQCar2Config instance

    Returns:
        QLabsSplineLine instance
    """
    print("\nInitializing planned route tracer...")
    planned_route_tracer = QLabsSplineLine(qlabs)

    # Spawn the spline line actor at origin
    # Configuration 1 = CURVE mode for smooth route visualization
    planned_route_tracer.spawn_id(
        actorNumber=101,  # Use actor number 101 for planned route
        location=[0, 0, 0.02],  # Slightly above ground
        rotation=[0, 0, 0],
        configuration=1,  # CURVE configuration
        waitForConfirmation=True
    )

    # Convert route waypoints to spline line format
    # Format: [x, y, z, width]
    route_points = []
    for waypoint in config.route_waypoints:
        route_points.append([
            waypoint[0],  # x
            waypoint[1],  # y
            0.02,  # z (slightly above ground)
            0.05  # width (use planned_route_tracer_width if available, else default)
        ])

    # Draw the planned route
    if len(route_points) >= 2:
        # Use green color for the route [R, G, B]
        planned_route_tracer.set_points(
            color=[0.0, 1.0, 0.0],  # Green
            pointList=route_points,
            alignEndPointTangents=False,
            waitForConfirmation=True
        )
        print(f"✓ Planned route tracer initialized (green line shows {len(route_points)} waypoints)")
    else:
        print("WARNING: Not enough route waypoints to display planned route")

    return planned_route_tracer


def main(scene_name=None):
    """
    Main function to setup QLabs environment and spawn QCar2 with obstacles.

    Args:
        scene_name: Name of the scene to load from scenes/ directory (optional)
    """
    # Load configuration and route
    print("="*70)
    print("QLabs Data Collection - Cityscape Lite")
    print("="*70)

    config = SimlingoQCar2Config()

    # Load scene if specified
    scene_definition = None
    scene_spawner = None

    if scene_name:
        print(f"\nLoading scene: {scene_name}")
        scene_loader = SceneLoader()
        scene_definition = scene_loader.load_scene(scene_name)

        if not scene_definition:
            print(f"\nERROR: Failed to load scene '{scene_name}'")
            return False

        print(f"Scene loaded: {scene_definition}")

        # Load route from scene definition
        route_name = scene_definition.ego_route
        print(f"Loading route from scene: {route_name}")
    else:
        # Default route if no scene specified
        route_name = 'simple_straight'
        print(f"\nNo scene specified, using default route: {route_name}")

    # Load route from JSON file
    if not config.load_route(route_name):
        print(f"\nERROR: Failed to load route '{route_name}'")
        print("Available routes are in the routes/ directory")
        return False

    # Create connection to QLabs
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
    # Use spawn location and rotation from loaded route

    print("\n" + "-"*70)
    print("Spawning Main QCar2 (Teleop Control)...")
    print("-"*70)
    qcar = QLabsQCar2(qlabs)

    # Get spawn location and rotation from loaded route
    spawn_location = config.qcar2_spawn_location
    spawn_rotation_rad = config.qcar2_spawn_rotation

    # Convert rotation from radians to degrees for spawn_id_degrees
    spawn_rotation_deg = [
        spawn_rotation_rad[0] * 180.0 / 3.14159265359,  # roll
        spawn_rotation_rad[1] * 180.0 / 3.14159265359,  # pitch
        spawn_rotation_rad[2] * 180.0 / 3.14159265359   # yaw
    ]

    status = qcar.spawn_id_degrees(
        actorNumber=0,
        location=spawn_location,
        rotation=spawn_rotation_deg,
        scale=[1.0, 1.0, 1.0],
        configuration=0,
        waitForConfirmation=True
    )

    if status == 0:
        print(f"✓ Main QCar2 spawned at [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}, {spawn_location[2]:.3f}]")
        print(f"  Rotation: {spawn_rotation_deg[2]:.1f}° (yaw)")
        print(f"  Route: {route_name}")
        print(f"  Mode: TELEOP CONTROL")
    else:
        print(f"✗ ERROR: Failed to spawn main QCar2. Status code: {status}")
        print("  Status codes: 0=success, 1=class not available, 2=actor in use, 3=unknown")
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

    teleop_controller = TeleopController()
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
    print(f"QCar2 (Main):       Spawned at [{spawn_location[0]:.3f}, {spawn_location[1]:.3f}] - TELEOP CONTROL")
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

