"""Scene spawner on separate QLabs connection (prevents ego vehicle interference)."""

import time
import threading
import math
from typing import List, Optional

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.person import QLabsPerson
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.basic_shape import QLabsBasicShape
from hal.products.mats import SDCSRoadMap

from core.scene_loader import SceneDefinition, ActorDefinition


class SceneSpawner:
    """Spawns scene actors on separate QLabs connection."""

    def __init__(self, scene_definition: SceneDefinition):
        """Initialize spawner with scene definition."""
        self.scene = scene_definition
        self.qlabs_actors = None
        self.roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

        # Actor instances
        self.autonomous_vehicles = []
        self.pedestrians = []
        self.parked_vehicles = []
        self.stop_signs = []
        self.crosswalks = []
        self.traffic_lights = []
        self.obstacles = []

        # Control
        self.threads = []
        self.running = False
        self.qlabs_lock = threading.Lock()
        self.traffic_light_threads: List[threading.Thread] = []
        
    def connect(self) -> bool:
        """Connect to QLabs for scene actors."""
        print("\nConnecting scene actors to QLabs...")
        self.qlabs_actors = QuanserInteractiveLabs()

        try:
            self.qlabs_actors.open("localhost")
            print("✓ Scene actors connected")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def spawn_all_actors(self) -> bool:
        """Spawn all scene actors."""
        if not self.qlabs_actors:
            print("ERROR: QLabs not connected. Call connect() first.")
            return False

        print(f"\nSpawning scene: {self.scene.name}")
        success = True

        # Spawn each actor type
        if self.scene.autonomous_vehicles:
            for vdef in self.scene.autonomous_vehicles:
                qcar = self._spawn_autonomous_vehicle(vdef)
                if qcar:
                    self.autonomous_vehicles.append((qcar, vdef))
                else:
                    success = False

        if self.scene.pedestrians:
            for pdef in self.scene.pedestrians:
                ped = self._spawn_pedestrian(pdef)
                if ped:
                    self.pedestrians.append((ped, pdef))
                else:
                    success = False

        if self.scene.parked_vehicles:
            for vdef in self.scene.parked_vehicles:
                veh = self._spawn_parked_vehicle(vdef)
                if veh:
                    self.parked_vehicles.append(veh)
                else:
                    success = False

        if self.scene.stop_signs:
            for sdef in self.scene.stop_signs:
                sign = self._spawn_stop_sign(sdef)
                if sign:
                    self.stop_signs.append(sign)
                else:
                    success = False

        if getattr(self.scene, 'crosswalks', None):
            for cdef in self.scene.crosswalks:
                crosswalk = self._spawn_crosswalk(cdef)
                if crosswalk:
                    self.crosswalks.append((crosswalk, cdef))
                else:
                    success = False

        if getattr(self.scene, 'traffic_lights', None):
            for tdef in self.scene.traffic_lights:
                traffic_light = self._spawn_traffic_light(tdef)
                if traffic_light:
                    self.traffic_lights.append((traffic_light, tdef))
                else:
                    success = False

        if getattr(self.scene, 'obstacles', None):
            for odef in self.scene.obstacles:
                obstacle = self._spawn_obstacle(odef)
                if obstacle:
                    self.obstacles.append((obstacle, odef))
                else:
                    success = False

        print(f"Spawn complete: {len(self.autonomous_vehicles)} vehicles, "
              f"{len(self.pedestrians)} pedestrians, {len(self.parked_vehicles)} parked, "
              f"{len(self.stop_signs)} signs, {len(self.crosswalks)} crosswalks, "
              f"{len(self.traffic_lights)} traffic lights, {len(self.obstacles)} obstacles")
        return success
    
    def _spawn_autonomous_vehicle(self, vehicle_def: ActorDefinition) -> Optional[QLabsQCar2]:
        """Spawn autonomous vehicle."""
        qcar = QLabsQCar2(self.qlabs_actors)
        status = qcar.spawn_id_degrees(
            actorNumber=vehicle_def.data['actor_number'],
            location=vehicle_def.data['spawn_location'],
            rotation=vehicle_def.data['spawn_rotation'],
            scale=[1.0, 1.0, 1.0],
            configuration=0,
            waitForConfirmation=True
        )
        if status == 0:
            print(f"  ✓ {vehicle_def.name}")
            return qcar
        print(f"  ✗ Failed: {vehicle_def.name}")
        return None
    
    def _spawn_pedestrian(self, ped_def: ActorDefinition) -> Optional[QLabsPerson]:
        """Spawn a pedestrian using the shared QLabs connection."""
        pedestrian = QLabsPerson(self.qlabs_actors)

        # Calculate rotation based on crossing direction
        curb_1 = ped_def.data['curb_1']
        curb_2 = ped_def.data['curb_2']

        if ped_def.data['crossing_direction'] == 'horizontal':
            rotation = 0.0 if curb_2[0] > curb_1[0] else 180.0
        else:  # vertical
            rotation = 90.0 if curb_2[1] > curb_1[1] else -90.0

        status = pedestrian.spawn_id_degrees(
            actorNumber=ped_def.actor_number,
            location=curb_1,
            rotation=[0.0, 0.0, rotation],
            scale=[1.0, 1.0, 1.0],
            configuration=(ped_def.actor_number % 12),
            waitForConfirmation=True
        )

        if status == 0:
            print(f"  ✓ {ped_def.name} spawned (actor {ped_def.actor_number})")
            return pedestrian
        else:
            print(f"  ✗ Failed to spawn {ped_def.name} (actor {ped_def.actor_number})")
            return None

    def _spawn_parked_vehicle(self, vehicle_def: ActorDefinition) -> Optional[QLabsQCar2]:
        """Spawn a parked vehicle."""
        qcar = QLabsQCar2(self.qlabs_actors)

        status = qcar.spawn_id_degrees(
            actorNumber=vehicle_def.actor_number,
            location=vehicle_def.data['location'],
            rotation=vehicle_def.data['rotation'],
            scale=[1.0, 1.0, 1.0],
            configuration=vehicle_def.data.get('configuration', 0),
            waitForConfirmation=True
        )

        if status == 0:
            print(f"  ✓ {vehicle_def.name} spawned (actor {vehicle_def.actor_number})")
            return qcar
        else:
            print(f"  ✗ Failed to spawn {vehicle_def.name} (actor {vehicle_def.actor_number})")
            return None

    def _spawn_stop_sign(self, sign_def: ActorDefinition) -> Optional[QLabsStopSign]:
        """Spawn a stop sign."""
        stop_sign = QLabsStopSign(self.qlabs_actors)

        status = stop_sign.spawn_id_degrees(
            actorNumber=sign_def.actor_number,
            location=sign_def.data['location'],
            rotation=sign_def.data['rotation'],
            scale=[1.0, 1.0, 1.0],
            configuration=0,
            waitForConfirmation=True
        )

        if status == 0:
            print(f"  ✓ {sign_def.name} spawned (actor {sign_def.actor_number})")
            return stop_sign
        else:
            print(f"  ✗ Failed to spawn {sign_def.name} (actor {sign_def.actor_number})")
            return None

    def _spawn_crosswalk(self, crosswalk_def: ActorDefinition) -> Optional[QLabsCrosswalk]:
        """Spawn a crosswalk marker."""
        crosswalk = QLabsCrosswalk(self.qlabs_actors)

        location = crosswalk_def.data.get('location', [0.0, 0.0, 0.0])
        rotation = crosswalk_def.data.get('rotation', [0.0, 0.0, 0.0])
        scale = crosswalk_def.data.get('scale', [1.0, 1.0, 1.0])
        configuration = crosswalk_def.data.get('configuration', 0)

        status = crosswalk.spawn_id_degrees(
            actorNumber=crosswalk_def.actor_number,
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration,
            waitForConfirmation=True
        )

        if status == 0:
            print(f"  ✓ {crosswalk_def.name} spawned (actor {crosswalk_def.actor_number})")
            return crosswalk

        print(f"  ✗ Failed to spawn {crosswalk_def.name} (actor {crosswalk_def.actor_number})")
        return None

    def _spawn_traffic_light(self, light_def: ActorDefinition) -> Optional[QLabsTrafficLight]:
        """Spawn a traffic light and set its initial color."""
        traffic_light = QLabsTrafficLight(self.qlabs_actors)

        location = light_def.data.get('location', [0.0, 0.0, 0.0])
        rotation = light_def.data.get('rotation', [0.0, 0.0, 0.0])
        scale = light_def.data.get('scale', [1.0, 1.0, 1.0])
        configuration = light_def.data.get('configuration', 0)

        status = traffic_light.spawn_id_degrees(
            actorNumber=light_def.actor_number,
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration,
            waitForConfirmation=True
        )

        if status != 0:
            print(f"  ✗ Failed to spawn {light_def.name} (actor {light_def.actor_number})")
            return None

        color_index = light_def.data.get('color_index')
        if color_index is not None:
            traffic_light.set_color(color_index, waitForConfirmation=True)

        print(f"  ✓ {light_def.name} spawned (actor {light_def.actor_number})")
        return traffic_light

    def _traffic_light_cycle_loop(self, traffic_light: QLabsTrafficLight, cfg: dict, name: str):
        """Cycle a traffic light through configured red/green/yellow durations."""
        # Extract durations (seconds)
        red_dur = max(0.0, float(cfg.get('red', 30.0)))
        yellow_dur = max(0.0, float(cfg.get('yellow', 3.0)))
        green_dur = max(0.0, float(cfg.get('green', 30.0)))
        start_color = int(cfg.get('start', 1))

        base_sequence = [1, 3, 2]  # Red -> Green -> Yellow
        durations = {1: red_dur, 3: green_dur, 2: yellow_dur}

        sequence = [color for color in base_sequence if durations[color] > 0.0]
        if not sequence:
            sequence = [1]

        if start_color in sequence:
            while sequence[0] != start_color:
                sequence.append(sequence.pop(0))

        # Minor delay to stagger API calls
        time.sleep(0.25)

        idx = 0
        while self.running:
            color = sequence[idx % len(sequence)]
            try:
                with self.qlabs_lock:
                    traffic_light.set_color(color, waitForConfirmation=True)
            except Exception as exc:
                print(f"  ✗ Traffic light cycle error for {name}: {exc}")
                break

            duration = max(0.05, durations.get(color, 0.05))
            elapsed = 0.0
            while self.running and elapsed < duration:
                sleep_step = min(0.5, duration - elapsed)
                time.sleep(sleep_step)
                elapsed += sleep_step

            idx += 1

    def _spawn_obstacle(self, obstacle_def: ActorDefinition) -> Optional[QLabsBasicShape]:
        """Spawn a static obstacle (e.g., cone) using the basic shape actor."""
        obstacle = QLabsBasicShape(self.qlabs_actors)

        location = obstacle_def.data.get('location', [0.0, 0.0, 0.0])
        rotation = obstacle_def.data.get('rotation', [0.0, 0.0, 0.0])
        scale = obstacle_def.data.get('scale', [1.0, 1.0, 1.0])
        configuration = obstacle_def.data.get('configuration', QLabsBasicShape.SHAPE_CONE)

        status = obstacle.spawn_id_degrees(
            actorNumber=obstacle_def.actor_number,
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration,
            waitForConfirmation=True
        )

        if status != 0:
            print(f"  ✗ Failed to spawn {obstacle_def.name} (actor {obstacle_def.actor_number})")
            return None

        color = obstacle_def.data.get('color')
        if color:
            roughness = obstacle_def.data.get('roughness', 0.5)
            metallic = obstacle_def.data.get('metallic', False)
            obstacle.set_material_properties(color=color, roughness=roughness, metallic=metallic, waitForConfirmation=True)

        enable_dynamics = obstacle_def.data.get('enable_dynamics', False)
        obstacle.set_enable_dynamics(enable_dynamics, waitForConfirmation=True)

        enable_collisions = obstacle_def.data.get('enable_collisions', True)
        obstacle.set_enable_collisions(enable_collisions, waitForConfirmation=True)

        print(f"  ✓ {obstacle_def.name} spawned (actor {obstacle_def.actor_number})")
        return obstacle

    def start_actor_control(self):
        """Start control threads for autonomous vehicles and pedestrians."""
        if not self.autonomous_vehicles and not self.pedestrians:
            print("\nNo dynamic actors to control")
            return

        print("\n" + "="*80)
        print("STARTING ACTOR CONTROL THREADS")
        print("="*80)

        self.running = True

        # Start autonomous vehicle threads
        speed_limit = self.scene.traffic_speed_limit if getattr(self.scene, 'traffic_speed_limit', None) is not None else 1.5

        for qcar, vehicle_def in self.autonomous_vehicles:
            route_nodes = vehicle_def.data.get('route_nodes', [])
            target_speed = vehicle_def.data.get('control_params', {}).get('target_speed', 2.5)
            target_speed = min(target_speed, speed_limit)
            # Reduce default update rate to 2Hz to prevent QLabs lock contention with main thread
            update_rate_hz = vehicle_def.data.get('control_params', {}).get('update_rate_hz', 2)

            if vehicle_def.data.get('route_type') in ['circular', 'roundabout']:
                route_type = vehicle_def.data.get('route_type')
                thread = threading.Thread(
                    target=self._vehicle_control_loop,
                    args=(qcar, vehicle_def.name, route_nodes, target_speed, update_rate_hz, route_type),
                    daemon=True,
                    name=f"{route_type}_{vehicle_def.name}"
                )
            else:
                print(f"  ✗ Unknown route type: {vehicle_def.data.get('route_type')}")
                continue

            thread.start()
            self.threads.append(thread)
            print(f"  ✓ Started {vehicle_def.name} control thread")

        # Start pedestrian threads
        for pedestrian, ped_def in self.pedestrians:
            thread = threading.Thread(
                target=self._pedestrian_crossing_loop,
                args=(pedestrian, ped_def),
                daemon=True,
                name=f"pedestrian_{ped_def.name}"
            )
            thread.start()
            self.threads.append(thread)
            print(f"  ✓ Started {ped_def.name} control thread")

        # Start traffic light cycle threads (if configured)
        for traffic_light, tdef in self.traffic_lights:
            cycle_cfg = tdef.data.get('cycle')
            if isinstance(cycle_cfg, dict):
                thread = threading.Thread(
                    target=self._traffic_light_cycle_loop,
                    args=(traffic_light, cycle_cfg, tdef.name),
                    daemon=True,
                    name=f"traffic_light_{tdef.name}"
                )
                thread.start()
                self.traffic_light_threads.append(thread)
                self.threads.append(thread)
                print(f"  ✓ Started {tdef.name} traffic light cycle thread")

        print(f"\nTotal control threads started: {len(self.threads)}")
        print("="*80)

    def _vehicle_control_loop(self, qcar: QLabsQCar2, actor_name: str, route_nodes: List[int], target_speed: float, update_rate_hz: int, route_type: str):
        """
        Control loop for vehicle using Stanley controller.

        Args:
            qcar: QLabsQCar2 instance
            actor_name: Name of the actor
            route_nodes: List of node IDs forming the route
            target_speed: Target speed in m/s
            update_rate_hz: Control loop frequency in Hz
            route_type: Type of route (e.g., 'circular', 'roundabout')
        """
        import numpy as np

        thread_id = threading.current_thread().name
        print(f"  [{route_type.upper()}-{thread_id}] Starting with route_nodes={route_nodes}, speed={target_speed}, rate={update_rate_hz}Hz", flush=True)

        # Generate full route waypoints
        all_waypoints = []
        for i in range(len(route_nodes)):
            from_node = route_nodes[i]
            to_node = route_nodes[(i + 1) % len(route_nodes)]  # Wrap around for circular route

            # Generate path for this edge
            path = self.roadmap.generate_path([from_node, to_node])
            x_coords = path[0, :] * 10.0  # Scale to QLabs coordinates
            y_coords = path[1, :] * 10.0

            # Add waypoints (skip first point if not the first edge to avoid duplicates)
            start_idx = 1 if i > 0 else 0
            for j in range(start_idx, len(x_coords)):
                all_waypoints.append([x_coords[j], y_coords[j]])

        all_waypoints = np.array(all_waypoints)
        print(f"  [{route_type.upper()}-{thread_id}] Generated {len(all_waypoints)} waypoints", flush=True)

        # Control parameters
        k_stanley = 0.3  # Stanley controller gain
        max_steering_angle = math.pi / 6  # 30 degrees max
        lookahead_distance = 3.0  # meters
        current_waypoint_idx = 0
        update_interval = 1.0 / update_rate_hz

        # State variables
        current_pos = None
        steering_angle = 0.0

        # Wait before starting (stagger by actor number to avoid simultaneous API calls)
        actor_offset = hash(actor_name) % 100 / 1000.0  # 0-99ms offset based on actor name
        time.sleep(2.0 + actor_offset)
        print(f"  [{route_type.upper()}-{thread_id}] Starting movement control loop (offset: {actor_offset*1000:.0f}ms)...", flush=True)

        iteration = 0
        while self.running:
            iteration += 1
            loop_start_time = time.time()

            try:
                # Query state and send control (with lock to prevent simultaneous API calls)
                with self.qlabs_lock:
                    success, location, rotation, _, _ = qcar.set_velocity_and_request_state(
                        forward=target_speed if current_pos is not None else 0.0,
                        turn=steering_angle,
                        headlights=True,
                        leftTurnSignal=False,
                        rightTurnSignal=False,
                        brakeSignal=False,
                        reverseSignal=False
                    )

                if not success:
                    time.sleep(update_interval)
                    continue

                current_pos = np.array([location[0], location[1]])
                current_yaw = rotation[2]

                # Find nearest waypoint on path
                distances = np.linalg.norm(all_waypoints - current_pos, axis=1)
                nearest_idx = np.argmin(distances)
                current_waypoint_idx = nearest_idx

                # Find lookahead waypoint
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

                # Get target waypoint
                target_wp = all_waypoints[lookahead_idx]

                # Calculate cross-track error
                path_vector = target_wp - current_pos
                path_distance = np.linalg.norm(path_vector)

                if path_distance < 0.01:  # Too close, skip
                    time.sleep(update_interval)
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

                # Cross-track error
                cross_track_error = path_distance * math.sin(heading_error)

                # Stanley controller
                if target_speed > 0.01:
                    steering_correction = math.atan(k_stanley * cross_track_error / target_speed)
                else:
                    steering_correction = 0.0

                steering_angle = heading_error + steering_correction

                # NEGATE steering angle - flip left/right convention
                steering_angle = -steering_angle

                # Clip steering angle
                steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

                # Debug output (first 10 iterations)
                if iteration <= 10:
                    print(f"  [{route_type.upper()}-{thread_id}] iter={iteration}, pos=[{current_pos[0]:.2f},{current_pos[1]:.2f}], wp={current_waypoint_idx}/{lookahead_idx}, heading_err={heading_error:.3f}, steering={steering_angle:.3f}", flush=True)

                # Sleep for remaining time to maintain update rate
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"  [{route_type.upper()}-{thread_id}] ERROR at iteration {iteration}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

        print(f"  [{route_type.upper()}-{thread_id}] Loop exited after {iteration} iterations", flush=True)

    def _pedestrian_crossing_loop(self, pedestrian: QLabsPerson, ped_def: ActorDefinition):
        """Control loop for pedestrian crossing."""
        curb_1 = ped_def.data['curb_1']
        curb_2 = ped_def.data['curb_2']
        movement_params = ped_def.data.get('movement_params', {})
        walk_speed = movement_params.get('walk_speed', 1.2)
        pace_time = movement_params.get('pace_time', 2.5)
        wait_time = movement_params.get('wait_time', 2.0)
        crossing_time = movement_params.get('crossing_time', 9.0)

        at_curb_1 = True

        while self.running:
            try:
                # Pace on curb
                time.sleep(pace_time)

                # Wait before crossing
                time.sleep(wait_time)

                # Cross to other curb (with lock to prevent simultaneous API calls)
                target = curb_2 if at_curb_1 else curb_1
                with self.qlabs_lock:
                    pedestrian.move_to(target, walk_speed)

                # Wait for crossing to complete
                time.sleep(crossing_time)

                # Toggle position
                at_curb_1 = not at_curb_1

            except Exception as e:
                print(f"Error in pedestrian crossing loop: {e}")
                break

    def stop_actor_control(self):
        """Stop all actor control threads."""
        if not self.running:
            return

        print("\nStopping actor control threads...")
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)

        print("✓ All actor control threads stopped")

    def cleanup(self):
        """Cleanup resources and close QLabs connection."""
        self.stop_actor_control()

        if self.qlabs_actors:
            print("Closing scene actors QLabs connection...")
            self.qlabs_actors.close()
            self.qlabs_actors = None
            print("✓ Scene actors QLabs connection closed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
