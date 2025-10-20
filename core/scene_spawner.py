#!/usr/bin/env python3
"""
Scene Spawner Module

Spawns scene actors on a SEPARATE QLabs connection from the ego vehicle.
This prevents buffer corruption and maintains the 20 Hz control loop.

Key Design:
- Ego vehicle uses Connection 1 (main QLabs instance)
- Scene actors use Connection 2 (separate QLabs instance)
- No lock needed - separate communication buffers
"""

import sys
import time
import threading
import math
from pathlib import Path
from typing import List, Optional

# Add python directory to path for QLabs SDK
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.person import QLabsPerson
from qvl.stop_sign import QLabsStopSign
from hal.products.mats import SDCSRoadMap

from core.scene_loader import SceneDefinition, ActorDefinition


class SceneSpawner:
    """
    Spawns and manages scene actors on a separate QLabs connection.
    
    This class creates a NEW QLabs connection specifically for scene actors,
    preventing interference with the ego vehicle's control loop.
    """
    
    def __init__(self, scene_definition: SceneDefinition):
        """
        Initialize scene spawner.
        
        Args:
            scene_definition: SceneDefinition object with actor configurations
        """
        self.scene = scene_definition
        self.qlabs_actors = None  # Shared connection for spawning
        self.roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

        # Actor instances
        self.autonomous_vehicles = []  # List of (QLabsQCar2, ActorDefinition) tuples
        self.pedestrians = []  # List of (QLabsPerson, ActorDefinition) tuples
        self.parked_vehicles = []  # List of QLabsQCar2 instances
        self.stop_signs = []  # List of QLabsStopSign instances

        # Control threads
        self.threads = []
        self.running = False

        # Shared lock for QLabs API calls (all actors use same connection)
        self.qlabs_lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        Create a separate QLabs connection for scene actors.
        
        Returns:
            True if connection successful, False otherwise
        """
        print("\n" + "="*80)
        print("CONNECTING SCENE ACTORS TO QLABS (Separate Connection)")
        print("="*80)
        
        self.qlabs_actors = QuanserInteractiveLabs()
        
        try:
            print("Connecting to QLabs for scene actors...")
            # Connect to the same QLabs server but as a separate client
            # QLabs supports multiple simultaneous connections
            self.qlabs_actors.open("localhost")
            print("✓ Scene actors connected to QLabs successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to connect scene actors to QLabs: {e}")
            return False
    
    def spawn_all_actors(self) -> bool:
        """
        Spawn all actors defined in the scene.
        
        Returns:
            True if all actors spawned successfully, False otherwise
        """
        if not self.qlabs_actors:
            print("ERROR: QLabs connection not established. Call connect() first.")
            return False
        
        print("\n" + "="*80)
        print(f"SPAWNING SCENE: {self.scene.name}")
        print(f"Description: {self.scene.description}")
        print("="*80)
        
        success = True
        
        # Spawn autonomous vehicles
        if self.scene.autonomous_vehicles:
            print(f"\nSpawning {len(self.scene.autonomous_vehicles)} autonomous vehicle(s)...")
            for vehicle_def in self.scene.autonomous_vehicles:
                qcar = self._spawn_autonomous_vehicle(vehicle_def)
                if qcar:
                    self.autonomous_vehicles.append((qcar, vehicle_def))
                else:
                    success = False

        # Spawn pedestrians
        if self.scene.pedestrians:
            print(f"\nSpawning {len(self.scene.pedestrians)} pedestrian(s)...")
            for ped_def in self.scene.pedestrians:
                pedestrian = self._spawn_pedestrian(ped_def)
                if pedestrian:
                    self.pedestrians.append((pedestrian, ped_def))
                else:
                    success = False
        
        # Spawn parked vehicles
        if self.scene.parked_vehicles:
            print(f"\nSpawning {len(self.scene.parked_vehicles)} parked vehicle(s)...")
            for vehicle_def in self.scene.parked_vehicles:
                vehicle = self._spawn_parked_vehicle(vehicle_def)
                if vehicle:
                    self.parked_vehicles.append(vehicle)
                else:
                    success = False
        
        # Spawn stop signs
        if self.scene.stop_signs:
            print(f"\nSpawning {len(self.scene.stop_signs)} stop sign(s)...")
            for sign_def in self.scene.stop_signs:
                sign = self._spawn_stop_sign(sign_def)
                if sign:
                    self.stop_signs.append(sign)
                else:
                    success = False
        
        print("\n" + "="*80)
        print(f"SCENE SPAWN COMPLETE: {self.scene.name}")
        print(f"  Autonomous vehicles: {len(self.autonomous_vehicles)}")
        print(f"  Pedestrians: {len(self.pedestrians)}")
        print(f"  Parked vehicles: {len(self.parked_vehicles)}")
        print(f"  Stop signs: {len(self.stop_signs)}")
        print("="*80)
        
        return success
    
    def _spawn_autonomous_vehicle(self, vehicle_def: ActorDefinition) -> Optional[QLabsQCar2]:
        """Spawn an autonomous vehicle using the shared QLabs connection."""
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
            print(f"  ✓ {vehicle_def.name} spawned (actor {vehicle_def.actor_number})")
            return qcar
        else:
            print(f"  ✗ Failed to spawn {vehicle_def.name} (actor {vehicle_def.actor_number})")
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
        for qcar, vehicle_def in self.autonomous_vehicles:
            route_nodes = vehicle_def.data.get('route_nodes', [])
            target_speed = vehicle_def.data.get('control_params', {}).get('target_speed', 2.5)
            update_rate_hz = vehicle_def.data.get('control_params', {}).get('update_rate_hz', 10)

            if vehicle_def.data.get('route_type') == 'circular':
                thread = threading.Thread(
                    target=self._circular_vehicle_loop,
                    args=(qcar, vehicle_def.name, route_nodes, target_speed, update_rate_hz),
                    daemon=True,
                    name=f"circular_{vehicle_def.name}"
                )
            elif vehicle_def.data.get('route_type') == 'roundabout':
                thread = threading.Thread(
                    target=self._roundabout_vehicle_loop,
                    args=(qcar, vehicle_def.name, route_nodes, target_speed, update_rate_hz),
                    daemon=True,
                    name=f"roundabout_{vehicle_def.name}"
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

        print(f"\nTotal control threads started: {len(self.threads)}")
        print("="*80)

    def _circular_vehicle_loop(self, qcar: QLabsQCar2, actor_name: str, route_nodes: List[int], target_speed: float, update_rate_hz: int):
        """
        Control loop for circular route vehicle using Stanley controller.

        Args:
            qcar: QLabsQCar2 instance
            route_nodes: List of node IDs forming the route
            target_speed: Target speed in m/s
            update_rate_hz: Control loop frequency in Hz
        """
        import numpy as np

        thread_id = threading.current_thread().name
        print(f"  [CIRCULAR-{thread_id}] Starting with route_nodes={route_nodes}, speed={target_speed}, rate={update_rate_hz}Hz", flush=True)

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
        print(f"  [CIRCULAR-{thread_id}] Generated {len(all_waypoints)} waypoints", flush=True)

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
        print(f"  [CIRCULAR-{thread_id}] Starting movement control loop (offset: {actor_offset*1000:.0f}ms)...", flush=True)

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
                    print(f"  [CIRCULAR-{thread_id}] iter={iteration}, pos=[{current_pos[0]:.2f},{current_pos[1]:.2f}], wp={current_waypoint_idx}/{lookahead_idx}, target=[{target_wp[0]:.2f},{target_wp[1]:.2f}], heading_err={heading_error:.3f}, steering={steering_angle:.3f}", flush=True)

                # Sleep for remaining time to maintain update rate
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"  [CIRCULAR-{thread_id}] ERROR at iteration {iteration}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

        print(f"  [CIRCULAR-{thread_id}] Loop exited after {iteration} iterations", flush=True)

    def _roundabout_vehicle_loop(self, qcar: QLabsQCar2, actor_name: str, route_nodes: List[int], target_speed: float, update_rate_hz: int):
        """
        Control loop for roundabout route vehicle using Stanley controller.

        Args:
            qcar: QLabsQCar2 instance
            route_nodes: List of node IDs forming the route
            target_speed: Target speed in m/s
            update_rate_hz: Control loop frequency in Hz
        """
        import numpy as np

        thread_id = threading.current_thread().name
        print(f"  [ROUNDABOUT-{thread_id}] Starting with route_nodes={route_nodes}, speed={target_speed}, rate={update_rate_hz}Hz", flush=True)

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
        print(f"  [ROUNDABOUT-{thread_id}] Generated {len(all_waypoints)} waypoints", flush=True)

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
        print(f"  [ROUNDABOUT-{thread_id}] Starting movement control loop (offset: {actor_offset*1000:.0f}ms)...", flush=True)

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
                    print(f"  [ROUNDABOUT-{thread_id}] iter={iteration}, pos=[{current_pos[0]:.2f},{current_pos[1]:.2f}], wp={current_waypoint_idx}/{lookahead_idx}, heading_err={heading_error:.3f}, steering={steering_angle:.3f}", flush=True)

                # Sleep for remaining time to maintain update rate
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"  [ROUNDABOUT-{thread_id}] ERROR at iteration {iteration}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break

        print(f"  [ROUNDABOUT-{thread_id}] Loop exited after {iteration} iterations", flush=True)

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
        print("\nStopping actor control threads...")
        self.running = False

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)

        print("✓ All actor control threads stopped")

    def cleanup(self):
        """Cleanup resources and close QLabs connection."""
        self.stop_actor_control()

        if self.qlabs_actors:
            print("Closing scene actors QLabs connection...")
            self.qlabs_actors.close()
            print("✓ Scene actors QLabs connection closed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

