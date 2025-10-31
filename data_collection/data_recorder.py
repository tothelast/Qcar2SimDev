#!/usr/bin/env python3
"""Data recorder for QLabs expert demonstrations."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class RecorderPaths:
    """Paths for the current recording run."""

    town_dir: Path
    rgb_dir: Path
    measurements_dir: Path
    results_path: Path


class DataRecorder:
    """Records expert driving demonstrations for SimLingo fine-tuning."""

    def __init__(
        self,
        config,
        database_root: Optional[Path] = None,
        dataset_name: str = "qlabs",
    ) -> None:
        self.config = config
        self.database_root = Path(database_root or (Path(__file__).parent.parent / "database"))
        self.dataset_name = dataset_name
        self.frame_interval = max(1, int(getattr(config, "data_save_freq", 5)))
        self.camera_id = getattr(config, "qcar2_camera", 3)
        self.target_lookahead = float(getattr(config, "target_point_lookahead", 7.5))
        self.secondary_lookahead = self.target_lookahead * 2.0  # legacy; not used in "next waypoint" mode

        self.paths: Optional[RecorderPaths] = None
        self.qcar = None
        self.route_xy: Optional[np.ndarray] = None
        self.route_original: Optional[list[list[float]]] = None
        self.num_route_points = int(getattr(config, "num_route_points", 20))
        self.frame_idx = 0
        self.prev_sample: Optional[Tuple[np.ndarray, float]] = None
        self.is_recording = False
        self.run_metadata = {}
        self._camera_failures = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start_run(
        self,
        qcar,
        route_name: str,
        scene_name: Optional[str],
        split: str = "train",
    ) -> None:
        """Initialise a new recording run."""
        if not getattr(self.config, "route_waypoints", None):
            raise RuntimeError("SimlingoQCar2Config has no route_waypoints. Call load_route() first.")

        split_folder = self._resolve_split(split)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_id = f"Rep_{route_name}_{timestamp}"
        run_root = (
            self.database_root
            / "data"
            / "simlingo"
            / split_folder
            / self.dataset_name
        )
        run_dir = run_root / base_run_id
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_dir = run_root / f"{base_run_id}_{suffix:02d}"

        town_dir = run_dir / "TownQLabs"
        rgb_dir = town_dir / "rgb"
        measurements_dir = town_dir / "measurements"

        measurements_dir.mkdir(parents=True, exist_ok=False)
        rgb_dir.mkdir(parents=True, exist_ok=False)

        self.paths = RecorderPaths(
            town_dir=town_dir,
            rgb_dir=rgb_dir,
            measurements_dir=measurements_dir,
            results_path=town_dir / "results.json.gz",
        )
        self.qcar = qcar

        route_wp = np.asarray(self.config.route_waypoints, dtype=np.float32)
        self.route_xy = route_wp[:, :2]
        self.route_original = [[float(x), float(y)] for x, y in self.route_xy]

        self.frame_idx = 0
        self.prev_sample = None
        self.is_recording = True
        self.run_metadata = {
            "route_name": route_name,
            "scene_name": scene_name,
            "split": split_folder,
            "start_time": timestamp,
        }
        self._camera_failures = 0

        print(f"Recording data to {town_dir}")

    def record_step(
        self,
        iteration: int,
        timestamp: float,
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
    ) -> None:
        """Record a single timestep if it aligns with the capture interval."""
        if not self.is_recording or self.paths is None:
            return

        if iteration % self.frame_interval != 0:
            return

        location_np = np.asarray(location, dtype=np.float64)
        rotation_np = np.asarray(rotation, dtype=np.float64)

        image_bgr = self._capture_image()
        if image_bgr is None:
            return

        ego_matrix = self._build_ego_matrix(location_np, rotation_np)
        target_point, next_target = self._compute_target_points_next_wp(location_np, ego_matrix)
        route_local, route_local_original = self._compute_route_segment(location_np, ego_matrix)
        speed = self._estimate_speed(location_np, timestamp)

        measurement = {
            "ego_matrix": ego_matrix.tolist(),
            "route_original": route_local_original,
            "route": route_local,
            "target_point": target_point.tolist(),
            "target_point_next": next_target.tolist(),
            "speed": float(speed),
            "augmentation_rotation": 0.0,
            "augmentation_translation": 0.0,
            "command": 4,
            "next_command": 4,
        }

        self._write_frame(image_bgr, measurement)
        self.prev_sample = (location_np, timestamp)

    def _compute_route_segment(
        self,
        location: np.ndarray,
        ego_matrix: np.ndarray,
    ) -> Tuple[list[list[float]], list[list[float]]]:
        """Compute remaining route in ego frame for current timestep."""

        if self.route_xy is None:
            raise RuntimeError("Route waypoints not initialised. Call start_run() first.")

        # Determine the closest waypoint to current position
        distances = np.linalg.norm(self.route_xy - location[:2], axis=1)
        nearest_idx = int(np.argmin(distances))

        # Select remaining route waypoints ahead of the vehicle
        remaining_world = self.route_xy[nearest_idx:]
        if remaining_world.shape[0] == 0:
            return [], []

        # Limit to configured number of waypoints for storage/padding
        max_points = max(self.num_route_points, 1)
        if remaining_world.shape[0] < max_points:
            pad = np.repeat(remaining_world[-1][np.newaxis, :], max_points - remaining_world.shape[0], axis=0)
            remaining_world = np.vstack([remaining_world, pad])
        else:
            remaining_world = remaining_world[:max_points]

        ego_inv = np.linalg.inv(ego_matrix)

        def to_ego(point_xy: np.ndarray) -> list[float]:
            return self._world_to_ego(point_xy, ego_inv).tolist()

        route_local = [to_ego(pt) for pt in remaining_world]

        # For QLabs recordings we do not distinguish between adjusted/original routes
        return route_local, route_local

    def finalize(self, success: bool = True) -> None:
        """Finalize the current run and write results metadata."""
        if not self.is_recording or self.paths is None:
            return

        self.is_recording = False
        scores = 100.0 if success and self.frame_idx > 0 else 0.0
        results = {
            "scores": {
                "score_composed": scores,
                "score_route": scores,
            },
            "num_infractions": 0 if scores >= 100.0 else 1,
            "infractions": {
                "min_speed_infractions": [] if scores >= 100.0 else ["insufficient_data"],
                "outside_route_lanes": [],
            },
            "meta": {
                "frames_recorded": self.frame_idx,
                **self.run_metadata,
            },
        }

        with gzip.open(self.paths.results_path, "wt", encoding="utf-8") as fh:
            json.dump(results, fh, separators=(",", ":"))

        print(
            f"Saved {self.frame_idx} frames to {self.paths.town_dir} "
            f"(success={success and self.frame_idx > 0})"
        )

        self.paths = None
        self.qcar = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_split(self, split: str) -> str:
        mapping = {
            "train": "routes_training",
            "training": "routes_training",
            "val": "routes_validation",
            "validation": "routes_validation",
        }
        if split not in mapping:
            raise ValueError(f"Unsupported split '{split}'. Use 'train' or 'val'.")
        return mapping[split]

    def _capture_image(self) -> Optional[np.ndarray]:
        if self.qcar is None:
            return None
        status, image_bgr = self.qcar.get_image(self.camera_id)
        if not status or image_bgr is None:
            self._camera_failures += 1
            if self._camera_failures <= 5 or self._camera_failures % 25 == 0:
                print("WARNING: Failed to capture image from QCar camera.")
            return None

        height, width = image_bgr.shape[:2]
        target_w = int(getattr(self.config, "camera_width", width))
        target_h = int(getattr(self.config, "camera_height", height))
        if (width, height) != (target_w, target_h):
            image_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return image_bgr

    def _build_ego_matrix(self, location: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rotation.tolist()
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)

        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        rotation_matrix = rz @ ry @ rx

        ego_matrix = np.eye(4, dtype=np.float64)
        ego_matrix[:3, :3] = rotation_matrix
        ego_matrix[:3, 3] = location
        return ego_matrix

    def _compute_target_points_next_wp(
        self,
        location: np.ndarray,
        ego_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute target points where the second target is the next route waypoint.

        This mirrors the CARLA command planner behavior on pruned routes: the primary
        lookahead is ~7.5 m, and the next target is the immediate next waypoint,
        which for our QLabs routes (~1 m spacing) is approximately +1 m further.
        """
        if self.route_xy is None:
            raise RuntimeError("Route waypoints not initialised.")

        ego_xy = location[:2]
        # Find nearest waypoint to current position
        distances = np.linalg.norm(self.route_xy - ego_xy, axis=1)
        nearest_idx = int(np.argmin(distances))

        # Accumulate along route until lookahead distance is reached
        target_idx = nearest_idx
        accumulated = np.linalg.norm(self.route_xy[nearest_idx] - ego_xy)
        for i in range(nearest_idx, len(self.route_xy) - 1):
            if accumulated >= self.target_lookahead:
                target_idx = i
                break
            seg = np.linalg.norm(self.route_xy[i + 1] - self.route_xy[i])
            accumulated += seg
            target_idx = i + 1

        target_idx = min(target_idx, len(self.route_xy) - 1)
        next_idx = min(target_idx + 1, len(self.route_xy) - 1)

        target_world = self.route_xy[target_idx]
        next_target_world = self.route_xy[next_idx]

        # Transform world -> ego frame using ego_matrix inverse
        ego_inv = np.linalg.inv(ego_matrix)
        target_ego = self._world_to_ego(target_world, ego_inv)
        next_target_ego = self._world_to_ego(next_target_world, ego_inv)

        return target_ego, next_target_ego

    def _advance_along_route(self, start_idx: int, distance: float) -> np.ndarray:
        point = self.route_xy[start_idx]
        remaining = max(float(distance), 0.0)

        for idx in range(start_idx, len(self.route_xy) - 1):
            start = self.route_xy[idx]
            end = self.route_xy[idx + 1]
            segment = end - start
            seg_len = float(np.linalg.norm(segment))
            if seg_len < 1e-4:
                continue
            if remaining <= seg_len:
                return start + (segment * (remaining / seg_len))
            remaining -= seg_len
            point = end
        return point

    def _world_to_ego(self, point_xy: np.ndarray, ego_inv: np.ndarray) -> np.ndarray:
        point_h = np.array([point_xy[0], point_xy[1], 0.0, 1.0], dtype=np.float64)
        transformed = ego_inv @ point_h
        return transformed[:2]

    def _estimate_speed(self, location: np.ndarray, timestamp: float) -> float:
        if self.prev_sample is None:
            return 0.0
        prev_location, prev_time = self.prev_sample
        dt = max(timestamp - prev_time, 1e-3)
        distance = float(np.linalg.norm(location[:2] - prev_location[:2]))
        return distance / dt

    def _write_frame(self, image_bgr: np.ndarray, measurement: dict) -> None:
        if self.paths is None:
            return

        image_path = self.paths.rgb_dir / f"{self.frame_idx:04d}.jpg"
        measurement_path = self.paths.measurements_dir / f"{self.frame_idx:04d}.json.gz"

        if not cv2.imwrite(str(image_path), image_bgr):
            print(f"WARNING: Failed to write image to {image_path}")
            return

        with gzip.open(measurement_path, "wt", encoding="utf-8") as fh:
            json.dump(measurement, fh, separators=(",", ":"))

        self.frame_idx += 1
