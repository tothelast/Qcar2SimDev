#!/usr/bin/env python3
"""
Interactive route.json builder for QLabs lanes (no teleop required).

What it does
- Connects to QLabs and spawns a Spline Line actor to visualize the route.
- Lets you add world-frame points in three ways:
  1) add x y                -> append a point directly
  2) move_car x y yaw_deg   -> place an existing QCar2 (ignores collisions)
     snapcar                -> append that car's current (x,y) as a point
  3) (optional) undo/clear/list/save commands

Output JSON schema (saved by `save path`):
{
  "name": "qlabs_lane",
  "points_world": [[x0, y0], [x1, y1], ...],
  "spacing_m": 0.5
}

Usage
  python scripts/route_builder.py --out route.json --actor 0 --name qlabs_lane --spacing 0.5

Notes
- Coordinates are QLabs world meters on the ground plane (z is assumed ~ 0).
- This tool only helps you author a centerline polyline; SimLingo sampling +
  ego-frame conversion happens in your integration at runtime.
- Requires QLabs to be running and accessible (default localhost).
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure local qvl/ is importable if not installed as a package
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
QVL_DIR = REPO_ROOT / "python"
if str(QVL_DIR) not in sys.path:
    sys.path.insert(0, str(QVL_DIR))

# QVL imports
from qvl.qlabs import QuanserInteractiveLabs
from qvl.spline_line import QLabsSplineLine
from qvl.qcar2 import QLabsQCar2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive route.json builder for QLabs")
    p.add_argument("--out", default="route.json", help="Output JSON path")
    p.add_argument("--name", default="qlabs_lane", help="Route name to embed in JSON")
    p.add_argument("--spacing", type=float, default=0.5, help="Desired resample spacing (m) to record in JSON")
    p.add_argument("--actor", type=int, default=0, help="ActorNumber of an existing QCar2 (for move_car/snapcar)")
    p.add_argument("--line_width", type=float, default=0.15, help="Spline line width (m) for visualization")
    p.add_argument("--host", default="localhost", help="QLabs host")
    return p.parse_args()


def draw_spline(spline: QLabsSplineLine, pts_xy: List[Tuple[float, float]], line_width: float) -> None:
    # SplineLine points are [x, y, z, width] in the actor's local frame.
    # Spawned at origin with zero rotation -> local == world.
    pts = [[float(x), float(y), 0.0, float(line_width)] for x, y in pts_xy]
    if len(pts) < 2:
        # Need at least 2 points to draw a line
        return
    spline.set_points(color=[1.0, 0.0, 0.0], pointList=pts, alignEndPointTangents=True)


def cmd_help() -> str:
    return (
        "Commands:\n"
        "  add x y              -> append a world (x,y) point\n"
        "  move_car x y yaw_deg -> place QCar2 (ignores collisions) at pose; auto-spawns if missing\n"
        "  snapcar              -> append QCar2 current (x,y) as a point\n"
        "  list                 -> print current points\n"
        "  undo                 -> remove last point\n"
        "  clear                -> remove all points\n"
        "  save [path]          -> write JSON (default: --out)\n"
        "  help                 -> show this help\n"
        "  exit/quit            -> exit\n"
    )


def main() -> None:
    args = parse_args()

    print("Connecting to QLabs at:", args.host)
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open(args.host):
        print("ERROR: Unable to connect to QLabs at", args.host)
        sys.exit(1)

    # Prepare Spline Line actor for visualization (spawn at origin, local==world)
    spline = QLabsSplineLine(qlabs)
    # configuration=QLabsSplineLine.CURVE produces a smooth curve; LINEAR also ok
    spline.spawn_degrees(location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1], configuration=QLabsSplineLine.CURVE, waitForConfirmation=True)

    # QCar2 handle for optional snap/move
    car = QLabsQCar2(qlabs)
    car.actorNumber = int(args.actor)

    points: List[Tuple[float, float]] = []
    out_path = Path(args.out)

    print("\nRoute builder ready. Type 'help' for commands.")
    print("Tip: Use 'move_car x y yaw_deg' to place the car over the lane center, then 'snapcar' to add that point.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("exit", "quit"):
                break
            elif cmd == "help":
                print(cmd_help())
            elif cmd == "add":
                if len(parts) < 3:
                    print("Usage: add x y")
                    continue
                x, y = float(parts[1]), float(parts[2])
                points.append((x, y))
                draw_spline(spline, points, args.line_width)
                print(f"Added ({x:.3f}, {y:.3f}). Total points: {len(points)}")
            elif cmd == "move_car":
                if len(parts) < 4:
                    print("Usage: move_car x y yaw_deg")
                    continue
                x, y, yaw_deg = float(parts[1]), float(parts[2]), float(parts[3])
                ok, *_ = car.set_transform_and_request_state_degrees(
                    location=[x, y, 0.005], rotation=[0, 0, yaw_deg],
                    enableDynamics=True, headlights=False, leftTurnSignal=False,
                    rightTurnSignal=False, brakeSignal=False, reverseSignal=False,
                    waitForConfirmation=True,
                )
                if not ok:
                    # Try to spawn this actor and retry placement
                    try:
                        status = car.spawn_id_degrees(
                            actorNumber=int(args.actor),
                            location=[x, y, 0.005],
                            rotation=[0, 0, yaw_deg],
                            scale=[1, 1, 1],
                            configuration=0,
                            waitForConfirmation=True,
                        )
                    except TypeError:
                        # Some QVL versions may not expose scale arg positionally; try without it
                        status = car.spawn_id_degrees(
                            actorNumber=int(args.actor),
                            location=[x, y, 0.005],
                            rotation=[0, 0, yaw_deg],
                            configuration=0,
                            waitForConfirmation=True,
                        )
                    if status == 0:
                        ok, *_ = car.set_transform_and_request_state_degrees(
                            location=[x, y, 0.005], rotation=[0, 0, yaw_deg],
                            enableDynamics=True, headlights=False, leftTurnSignal=False,
                            rightTurnSignal=False, brakeSignal=False, reverseSignal=False,
                            waitForConfirmation=True,
                        )
                print("Car placed" if ok else "Failed to place car (actor may not have spawned)")
            elif cmd == "snapcar":
                ok, loc, rot, scale = car.get_world_transform()
                if not ok:
                    print("ERROR: Could not read car world transform. Is the actorNumber correct?")
                    continue
                x, y = float(loc[0]), float(loc[1])
                points.append((x, y))
                draw_spline(spline, points, args.line_width)
                print(f"Snapped car position ({x:.3f}, {y:.3f}). Total points: {len(points)}")
            elif cmd == "list":
                if not points:
                    print("(no points)")
                else:
                    for i, (x, y) in enumerate(points):
                        print(f"{i:03d}: {x:.3f}, {y:.3f}")
            elif cmd == "undo":
                if points:
                    x, y = points.pop()
                    draw_spline(spline, points, args.line_width)
                    print(f"Removed ({x:.3f}, {y:.3f}). Total points: {len(points)}")
                else:
                    print("(no points to remove)")
            elif cmd == "clear":
                points.clear()
                draw_spline(spline, points, args.line_width)
                print("Cleared all points")
            elif cmd == "save":
                save_path = out_path
                if len(parts) >= 2:
                    save_path = Path(parts[1])
                data = {
                    "name": args.name,
                    "points_world": [[float(x), float(y)] for x, y in points],
                    "spacing_m": float(args.spacing),
                }
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"Saved {len(points)} points to {save_path}")
            else:
                print("Unknown command. Type 'help' for list of commands.")
        except Exception as e:
            print("ERROR:", e)

    print("Closing QLabs connection...")
    try:
        # Do not destroy actors; leave spline for visual inspection after exit
        qlabs.close()
    except Exception:
        pass
    print("Done.")


if __name__ == "__main__":
    main()

