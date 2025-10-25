# Implement QLabs Data Recorder for Simlingo Fine‑Tuning

## Goal
- Add a minimal, reliable data recorder that captures expert demonstrations in QLabs for Simlingo fine‑tuning and matches our inference contract exactly (tokens, timing, geometry, scaling).
- Do not change inference code. The dataset we write must be consumable by the existing Simlingo training dataloaders.

## Environment
- Control loop: 20 Hz
- Save frequency: 4 Hz (every 5th tick)
- Camera: save raw BGR JPEG 1024×512
- Coordinates: world in meters; ego frame x=forward, y=left; yaw in radians
- QLabs vehicle: `QLabsQCar2.set_velocity_and_request_state`

## Files to modify
- `data_collection/data_recorder.py` (implement)
- `data_collection/collect_data.py` (instantiate recorder, call record every 5 ticks)
- Do not modify inference files.

## Directory structure (match Simlingo)
- Root: `<DATA_ROOT>/data/simlingo/{routes_training|routes_validation}/{dataset_name}/{run_id}/TownQLabs/`
  - `rgb/0000.jpg`, `0001.jpg`, …
  - `measurements/0000.json.gz`, `0001.json.gz`, …
  - `results.json.gz`
- Use zero‑padded 4‑digit frame indices.

## Measurements JSON schema (per frame)
- `ego_matrix`: 4×4 pose matrix (world → ego)
- `speed`: float m/s (scalar)
- `target_point`: `[x, y]` in ego frame (7.5 m lookahead)
- `target_point_next`: `[x, y]` next lookahead point
- `route_original`: `[[x, y], …]` future route path (world is preferred; leave resampling/ego conversion to the dataloader)
- `route`: same as `route_original` (if no edits); keep identical unless you apply route edits
- `command`: int 1–6 (HLC). If no command logic, set 4 (“follow the road”)
- `next_command`: int 1–6; mirror command or compute from geometry if available
- `augmentation_rotation`: 0.0
- `augmentation_translation`: 0.0
- `dataset`: string (e.g., `"qlabs"`)
- Optional: `eval_infos` (null)

## Image saving
- Save raw camera frame as BGR JPEG 1024×512.
- If QLabs output is 820×410, upscale to 1024×512 with `cv2.resize` (INTER_AREA or INTER_CUBIC), no letterboxing.
- Do not crop here; the training dataloader handles preprocessing.

## Timing and indexing
- Run teleop loop at 20 Hz.
- Maintain a step counter; save every 5th frame (0,5,10,…).
- Ensure `000n.jpg` and `000n.json.gz` are saved atomically for the same index.

## Compute fields (reuse existing code)
- Pose/state
  - Use `QLabsQCar2.set_velocity_and_request_state` response and a simple `StateEstimator` to get world position `[x,y,z]`, yaw (radians), and speed (m/s).
  - Build `ego_matrix` (4×4) from translation and yaw (Z‑yaw rotation).
- Target points
  - Use `RouteManager.get_target_point` for world target and next target, then convert to ego with `CoordinateTransformer.world_to_ego`.
  - Use `SimlingoQCar2Config.target_point_lookahead = 7.5`.
- Route path
  - Use `config.route_waypoints` (world) and either:
    - write world points directly (preferred; the dataloader handles conversion), or
    - convert the next ~20 points to ego and save.
- HLC
  - If you don’t implement geometry‑based command, set `command = next_command = 4`.

## Results file
- `results.json.gz` minimal content to include the run:
  - `scores.score_composed = 100.0`, `scores.score_route = 100.0`
  - `num_infractions = 0`, `infractions = {}`

## Recorder API (proposed)
- `class DataRecorder(config, data_root, dataset_name, split)`
  - `start_run(route_name, run_id)`: create directories, initialize counters
  - `record(camera_bgr_np, state: {position, yaw, speed}, route_manager)`: if `step%5==0`, save `rgb/NNNN.jpg` and `measurements/NNNN.json.gz`
  - `finalize_run(success=True)`: write `results.json.gz`
- Make it robust:
  - Create dirs if not exist
  - Use try/except around file I/O; log and skip on error
  - Ensure indices remain in sync

## Hookup in collect_data.py
- Instantiate `DataRecorder` after QCar spawn; call `start_run(route_name, auto_run_id)` (e.g., timestamp).
- Every 20 Hz tick:
  - Grab camera with `qcar.get_image(camera=config.qcar2_camera)`.
  - Build state using pose plus speed computation.
  - Call `recorder.record(...)`.
- On exit: `recorder.finalize_run()`.

## Parity with inference
- Lookahead distance must remain 7.5 m.
- HLC default 4 unless you implement geometry‑based HLC.
- No changes to inference tokenization or prompts; training code builds the prompt from measurements, so keep field names/types identical to Simlingo’s CARLA dataset.
- 4 Hz saving cadence must be exact.

## Acceptance checks
- A single run directory contains matching counts for `rgb/*.jpg` and `measurements/*.json.gz`.
- A random `measurements/*.json.gz` has `ego_matrix`, `speed`, `target_point`, `target_point_next`, and `route` fields with sane numeric values.
- The dataloader (`simlingo/simlingo_training/dataloader/dataset_driving.py`) can iterate this run without modification (point `data_path` to your dataset root).

## Non‑goals
- Do not change inference stack, model wrapper, or token logic.
- Do not add dynamic instruction or safety flags here.

## Edge cases
- First few frames: allow saving from index 0000; if the system hasn’t moved yet, still save zero speed.
- Camera failures: skip save for that index; don’t crash the run.
- Directory collisions: new `run_id` per session (timestamp + scene).

If anything is ambiguous, prefer parity with the Simlingo CARLA dataset and the behavior of `agent_simlingo.py`; do not invent new fields.

