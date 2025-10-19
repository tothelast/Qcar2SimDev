## Data Collection Requirements for SimLingo Fine-Tuning in QLabs

Each recorded sample should provide all the inputs **SimLingo** expects, along with **ground-truth outputs**, in the same structure as the original.  

---

### 1. Camera Image
- **Source:** QCar’s front camera (preferably the CSI front camera).  
- **Resolution:** Use raw 820×410 px, then resize to **1024×512** to match SimLingo’s input size.  
- **Intrinsics:** Compute a 3×3 intrinsics matrix for the resized image (as described in the docs).  
  - QLabs’s front CSI has a wide **~160° FOV**.  
- **Extrinsics:** Record the camera’s mounting position and orientation.  
  - For QCar2’s front camera: `[+1.83 m forward, 0.0, +1.10 m up]` relative to car center, no rotation.  
- **Note:** This differs from CARLA’s setup; logging extrinsics ensures correct perspective geometry.

---

### 2. Vehicle State
- Log the **current speed (m/s)** from QLabs telemetry or compute from position deltas.  
- Used in the model prompt (e.g., “Current speed: 2.00 m/s…”) and as planning context.

---

### 3. Route Target Points
- Obtain from **RouteManager**:
  - Current target waypoint.
  - Next waypoint along the predefined route.
- Transform these into **ego-frame coordinates**.  
- Insert as `<TARGET_POINT>` placeholders in the model’s prompt.  
- For each sample, log vehicle position and heading to compute target waypoints.

---

### 4. Language Prompt String
- Use a **fixed template** during training:

```
"Current speed: {speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
```

- The wording isn’t critical — just keep `<TARGET_POINT>` tokens and stylistic consistency with original training data.

---

### 5. Ground-Truth Trajectory Waypoints
- Training labels: `DrivingLabel.waypoints/path`  
- Format: **11 future waypoints** (`x, y` in ego-frame), spaced ~0.2 s apart (~2.2 s horizon).  
- Derive from:
- Expert controller predictions (using kinematic model), **or**
- Actual recorded vehicle path (4 Hz sampling → ~0.25 s spacing).  
- Optionally log **route path 20 m ahead** (20 points, 1 m spacing).  
- All coordinates must be **in ego frame**.

---

### 6. Language “Answer”
- Default for driving: `"Waypoints:"`  
- No numeric coordinates in text.  
- Used by SimLingo as the cue for output generation.  
- If including commentary or Q&A, include relevant answers instead.

---

## Integration and Controller Notes

SimLingo produces **waypoints and speed profiles**, then a **controller** actuates the QCar:  
- **Lateral control:** PID steering.  
- **Longitudinal control:** Linear regression or PID throttle/brake.  
- May use a **kinematic bicycle model** for stability.  

Fine-tuning affects predicted waypoints but not controller logic.  
→ Train the model to maintain the same **interface assumptions** expected by the controller.

---

## Pre-Fine-Tuning Consistency Checks

### Camera Setup
- Ensure QLabs feed matches SimLingo’s format:
- 1024×512 image.
- Correct intrinsics (~160° FOV).
- Accurate extrinsics (front-mounted position).
- No model changes needed — just provide correct camera matrices in data.

### Simulation Scale and Units
- QLabs QCar2 uses **1:10 scale**, but internally treated as real-world meters.  
- Record all data in **m/s** and **meters**.  
- Maintain consistent full-scale units for images, speeds, and waypoints.

### Target Speed and Dynamics
- Typical target speed: **~2.0 m/s** for QLabs.  
- CARLA training likely used higher speeds — that’s fine.  
- Keep using consistent speed inputs in prompts.  
- Expect smoother, slower driving after fine-tuning; test with different speeds if needed.

---

**Summary:**  
Your dataset should replicate the SimLingo input-output structure, ensuring camera geometry, speed, route targets, prompts, and trajectory labels align with the model’s expectations. Fine-tuning will teach the model to adapt to QLabs’ environment without altering downstream controller behavior.
