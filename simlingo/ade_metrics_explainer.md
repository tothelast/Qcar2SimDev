
- For each route we have:
  - A recorded **human (teleop) trajectory** of the car.
  - An **ideal route polyline**: a sequence of 2D points describing the planned path.
- The model predicts a sequence of future 2D positions (waypoints) for the car.
- **ADE (Average Displacement Error)** = average Euclidean distance (in meters)
  between two trajectories, averaged over time steps and over all validation routes.

### Line 1: Model vs human (teleop) ADE

- For every validation sample:
  - Take the model’s predicted waypoints.
  - Take the human teleop waypoints for the same future time steps.
  - Compute the ADE between these two sequences.
- Then average this ADE over all validation routes.
- This line shows **how closely the model imitates the human driver**.

### Line 2: Model vs route ADE

- For every validation sample:
  - Take the model’s predicted waypoints.
  - Take the corresponding segment of the **route polyline**.
  - Truncate the route to F points so that its length matches the model’s and teleop’s future waypoint sequences, and then compute ADE over those F positions.
- Then average over all validation routes.
- This line shows **how closely the model follows the planned route geometry**.

### Line 3: Human (teleop) vs route ADE

- For every validation sample:
  - Take the human teleop waypoints.
  - Take the same segment of the route polyline.
  - Truncate the route to F points so that its length matches the model’s and teleop’s future waypoint sequences, and then compute ADE over those F positions.
- Then average over all validation routes.
- This is a **property of the validation data only**:
  - It does not use the model at all.
  - It is the same constant value for all training budgets (B1, B3, Full).
- The **human route‑following baseline**.

### What the x‑axis shows

- The x‑axis is the **number of training annotations** used to train each model.
- The validation set (and thus the human vs route ADE) stays the same for all points.
- Only the **model’s behavior** changes as we move from B1 → B3 → Full.

