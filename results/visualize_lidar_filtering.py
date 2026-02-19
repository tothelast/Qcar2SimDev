"""
Visualization: How LiDAR points are filtered to only keep obstacles within the lane.

Generates a 4-panel figure showing each stage of the filtering pipeline:
  Panel 1: Raw LiDAR scan in vehicle frame (polar → cartesian)
  Panel 2: Detection cone filter (keep only points ahead within the cone)
  Panel 3: Transform to world frame + show route with lane boundaries
  Panel 4: Lane boundary filter (perpendicular distance check) → final result
"""
import numpy as np
import matplotlib.pyplot as plt


# --- Parameters (same as test_acc_baseline.py) ---
LIDAR_HALF_WIDTH = 5.0
STOP_DISTANCE = 8.0
MIN_DETECT_DIST = 1.0
ROAD_LANE_HALF_WIDTH = 1.0

# --- Build a curved route (simulating a roundabout section) ---
t = np.linspace(np.pi * 0.6, np.pi * 0.1, 80)
radius = 15.0
route_x = radius * np.cos(t)
route_y = radius * np.sin(t) + 25.0
waypoints = np.column_stack([route_x, route_y])

# --- Car state: on the curve ---
car_wp_idx = 30
car_pos = waypoints[car_wp_idx]
# Heading = tangent direction at this waypoint
tang = waypoints[car_wp_idx + 1] - waypoints[car_wp_idx - 1]
car_heading = np.arctan2(tang[1], tang[0])

# --- Simulate LiDAR points ---
np.random.seed(42)

def make_points(n, angle_range, dist_range):
    angles = np.random.uniform(*angle_range, n)
    dists = np.random.uniform(*dist_range, n)
    return angles, dists

# Obstacle car: ~5m ahead, slightly left of center
obs_angles, obs_dists = np.array([0.05, -0.02, 0.08, -0.05, 0.03]), np.array([5.0, 5.1, 4.9, 5.2, 5.0])

# Wall/curb outside the curve: to the right, 3-6m away
wall_angles = np.random.uniform(0.4, 1.0, 15)
wall_dists = np.random.uniform(3.0, 7.0, 15)

# Buildings far away and to the sides
bldg_angles = np.concatenate([np.random.uniform(-1.5, -0.8, 10), np.random.uniform(1.0, 1.5, 8)])
bldg_dists = np.concatenate([np.random.uniform(8.0, 15.0, 10), np.random.uniform(6.0, 12.0, 8)])

# Behind the car
behind_angles = np.random.uniform(2.5, 3.8, 12)
behind_dists = np.random.uniform(2.0, 10.0, 12)

# Combine all
all_angles = np.concatenate([obs_angles, wall_angles, bldg_angles, behind_angles])
all_dists = np.concatenate([obs_dists, wall_dists, bldg_dists, behind_dists])

# --- Convert to vehicle frame ---
x_veh = np.sin(all_angles) * all_dists
y_veh = np.cos(all_angles) * all_dists

# --- Cone filter ---
in_cone = (y_veh > MIN_DETECT_DIST) & (y_veh < STOP_DISTANCE) & (np.abs(x_veh) < LIDAR_HALF_WIDTH)

# --- World frame transform ---
cos_h, sin_h = np.cos(car_heading), np.sin(car_heading)
x_world = car_pos[0] + x_veh * sin_h + y_veh * cos_h
y_world = car_pos[1] - x_veh * cos_h + y_veh * sin_h

# --- Lane filter ---
def perp_distance(px, py, wps):
    d = np.sqrt((wps[:, 0] - px)**2 + (wps[:, 1] - py)**2)
    idx = np.argmin(d)
    n = len(wps)
    if idx == 0:
        dx, dy = wps[1, 0] - wps[0, 0], wps[1, 1] - wps[0, 1]
    elif idx == n - 1:
        dx, dy = wps[-1, 0] - wps[-2, 0], wps[-1, 1] - wps[-2, 1]
    else:
        dx, dy = wps[idx+1, 0] - wps[idx-1, 0], wps[idx+1, 1] - wps[idx-1, 1]
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        dx, dy = dx / length, dy / length
    vx, vy = px - wps[idx, 0], py - wps[idx, 1]
    return abs(vx * (-dy) + vy * dx), idx

in_lane = np.zeros(len(x_world), dtype=bool)
perp_dists = np.zeros(len(x_world))
for i in range(len(x_world)):
    perp_dists[i], _ = perp_distance(x_world[i], y_world[i], waypoints)
    in_lane[i] = perp_dists[i] <= ROAD_LANE_HALF_WIDTH

# --- Lane boundaries for plotting ---
def lane_boundaries(wps, hw):
    n = len(wps)
    lx, ly, rx, ry = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if i == 0: dx, dy = wps[1,0]-wps[0,0], wps[1,1]-wps[0,1]
        elif i == n-1: dx, dy = wps[-1,0]-wps[-2,0], wps[-1,1]-wps[-2,1]
        else: dx, dy = wps[i+1,0]-wps[i-1,0], wps[i+1,1]-wps[i-1,1]
        l = np.sqrt(dx**2+dy**2)
        if l > 0: dx, dy = dx/l, dy/l
        px, py = -dy, dx
        lx[i], ly[i] = wps[i,0]+px*hw, wps[i,1]+py*hw
        rx[i], ry[i] = wps[i,0]-px*hw, wps[i,1]-py*hw
    return lx, ly, rx, ry

lx, ly, rx, ry = lane_boundaries(waypoints, ROAD_LANE_HALF_WIDTH)

# ===================== PLOTTING =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("How LiDAR Points Are Filtered to Only Keep Obstacles Within the Lane", fontsize=15, fontweight='bold', y=0.98)

# --- Color categories ---
n_obs = len(obs_angles)
n_wall = len(wall_angles)
n_bldg = len(bldg_angles)
n_behind = len(behind_angles)
colors = np.array(
    ['red'] * n_obs + ['orange'] * n_wall + ['purple'] * n_bldg + ['gray'] * n_behind
)
labels_map = {'red': 'Obstacle car', 'orange': 'Wall/curb', 'purple': 'Building', 'gray': 'Behind car'}

# ============ PANEL 1: Raw LiDAR in vehicle frame ============
ax = axes[0, 0]
ax.set_title("Step 1: Raw LiDAR → Vehicle Frame", fontsize=12, fontweight='bold')
for c, lab in labels_map.items():
    mask = colors == c
    ax.scatter(x_veh[mask], y_veh[mask], c=c, s=30, label=lab, zorder=3, edgecolors='k', linewidths=0.3)
ax.plot(0, 0, 'gs', markersize=12, label='Car (origin)', zorder=5)
ax.annotate('', xy=(0, 2), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.set_xlabel("x_veh (left/right) [m]")
ax.set_ylabel("y_veh (forward) [m]")
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 16)
ax.text(0.02, 0.02, "x = sin(angle) × dist\ny = cos(angle) × dist", transform=ax.transAxes,
        fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============ PANEL 2: Cone filter in vehicle frame ============
ax = axes[0, 1]
ax.set_title("Step 2: Detection Cone Filter (Vehicle Frame)", fontsize=12, fontweight='bold')
# Draw cone
cone_rect = plt.Rectangle((-LIDAR_HALF_WIDTH, MIN_DETECT_DIST),
                           2*LIDAR_HALF_WIDTH, STOP_DISTANCE - MIN_DETECT_DIST,
                           fill=True, facecolor='orange', alpha=0.15, edgecolor='orange', linewidth=2, label='Detection cone')
ax.add_patch(cone_rect)
# Points outside cone (faded)
out_cone = ~in_cone
for c, lab in labels_map.items():
    mask_out = (colors == c) & out_cone
    mask_in = (colors == c) & in_cone
    if np.any(mask_out):
        ax.scatter(x_veh[mask_out], y_veh[mask_out], c=c, s=20, alpha=0.15, zorder=2)
    if np.any(mask_in):
        ax.scatter(x_veh[mask_in], y_veh[mask_in], c=c, s=40, label=f'{lab} (in cone)', zorder=3, edgecolors='k', linewidths=0.5)
ax.plot(0, 0, 'gs', markersize=12, zorder=5)
ax.annotate('', xy=(0, 2), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.set_xlabel("x_veh [m]")
ax.set_ylabel("y_veh [m]")
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 16)
ax.text(0.02, 0.02, f"Keep if:\n  y > {MIN_DETECT_DIST}m  AND  y < {STOP_DISTANCE}m\n  |x| < {LIDAR_HALF_WIDTH}m",
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============ PANEL 3: World frame + route ============
ax = axes[1, 0]
ax.set_title("Step 3: Transform to World Frame", fontsize=12, fontweight='bold')
ax.plot(route_x, route_y, 'b-', linewidth=2.5, label='Route centerline', zorder=1)
ax.plot(lx, ly, 'b--', linewidth=1, alpha=0.5, label='Lane boundaries')
ax.plot(rx, ry, 'b--', linewidth=1, alpha=0.5)
# All cone points in world frame
cone_x_w, cone_y_w = x_world[in_cone], y_world[in_cone]
cone_colors = colors[in_cone]
for c, lab in labels_map.items():
    mask = cone_colors == c
    if np.any(mask):
        ax.scatter(cone_x_w[mask], cone_y_w[mask], c=c, s=40, label=lab, zorder=3, edgecolors='k', linewidths=0.5)
# Car
dx_arrow = 1.5 * np.cos(car_heading)
dy_arrow = 1.5 * np.sin(car_heading)
ax.arrow(car_pos[0], car_pos[1], dx_arrow, dy_arrow, head_width=0.5, head_length=0.3, fc='green', ec='green', zorder=5)
ax.plot(car_pos[0], car_pos[1], 'gs', markersize=10, zorder=5)
ax.set_xlabel("X world [m]")
ax.set_ylabel("Y world [m]")
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.02, "x_w = px + x_veh·sin(θ) + y_veh·cos(θ)\ny_w = py − x_veh·cos(θ) + y_veh·sin(θ)",
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ============ PANEL 4: Lane filter → final result ============
ax = axes[1, 1]
ax.set_title("Step 4: Lane Boundary Filter → Final Result", fontsize=12, fontweight='bold')
ax.plot(route_x, route_y, 'b-', linewidth=2.5, label='Route centerline', zorder=1)
ax.plot(lx, ly, 'b--', linewidth=1, alpha=0.5, label=f'Lane boundary (±{ROAD_LANE_HALF_WIDTH}m)')
ax.plot(rx, ry, 'b--', linewidth=1, alpha=0.5)
# Shade the lane
ax.fill(np.concatenate([lx, rx[::-1]]), np.concatenate([ly, ry[::-1]]),
        alpha=0.08, color='blue', label='Lane area')

# Cone points: rejected vs accepted
cone_in_lane = in_lane[in_cone]
# Rejected (faded)
rej = ~cone_in_lane
if np.any(rej):
    ax.scatter(cone_x_w[rej], cone_y_w[rej], c='gray', s=30, alpha=0.25, marker='x',
               linewidths=1.5, label='Rejected (outside lane)', zorder=2)
# Accepted
acc = cone_in_lane
if np.any(acc):
    ax.scatter(cone_x_w[acc], cone_y_w[acc], c='red', s=80, marker='o',
               edgecolors='darkred', linewidths=1.5, label='ACCEPTED (in lane) → STOP', zorder=4)

# Draw a perpendicular distance example for one rejected point
ex_idx_rej = np.where(rej)[0]
if len(ex_idx_rej) > 0:
    ei = ex_idx_rej[0]
    ep = np.array([cone_x_w[ei], cone_y_w[ei]])
    ed = np.sqrt((waypoints[:, 0] - ep[0])**2 + (waypoints[:, 1] - ep[1])**2)
    nearest = waypoints[np.argmin(ed)]
    ax.plot([ep[0], nearest[0]], [ep[1], nearest[1]], 'k--', linewidth=1, alpha=0.6)
    mid = (ep + nearest) / 2
    pd = perp_dists[np.where(in_cone)[0][ei]]
    ax.annotate(f'{pd:.1f}m > {ROAD_LANE_HALF_WIDTH}m ✗', xy=mid, fontsize=8, color='gray',
                ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# Draw perpendicular distance example for one accepted point
ex_idx_acc = np.where(acc)[0]
if len(ex_idx_acc) > 0:
    ei = ex_idx_acc[0]
    ep = np.array([cone_x_w[ei], cone_y_w[ei]])
    ed = np.sqrt((waypoints[:, 0] - ep[0])**2 + (waypoints[:, 1] - ep[1])**2)
    nearest = waypoints[np.argmin(ed)]
    ax.plot([ep[0], nearest[0]], [ep[1], nearest[1]], 'r-', linewidth=1.5, alpha=0.8)
    mid = (ep + nearest) / 2
    pd = perp_dists[np.where(in_cone)[0][ei]]
    ax.annotate(f'{pd:.1f}m ≤ {ROAD_LANE_HALF_WIDTH}m ✓', xy=mid, fontsize=8, color='red', fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9))

# Car
ax.arrow(car_pos[0], car_pos[1], dx_arrow, dy_arrow, head_width=0.5, head_length=0.3, fc='green', ec='green', zorder=5)
ax.plot(car_pos[0], car_pos[1], 'gs', markersize=10, zorder=5)
ax.set_xlabel("X world [m]")
ax.set_ylabel("Y world [m]")
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.02, f"perp_dist = |V × T|\nKeep if perp_dist ≤ {ROAD_LANE_HALF_WIDTH}m\nNeed ≥ 3 points to confirm",
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("results/lidar_filtering_explained.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/lidar_filtering_explained.png")

