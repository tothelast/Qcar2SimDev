import cv2
import numpy as np
import os
from datetime import datetime

class DebugVisualizer:
    def __init__(self, output_dir="debug_output/frames"):
        self.output_dir = output_dir
        # Create unique timestamped folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Colors (BGR)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_WP_ROUTE = (0, 255, 0)   # Green
        self.COLOR_WP_SPEED = (0, 255, 255) # Yellow
        self.COLOR_CMD = (0, 0, 255)        # Red

    def project_points(self, points_3d, intrinsics, extrinsics):
        """
        Project 3D points (in ego frame) to 2D image plane.
        Args:
            points_3d: (N, 3) or (N, 2) array in ego frame (x=forward, y=right/left, z=up)
            intrinsics: (3, 3) camera intrinsic matrix
            extrinsics: (4, 4) camera extrinsic matrix (world-to-camera or ego-to-camera)
        """
        if points_3d.shape[1] == 2:
            # Add z=0 if only x,y provided (assuming ground plane)
            points_3d = np.hstack([points_3d, np.zeros((len(points_3d), 1))])
            
        # Add homogeneous coordinate
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # Transform to camera frame
        # Note: extrinsics usually transform World -> Camera. 
        # Here points are in Ego frame. So we need Ego -> Camera.
        # Assuming the passed extrinsics are Ego -> Camera.
        points_cam = (extrinsics @ points_h.T).T
        
        # Project to image plane
        # z is forward in camera frame usually? 
        # Standard convention: x=right, y=down, z=forward
        # CARLA/Unreal convention: x=forward, y=right, z=up
        # We need to check the coordinate system of the camera processor.
        # Assuming standard CV convention for projection: u = fx * x/z + cx
        
        # If points_cam is (N, 4), take first 3
        points_cam = points_cam[:, :3]
        
        # Avoid division by zero
        mask = points_cam[:, 2] > 0.1
        points_cam = points_cam[mask]
        
        if len(points_cam) == 0:
            return []

        points_2d = (intrinsics @ points_cam.T).T
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]
        
        return points_2d[:, :2].astype(int)

    def save_frame(self, 
                   image, 
                   step, 
                   velocity, 
                   steer, 
                   target_speed_cmd, 
                   brake, 
                   desired_speed,
                   route_wps, 
                   speed_wps,
                   intrinsics,
                   extrinsics):
        """
        Draw debug info on image and save it.
        """
        # Ensure image is BGR for OpenCV
        # Input image is RGB (from qcar_interface)
        if image.shape[2] == 3:
            canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            canvas = image.copy()
        h, w = canvas.shape[:2]
        
        # 1. Draw Waypoints
        # Need to handle coordinate systems carefully.
        # For now, just try projecting. If it fails/looks wrong, we debug.
        # Assuming route_wps are (N, 2) in ego frame.
        
        # Convert tensors to numpy if needed
        if hasattr(intrinsics, 'cpu'): intrinsics = intrinsics.cpu().numpy()
        if hasattr(extrinsics, 'cpu'): extrinsics = extrinsics.cpu().numpy()
        
        # Project Route Waypoints (Green)
        # if route_wps is not None:
        #     pts = self.project_points(route_wps, intrinsics, extrinsics)
        #     for pt in pts:
        #         cv2.circle(canvas, tuple(pt), 3, self.COLOR_WP_ROUTE, -1)
                
        # 2. Draw Text Info
        # Create a semi-transparent overlay for text
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (400, 250), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        
        # Format raw speed waypoints (first 3 points)
        raw_wps_str = "[]"
        if speed_wps is not None and len(speed_wps) > 0:
            # speed_wps are 2D vectors? No, usually just scalars or 2D positions?
            # In control_converter: norm(speed_waypoints[2] - speed_waypoints[0])
            # So they are POSITIONS in the speed prediction head (Simlingo specific).
            # Let's show the norm of the first few relative to 0?
            # Or just the raw X,Y of the first couple.
            # Let's show the first 3 raw points.
            points_to_show = speed_wps[:3]
            raw_wps_str = str([list(map(lambda x: round(x, 2), p)) for p in points_to_show])

        info = [
            f"Step: {step}",
            f"Speed: {velocity:.2f} m/s",
            f"Desired: {desired_speed:.2f} m/s",
            f"Cmd Speed: {target_speed_cmd:.2f} m/s",
            f"Brake: {brake}",
            f"Steer: {steer:.3f}",
            f"Raw Speed WPs: {raw_wps_str}",
            f"WPs: {len(route_wps) if route_wps is not None else 0}"
        ]
        
        y = 30
        for line in info:
            color = self.COLOR_TEXT
            if "Brake: True" in line: color = (0, 0, 255) # Red for brake
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30
            
        # 3. Save
        filename = os.path.join(self.run_dir, f"frame_{step:04d}.jpg")
        cv2.imwrite(filename, canvas)
