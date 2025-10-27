#!/usr/bin/env python3
"""
Data Visualization Tool for QLabs Simlingo Dataset
Generates comprehensive visualizations of collected data with 10 frames (5x2 grid)
showing local/global maps, ego vehicle, speed, target points, and positions.
"""

import sys
import gzip
import json
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow, Circle, Rectangle, Wedge
from datetime import datetime


class DataVisualizer:
    """Visualizes QLabs collected data with comprehensive 10-frame layout."""
    
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        
    def find_all_routes(self):
        """Find all route directories."""
        simlingo_path = self.database_path / "data" / "simlingo"
        routes = []
        
        for split_dir in simlingo_path.iterdir():
            if not split_dir.is_dir():
                continue
            if split_dir.name not in ["routes_training", "routes_validation"]:
                continue
                
            for dataset_dir in split_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                for route_dir in dataset_dir.iterdir():
                    if not route_dir.is_dir():
                        continue
                        
                    for town_dir in route_dir.iterdir():
                        if town_dir.is_dir() and town_dir.name.startswith("Town"):
                            routes.append(town_dir)
                            
        return sorted(routes)
        
    def load_frame_data(self, route_path: Path, frame_idx: int):
        """Load image and measurement data for a specific frame."""
        rgb_path = route_path / "rgb" / f"{frame_idx:04d}.jpg"
        measurement_path = route_path / "measurements" / f"{frame_idx:04d}.json.gz"
        
        if not rgb_path.exists() or not measurement_path.exists():
            return None, None
            
        # Load image
        image = cv2.imread(str(rgb_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Load measurement
        with gzip.open(measurement_path, 'rt') as f:
            measurement = json.load(f)
            
        return image, measurement
        
    def world_to_ego(self, world_points, ego_matrix):
        """Transform world coordinates to ego vehicle frame."""
        ego_matrix = np.array(ego_matrix)
        
        # Inverse of ego_matrix transforms from world to ego
        ego_matrix_inv = np.linalg.inv(ego_matrix)
        
        # Convert points to homogeneous coordinates
        world_points = np.array(world_points)
        if world_points.shape[-1] == 2:
            # Add z=0 for 2D points
            world_points = np.column_stack([world_points, np.zeros(len(world_points))])
        
        ones = np.ones((world_points.shape[0], 1))
        world_points_h = np.column_stack([world_points, ones])
        
        # Transform to ego frame
        ego_points_h = (ego_matrix_inv @ world_points_h.T).T
        
        # Return x, y in ego frame
        return ego_points_h[:, :2]
        
    def extract_ego_position(self, ego_matrix):
        """Extract ego vehicle position from ego_matrix."""
        ego_matrix = np.array(ego_matrix)
        return ego_matrix[:3, 3]  # [x, y, z]
        
    def extract_ego_yaw(self, ego_matrix):
        """Extract ego vehicle yaw from ego_matrix."""
        ego_matrix = np.array(ego_matrix)
        # Yaw from rotation matrix (R[1,0] and R[0,0])
        yaw = np.arctan2(ego_matrix[1, 0], ego_matrix[0, 0])
        return yaw
        
    def visualize_frame(self, ax_img, ax_local, ax_global, image, measurement, frame_idx, route_name):
        """Visualize a single frame with image, local map, and global map."""
        
        # ========== SUBPLOT 1: Camera Image with Overlay ==========
        ax_img.imshow(image)
        ax_img.axis('off')
        
        # Add text overlay with frame info
        speed = measurement.get('speed', 0.0)
        ego_pos = self.extract_ego_position(measurement['ego_matrix'])
        
        text_str = f"Frame {frame_idx}\n"
        text_str += f"Speed: {speed:.2f} m/s\n"
        text_str += f"Pos: [{ego_pos[0]:.1f}, {ego_pos[1]:.1f}]"
        
        ax_img.text(0.02, 0.98, text_str,
                   transform=ax_img.transAxes,
                   fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white',
                   family='monospace')
        
        # ========== SUBPLOT 2: Local Map (Ego Frame) ==========
        ax_local.clear()
        ax_local.set_aspect('equal')
        ax_local.grid(True, alpha=0.3)
        ax_local.set_xlabel('X (m)', fontsize=8)
        ax_local.set_ylabel('Y (m)', fontsize=8)
        ax_local.set_title('Local Map (Ego Frame)', fontsize=9, fontweight='bold')
        
        # Draw ego vehicle at origin (0, 0) facing +X
        ego_length = 0.6
        ego_width = 0.3
        ego_rect = Rectangle((-ego_length/2, -ego_width/2), ego_length, ego_width,
                            linewidth=2, edgecolor='blue', facecolor='lightblue', zorder=10)
        ax_local.add_patch(ego_rect)
        
        # Draw ego heading arrow
        ax_local.arrow(0, 0, 1.0, 0, head_width=0.3, head_length=0.3,
                      fc='blue', ec='blue', linewidth=2, zorder=11)
        
        # Draw target points (already in ego frame)
        target_point = np.array(measurement['target_point'])
        next_target_point = np.array(measurement['target_point_next'])
        
        ax_local.plot(target_point[0], target_point[1], 'ro', markersize=10,
                     label='Target Point', zorder=15)
        ax_local.plot(next_target_point[0], next_target_point[1], 'rs', markersize=8,
                     label='Next Target', zorder=15)
        
        # Draw route in ego frame
        route_world = np.array(measurement['route'])[:, :2]  # Take only x, y
        route_ego = self.world_to_ego(route_world, measurement['ego_matrix'])
        
        ax_local.plot(route_ego[:, 0], route_ego[:, 1], 'g-', linewidth=2,
                     label='Planned Route', zorder=5, alpha=0.7)
        ax_local.plot(route_ego[:, 0], route_ego[:, 1], 'g.', markersize=4, zorder=6)
        
        # Set reasonable axis limits for local view
        max_range = 25
        ax_local.set_xlim(-5, max_range)
        ax_local.set_ylim(-max_range/2, max_range/2)
        
        ax_local.legend(loc='upper right', fontsize=7)
        
        # Add speed text
        ax_local.text(0.02, 0.02, f"Speed: {speed:.2f} m/s",
                     transform=ax_local.transAxes,
                     fontsize=8,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # ========== SUBPLOT 3: Global Map (World Frame) ==========
        ax_global.clear()
        ax_global.set_aspect('equal')
        ax_global.grid(True, alpha=0.3)
        ax_global.set_xlabel('X (m)', fontsize=8)
        ax_global.set_ylabel('Y (m)', fontsize=8)
        ax_global.set_title('Global Map (World Frame)', fontsize=9, fontweight='bold')
        
        # Extract ego position and yaw
        ego_pos = self.extract_ego_position(measurement['ego_matrix'])
        ego_yaw = self.extract_ego_yaw(measurement['ego_matrix'])
        
        # Draw ego vehicle in world frame
        ego_x, ego_y = ego_pos[0], ego_pos[1]
        
        # Rotated rectangle for ego vehicle
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)
        
        # Vehicle corners (local)
        corners_local = np.array([
            [ego_length/2, ego_width/2],
            [ego_length/2, -ego_width/2],
            [-ego_length/2, -ego_width/2],
            [-ego_length/2, ego_width/2],
            [ego_length/2, ego_width/2]  # Close the loop
        ])
        
        # Rotate and translate to world frame
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_world = (rotation_matrix @ corners_local.T).T + np.array([ego_x, ego_y])
        
        ax_global.plot(corners_world[:, 0], corners_world[:, 1], 'b-', linewidth=2, zorder=10)
        ax_global.fill(corners_world[:, 0], corners_world[:, 1], color='lightblue', alpha=0.6, zorder=9)
        
        # Draw heading arrow in world frame
        arrow_length = 1.5
        arrow_end_x = ego_x + arrow_length * cos_yaw
        arrow_end_y = ego_y + arrow_length * sin_yaw
        ax_global.arrow(ego_x, ego_y, arrow_end_x - ego_x, arrow_end_y - ego_y,
                       head_width=0.5, head_length=0.4, fc='blue', ec='blue',
                       linewidth=2, zorder=11)
        
        # Draw route in world frame
        route_world = np.array(measurement['route'])[:, :2]
        ax_global.plot(route_world[:, 0], route_world[:, 1], 'g-', linewidth=2,
                      label='Planned Route', zorder=5, alpha=0.7)
        ax_global.plot(route_world[:, 0], route_world[:, 1], 'g.', markersize=4, zorder=6)
        
        # Draw target points in world frame (transform from ego to world)
        target_ego = np.array([measurement['target_point']])
        next_target_ego = np.array([measurement['target_point_next']])
        
        # Transform target points to world frame
        ego_matrix = np.array(measurement['ego_matrix'])
        target_world = (ego_matrix[:2, :2] @ target_ego.T).T + ego_matrix[:2, 3]
        next_target_world = (ego_matrix[:2, :2] @ next_target_ego.T).T + ego_matrix[:2, 3]
        
        ax_global.plot(target_world[0, 0], target_world[0, 1], 'ro', markersize=10,
                      label='Target Point', zorder=15)
        ax_global.plot(next_target_world[0, 0], next_target_world[0, 1], 'rs', markersize=8,
                      label='Next Target', zorder=15)
        
        # Set axis limits based on route extent
        all_points = np.vstack([route_world, [[ego_x, ego_y]]])
        x_min, x_max = all_points[:, 0].min() - 10, all_points[:, 0].max() + 10
        y_min, y_max = all_points[:, 1].min() - 10, all_points[:, 1].max() + 10
        
        ax_global.set_xlim(x_min, x_max)
        ax_global.set_ylim(y_min, y_max)
        
        ax_global.legend(loc='upper right', fontsize=7)
        
        # Add position text
        position_str = f"Position: [{ego_pos[0]:.2f}, {ego_pos[1]:.2f}]\n"
        position_str += f"Yaw: {np.degrees(ego_yaw):.1f}°"
        ax_global.text(0.02, 0.02, position_str,
                      transform=ax_global.transAxes,
                      fontsize=8,
                      verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
    def generate_visualization(self, route_path: Path, output_path: str = None, num_frames: int = 10):
        """Generate comprehensive visualization with 10 frames (5x2 grid)."""
        
        route_name = route_path.relative_to(self.database_path)
        print(f"\nGenerating visualization for: {route_name}")
        
        # Find all available frames
        rgb_dir = route_path / "rgb"
        rgb_files = sorted(list(rgb_dir.glob("*.jpg")))
        
        if len(rgb_files) == 0:
            print(f"  ❌ No frames found in {rgb_dir}")
            return None
            
        num_available = len(rgb_files)
        print(f"  Found {num_available} frames")
        
        # Select random frames
        if num_available < num_frames:
            print(f"  ⚠️  Only {num_available} frames available (requested {num_frames})")
            frame_indices = list(range(num_available))
        else:
            frame_indices = sorted(random.sample(range(num_available), num_frames))
            
        print(f"  Selected frames: {frame_indices}")
        
        # Create figure with 10 subplots (5 rows x 2 columns)
        # Each row has 3 subplots: image (left, wide) and two maps (right, stacked)
        fig = plt.figure(figsize=(24, 30))
        
        # Create main title
        fig.suptitle(f'QLabs Data Visualization - {route_name}\n'
                    f'10 Random Frames from {num_available} total frames',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Create grid: 5 rows, each row has image + 2 maps
        # Layout: [Image | Local Map]
        #         [      | Global Map]
        
        for row_idx in range(5):
            frame_idx = frame_indices[row_idx] if row_idx < len(frame_indices) else 0
            
            # Load frame data
            image, measurement = self.load_frame_data(route_path, frame_idx)
            
            if image is None or measurement is None:
                print(f"  ⚠️  Failed to load frame {frame_idx}")
                continue
            
            # Create subplots for this row
            # Position: [left, bottom, width, height]
            row_start = 0.02 + (4 - row_idx) * 0.195  # 5 rows, each 19.5% height
            
            # Image (left side, 50% width)
            ax_img = fig.add_axes([0.02, row_start, 0.45, 0.18])
            
            # Local map (top right, 45% width, 9% height)
            ax_local = fig.add_axes([0.52, row_start + 0.09, 0.45, 0.09])
            
            # Global map (bottom right, 45% width, 9% height)
            ax_global = fig.add_axes([0.52, row_start, 0.45, 0.09])
            
            # Visualize this frame
            self.visualize_frame(ax_img, ax_local, ax_global, image, measurement,
                               frame_idx, route_name)
        
        # Fill remaining rows if we have fewer than 10 frames
        for row_idx in range(len(frame_indices), 5):
            if row_idx < 5:
                row_start = 0.02 + (4 - row_idx) * 0.195
                ax_empty = fig.add_axes([0.02, row_start, 0.95, 0.18])
                ax_empty.text(0.5, 0.5, 'No data available',
                            ha='center', va='center', fontsize=14, color='gray')
                ax_empty.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"visualization_{timestamp}.png"
            
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved visualization to: {output_path}")
        
        plt.close()
        return output_path
        
    def visualize_all_routes(self, output_dir: str = "visualizations"):
        """Generate visualizations for all routes."""
        routes = self.find_all_routes()
        
        if not routes:
            print("❌ No routes found!")
            return
            
        print(f"Found {len(routes)} routes to visualize")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for route_path in routes:
            route_name = route_path.relative_to(self.database_path)
            safe_name = str(route_name).replace('/', '_')
            output_path = output_dir / f"{safe_name}.png"
            
            try:
                self.generate_visualization(route_path, str(output_path))
            except Exception as e:
                print(f"  ❌ Error visualizing {route_name}: {e}")
                import traceback
                traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize QLabs collected data with comprehensive 10-frame layout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all routes in database
  python tools/visualize_collected_data.py

  # Visualize specific route
  python tools/visualize_collected_data.py \\
      --route database/data/simlingo/routes_training/qlabs/Rep_20251026_205939/TownQLabs

  # Specify custom output directory
  python tools/visualize_collected_data.py --output-dir my_visualizations

Visualization Layout:
  Each figure shows 10 random frames (5 rows x 2 columns):
  - Left: Camera image with overlaid speed and position
  - Top Right: Local map (ego frame) with target points and route
  - Bottom Right: Global map (world frame) with ego vehicle and route
        """
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default='/home/garegin/Documents/Qcar2SimDev/database',
        help='Path to database root directory'
    )
    
    parser.add_argument(
        '--route',
        type=str,
        default=None,
        help='Path to specific route directory (relative to database or absolute)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations (default: visualizations/)'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Number of frames to visualize (default: 10)'
    )
    
    args = parser.parse_args()
    
    visualizer = DataVisualizer(args.database)
    
    if args.route:
        # Visualize specific route
        route_path = Path(args.route)
        if not route_path.is_absolute():
            route_path = Path(args.database) / route_path
            
        if not route_path.exists():
            print(f"❌ Route not found: {route_path}")
            sys.exit(1)
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        route_name = route_path.relative_to(args.database)
        safe_name = str(route_name).replace('/', '_')
        output_path = output_dir / f"{safe_name}.png"
        
        visualizer.generate_visualization(route_path, str(output_path), args.num_frames)
    else:
        # Visualize all routes
        visualizer.visualize_all_routes(args.output_dir)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
