#!/usr/bin/env python3
"""
Interactive tool to adjust the parking target for kink_street route.
This allows you to specify the exact parking coordinates and test different positions.
"""

import sys
import os
import json
import numpy as np

# Add parent directory and python directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python'))

from hal.products.mats import SDCSRoadMap


def print_current_config():
    """Print current configuration from the route file."""
    config_path = os.path.join(parent_dir, "config", "routes", "kink_street.json")
    
    if not os.path.exists(config_path):
        print("No existing kink_street.json found")
        return
    
    with open(config_path, 'r') as f:
        route_data = json.load(f)
    
    print("\nCurrent Route Configuration:")
    print(f"  Total waypoints: {route_data.get('num_waypoints', 'N/A')}")
    print(f"  Total distance: {route_data.get('total_distance', 'N/A'):.1f}m")
    
    if 'parking_target' in route_data:
        target = route_data['parking_target']
        print(f"  Parking target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]}]")
        print(f"  Parking heading: {route_data.get('parking_heading', 'N/A')}°")
    
    # Show last few waypoints
    waypoints = route_data.get('waypoints', [])
    if waypoints:
        print("\n  Last 5 waypoints:")
        for i, wp in enumerate(waypoints[-5:], start=len(waypoints)-4):
            print(f"    {i}: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]}]")


def calculate_parking_spot_relative_to_spot4(offset_x, offset_y):
    """
    Calculate parking spot coordinates relative to spot4.
    
    Args:
        offset_x: Offset in x direction (positive = east)
        offset_y: Offset in y direction (positive = north)
    
    Returns:
        [x, y, z] parking target coordinates
    """
    spot4_pos = [-13.0, -7.5, 0.001]
    spot4_heading_deg = -40.0
    
    # Convert to radians
    heading_rad = np.radians(spot4_heading_deg)
    
    # Create rotation matrix
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)
    
    # Rotate offset by the heading to get local coordinates
    # In local frame: +x is forward, +y is left
    rotated_x = offset_x * cos_h - offset_y * sin_h
    rotated_y = offset_x * sin_h + offset_y * cos_h
    
    parking_target = [
        spot4_pos[0] + rotated_x,
        spot4_pos[1] + rotated_y,
        0.001
    ]
    
    return parking_target, spot4_heading_deg


def regenerate_route_with_target(parking_target, parking_heading, approach_distance=10.0):
    """Regenerate route with new parking target."""
    # Import the generation function
    sys.path.insert(0, os.path.join(parent_dir, 'tools'))
    from generate_parking_route import generate_parking_route, downsample_waypoints
    
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    node_sequence = [18, 11, 12, 8]
    
    route_data = generate_parking_route(
        roadmap, 
        node_sequence, 
        parking_target, 
        parking_heading,
        "kink_street",
        approach_distance=approach_distance
    )
    
    if route_data:
        # Save to JSON file
        routes_dir = os.path.join(parent_dir, "config", "routes")
        filename = "kink_street.json"
        filepath = os.path.join(routes_dir, filename)
        
        # Backup existing file
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            if os.path.exists(backup_path):
                # Keep only one backup
                os.remove(backup_path)
            os.rename(filepath, backup_path)
        
        with open(filepath, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        # Save preview
        debug_dir = os.path.join(parent_dir, "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        preview_path = os.path.join(debug_dir, "kink_street_preview.json")
        
        preview_data = {
            "waypoints": route_data['waypoints'],
            "spawn_location": route_data['spawn_location'],
            "spawn_rotation": route_data['spawn_rotation'],
            "parking_target": route_data.get('parking_target'),
            "parking_heading": route_data.get('parking_heading')
        }
        
        with open(preview_path, 'w') as f:
            json.dump(preview_data, f, indent=2)
        
        print(f"\n✓ Route updated successfully")
        print(f"  Waypoints: {route_data['num_waypoints']}")
        print(f"  Distance: {route_data['total_distance']:.1f}m")
        print(f"  Saved to: {filepath}")
        
        return True
    
    return False


def main():
    print("="*80)
    print("PARKING TARGET ADJUSTMENT TOOL")
    print("="*80)
    
    print("\nReference Information:")
    print("  Spot4 vehicle: location=[-13.0, -7.5, 0.001], heading=-40.0°")
    print("  Coordinate system: +X = East, +Y = North")
    print("  Heading: -40° means facing Southwest")
    
    print_current_config()
    
    print("\n" + "="*80)
    print("PARKING TARGET OPTIONS")
    print("="*80)
    
    print("\nPreset Options:")
    print("  1. Behind spot4 (3.5m backward from car)")
    print("  2. Next to spot4, right side (3.5m perpendicular)")
    print("  3. Diagonal behind-right (2.5m back, 2.5m right)")
    print("  4. Custom position (specify coordinates)")
    print("  5. Custom offset from spot4 (specify offset)")
    print("  6. Exit without changes")
    
    try:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            # Behind: offset in -X direction (backward in car's local frame)
            parking_target, heading = calculate_parking_spot_relative_to_spot4(-3.5, 0.0)
            print(f"\nTarget: Behind spot4 at {parking_target}")
            
        elif choice == '2':
            # Right side: offset in +Y direction (right in car's local frame)
            parking_target, heading = calculate_parking_spot_relative_to_spot4(0.0, 3.5)
            print(f"\nTarget: Right of spot4 at {parking_target}")
            
        elif choice == '3':
            # Diagonal: offset in both directions
            parking_target, heading = calculate_parking_spot_relative_to_spot4(-2.5, 2.5)
            print(f"\nTarget: Diagonal behind-right of spot4 at {parking_target}")
            
        elif choice == '4':
            # Custom coordinates
            x = float(input("Enter X coordinate: "))
            y = float(input("Enter Y coordinate: "))
            heading = float(input("Enter heading in degrees (e.g., -40): "))
            parking_target = [x, y, 0.001]
            print(f"\nTarget: Custom position at {parking_target}")
            
        elif choice == '5':
            # Custom offset
            print("\nOffset in car's local frame:")
            print("  +X = forward, -X = backward")
            print("  +Y = left, -Y = right")
            offset_x = float(input("Enter X offset (meters): "))
            offset_y = float(input("Enter Y offset (meters): "))
            parking_target, heading = calculate_parking_spot_relative_to_spot4(offset_x, offset_y)
            print(f"\nTarget: Offset position at {parking_target}")
            
        elif choice == '6':
            print("\nExiting without changes.")
            return 0
            
        else:
            print("\nInvalid choice. Exiting.")
            return 1
        
        # Ask for approach distance
        approach_input = input("\nEnter approach distance in meters (default=10.0, larger=smoother): ").strip()
        approach_distance = float(approach_input) if approach_input else 10.0
        
        # Confirm
        print(f"\nGenerating route with:")
        print(f"  Parking target: [{parking_target[0]:.3f}, {parking_target[1]:.3f}]")
        print(f"  Parking heading: {heading}°")
        print(f"  Approach distance: {approach_distance}m")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        
        if confirm == 'y':
            if regenerate_route_with_target(parking_target, heading, approach_distance):
                print("\n" + "="*80)
                print("SUCCESS - Route has been updated!")
                print("="*80)
                print("\nNext steps:")
                print("  1. Test the route in your simulation")
                print("  2. Check debug_output/kink_street_preview.json for waypoint visualization")
                print("  3. If needed, run this script again to adjust")
                return 0
            else:
                print("\n✗ Failed to generate route")
                return 1
        else:
            print("\nCancelled.")
            return 0
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
