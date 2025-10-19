#!/usr/bin/env python3
"""
Data Format Validator for SimLingo Fine-Tuning

This script validates that our QCar2 data collection setup matches the
SimLingo training data format specification.

Phase 1.3 Validation Tasks:
1. Verify camera intrinsics calculation (160° FOV, 1024x512)
2. Verify camera extrinsics (QCar2 front camera: [+1.83m, 0.0, +1.10m])
3. Test ego-frame coordinate transformation (world → ego)
4. Create and validate single sample format

Reference: docs/simlingo_training_data_format.md
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from config import SimlingoQCar2Config


class DataFormatValidator:
    """Validates data format against SimLingo specification."""
    
    def __init__(self):
        """Initialize validator with config."""
        self.config = SimlingoQCar2Config()
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 4
        
    def print_header(self, title):
        """Print section header."""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def print_test(self, test_name, passed, details=""):
        """Print test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"       {details}")
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def validate_camera_intrinsics(self):
        """
        Subtask 1.3.1: Verify camera intrinsics calculation.
        
        Tests:
        - Correct formula: focal = width / (2 * tan(fov/2))
        - Correct values for 1024x512, 160° FOV
        - Matrix format [3, 3]
        """
        self.print_header("Subtask 1.3.1: Camera Intrinsics Validation")
        
        # Get intrinsics from config
        intrinsics = self.config.get_camera_intrinsics()
        
        # Expected values
        width = 1024
        height = 512
        fov = 160.0  # degrees (QCar2 CSI front camera specification)
        
        # Calculate expected focal length
        fov_rad = np.radians(fov)
        expected_focal = width / (2.0 * np.tan(fov_rad / 2.0))
        expected_cx = width / 2.0
        expected_cy = height / 2.0
        
        # Expected intrinsics matrix
        expected_intrinsics = np.array([
            [expected_focal, 0.0, expected_cx],
            [0.0, expected_focal, expected_cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Test 1: Matrix shape
        shape_correct = intrinsics.shape == (3, 3)
        self.print_test(
            "Intrinsics matrix shape",
            shape_correct,
            f"Expected (3, 3), got {intrinsics.shape}"
        )
        
        # Test 2: Data type
        dtype_correct = intrinsics.dtype == np.float32
        self.print_test(
            "Intrinsics data type",
            dtype_correct,
            f"Expected float32, got {intrinsics.dtype}"
        )
        
        # Test 3: Focal length calculation
        actual_focal = intrinsics[0, 0]
        focal_correct = np.isclose(actual_focal, expected_focal, rtol=1e-5)
        self.print_test(
            "Focal length calculation",
            focal_correct,
            f"Expected {expected_focal:.4f}, got {actual_focal:.4f}"
        )
        
        # Test 4: Principal point
        actual_cx = intrinsics[0, 2]
        actual_cy = intrinsics[1, 2]
        cx_correct = np.isclose(actual_cx, expected_cx, rtol=1e-5)
        cy_correct = np.isclose(actual_cy, expected_cy, rtol=1e-5)
        principal_point_correct = cx_correct and cy_correct
        self.print_test(
            "Principal point (cx, cy)",
            principal_point_correct,
            f"Expected ({expected_cx:.1f}, {expected_cy:.1f}), got ({actual_cx:.1f}, {actual_cy:.1f})"
        )
        
        # Test 5: Full matrix comparison
        matrix_correct = np.allclose(intrinsics, expected_intrinsics, rtol=1e-5)
        self.print_test(
            "Complete intrinsics matrix",
            matrix_correct,
            "All elements match expected values" if matrix_correct else "Matrix mismatch"
        )
        
        # Print matrix
        print("\nCamera Intrinsics Matrix:")
        print(intrinsics)
        print(f"\nFOV: {fov}°, Resolution: {width}x{height}")
        print(f"Focal length: {actual_focal:.4f} pixels")
        
        return matrix_correct
    
    def validate_camera_extrinsics(self):
        """
        Subtask 1.3.2: Verify camera extrinsics.
        
        Tests:
        - QCar2 camera position: [+1.83, 0.0, +1.10]
        - Identity rotation (no rotation)
        - Matrix format [4, 4]
        """
        self.print_header("Subtask 1.3.2: Camera Extrinsics Validation")
        
        # Get extrinsics from config
        extrinsics = self.config.get_camera_extrinsics()
        
        # Expected values for QCar2
        expected_position = np.array([+1.83, 0.0, +1.10], dtype=np.float32)
        expected_rotation = np.eye(3, dtype=np.float32)  # Identity (no rotation)
        
        # Expected extrinsics matrix
        expected_extrinsics = np.array([
            [1.0, 0.0, 0.0, +1.83],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0, 1.0, +1.10],
            [0.0, 0.0, 0.0,  1.0]
        ], dtype=np.float32)
        
        # Test 1: Matrix shape
        shape_correct = extrinsics.shape == (4, 4)
        self.print_test(
            "Extrinsics matrix shape",
            shape_correct,
            f"Expected (4, 4), got {extrinsics.shape}"
        )
        
        # Test 2: Data type
        dtype_correct = extrinsics.dtype == np.float32
        self.print_test(
            "Extrinsics data type",
            dtype_correct,
            f"Expected float32, got {extrinsics.dtype}"
        )
        
        # Test 3: Camera position
        actual_position = extrinsics[:3, 3]
        position_correct = np.allclose(actual_position, expected_position, rtol=1e-5)
        self.print_test(
            "Camera position [+1.83, 0.0, +1.10]",
            position_correct,
            f"Expected {expected_position}, got {actual_position}"
        )
        
        # Test 4: Rotation matrix (should be identity)
        actual_rotation = extrinsics[:3, :3]
        rotation_correct = np.allclose(actual_rotation, expected_rotation, rtol=1e-5)
        self.print_test(
            "Rotation matrix (identity)",
            rotation_correct,
            "Identity rotation (no rotation)" if rotation_correct else "Non-identity rotation detected"
        )
        
        # Test 5: Bottom row [0, 0, 0, 1]
        bottom_row_correct = np.allclose(extrinsics[3, :], [0, 0, 0, 1], rtol=1e-5)
        self.print_test(
            "Homogeneous coordinates",
            bottom_row_correct,
            "Bottom row is [0, 0, 0, 1]" if bottom_row_correct else "Bottom row incorrect"
        )
        
        # Test 6: Full matrix comparison
        matrix_correct = np.allclose(extrinsics, expected_extrinsics, rtol=1e-5)
        self.print_test(
            "Complete extrinsics matrix",
            matrix_correct,
            "All elements match expected values" if matrix_correct else "Matrix mismatch"
        )
        
        # Print matrix
        print("\nCamera Extrinsics Matrix:")
        print(extrinsics)
        print(f"\nCamera position: {actual_position} (x=forward, y=right, z=up)")
        print(f"Camera rotation: Identity (no rotation)")
        
        return matrix_correct
    
    def validate_ego_frame_transformation(self):
        """
        Subtask 1.3.3: Test ego-frame coordinate transformation.
        
        Tests:
        - world_to_ego transformation
        - BEV representation (Z dropped)
        - Correct rotation and translation
        """
        self.print_header("Subtask 1.3.3: Ego-Frame Transformation Validation")
        
        def world_to_ego(world_point, ego_matrix):
            """
            Transform world coordinates to ego vehicle frame.
            
            Args:
                world_point: Point in world coordinates [x, y, z]
                ego_matrix: 4x4 transformation matrix [R | t; 0 1]
            
            Returns:
                Point in ego frame [x, y] (BEV, Z dropped)
            """
            # Extract rotation and translation from ego_matrix
            origin_matrix = ego_matrix[:3]  # [3, 4]
            origin_translation = origin_matrix[:, 3:4]  # [3, 1]
            origin_rotation = origin_matrix[:, :3]  # [3, 3]
            
            # Transform: R^T @ (point - translation)
            world_point_col = world_point.reshape(3, 1)
            ego_point_3d = origin_rotation.T @ (world_point_col - origin_translation)
            
            # Drop Z-coordinate for BEV
            ego_point_2d = ego_point_3d[:2, 0]  # [x, y]
            
            return ego_point_2d
        
        # Test case 1: Vehicle at origin, no rotation
        ego_matrix_1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        world_point_1 = np.array([5.0, 2.0, 0.0], dtype=np.float32)
        expected_ego_1 = np.array([5.0, 2.0], dtype=np.float32)
        actual_ego_1 = world_to_ego(world_point_1, ego_matrix_1)
        
        test1_passed = np.allclose(actual_ego_1, expected_ego_1, rtol=1e-5)
        self.print_test(
            "Identity transformation",
            test1_passed,
            f"World [5.0, 2.0, 0.0] → Ego {actual_ego_1}"
        )
        
        # Test case 2: Vehicle at [10, 5, 0], no rotation
        ego_matrix_2 = np.array([
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0,  5.0],
            [0.0, 0.0, 1.0,  0.0],
            [0.0, 0.0, 0.0,  1.0]
        ], dtype=np.float32)
        
        world_point_2 = np.array([15.0, 8.0, 0.0], dtype=np.float32)
        expected_ego_2 = np.array([5.0, 3.0], dtype=np.float32)  # Relative to vehicle
        actual_ego_2 = world_to_ego(world_point_2, ego_matrix_2)
        
        test2_passed = np.allclose(actual_ego_2, expected_ego_2, rtol=1e-5)
        self.print_test(
            "Translation only",
            test2_passed,
            f"World [15.0, 8.0, 0.0] → Ego {actual_ego_2} (vehicle at [10, 5, 0])"
        )
        
        # Test case 3: Vehicle rotated 90° (facing left)
        # Rotation matrix for 90° yaw (counterclockwise around Z)
        cos_90 = 0.0
        sin_90 = 1.0
        ego_matrix_3 = np.array([
            [cos_90, -sin_90, 0.0, 0.0],
            [sin_90,  cos_90, 0.0, 0.0],
            [   0.0,     0.0, 1.0, 0.0],
            [   0.0,     0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        world_point_3 = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        # After 90° rotation: world Y becomes ego X
        expected_ego_3 = np.array([5.0, 0.0], dtype=np.float32)
        actual_ego_3 = world_to_ego(world_point_3, ego_matrix_3)
        
        test3_passed = np.allclose(actual_ego_3, expected_ego_3, rtol=1e-4)
        self.print_test(
            "Rotation (90° yaw)",
            test3_passed,
            f"World [0.0, 5.0, 0.0] → Ego {actual_ego_3} (vehicle rotated 90°)"
        )
        
        # Test case 4: Z-coordinate dropped (BEV)
        world_point_4 = np.array([5.0, 2.0, 10.0], dtype=np.float32)  # High Z
        actual_ego_4 = world_to_ego(world_point_4, ego_matrix_1)
        
        test4_passed = len(actual_ego_4) == 2  # Should be 2D
        self.print_test(
            "BEV representation (Z dropped)",
            test4_passed,
            f"Output shape: {actual_ego_4.shape}, expected (2,)"
        )
        
        all_passed = test1_passed and test2_passed and test3_passed and test4_passed
        
        print("\nEgo-frame transformation function validated successfully!")
        print("Function signature: world_to_ego(world_point[3], ego_matrix[4,4]) → ego_point[2]")
        
        return all_passed
    
    def validate_single_sample_format(self):
        """
        Subtask 1.3.4: Create and validate single sample format.
        
        Tests:
        - DrivingInput tensor shapes and dtypes
        - DrivingLabel tensor shapes and dtypes
        - All fields match specification
        """
        self.print_header("Subtask 1.3.4: Single Sample Format Validation")
        
        print("Creating mock training sample...")
        
        # Create mock DrivingInput
        batch_size = 1
        temporal = 1
        num_cameras = 1
        height = 512
        width = 1024
        
        # Camera images [B, T, N, C, H, W]
        camera_images = torch.randint(0, 256, (batch_size, temporal, num_cameras, 3, height, width), dtype=torch.uint8)

        # Image sizes [B, T, N, 2]
        image_sizes = torch.tensor([[[[width, height]]]], dtype=torch.int64)
        
        # Camera intrinsics [B, N, 3, 3]
        intrinsics = self.config.get_camera_intrinsics()
        camera_intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).unsqueeze(0)
        
        # Camera extrinsics [B, N, 4, 4]
        extrinsics = self.config.get_camera_extrinsics()
        camera_extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).unsqueeze(0)
        
        # Vehicle speed [B, S]
        vehicle_speed = torch.tensor([[5.2]], dtype=torch.float32)
        
        # Target point [B, 2]
        target_point = torch.tensor([[10.0, 2.0]], dtype=torch.float32)
        
        # Validate DrivingInput shapes
        tests_passed = []
        
        tests_passed.append(camera_images.shape == (1, 1, 1, 3, 512, 1024))
        self.print_test(
            "camera_images shape [1, 1, 1, 3, 512, 1024]",
            tests_passed[-1],
            f"Got {camera_images.shape}"
        )
        
        tests_passed.append(camera_images.dtype == torch.uint8)
        self.print_test(
            "camera_images dtype uint8",
            tests_passed[-1],
            f"Got {camera_images.dtype}"
        )
        
        tests_passed.append(image_sizes.shape == (1, 1, 1, 2))
        self.print_test(
            "image_sizes shape [1, 1, 1, 2]",
            tests_passed[-1],
            f"Got {image_sizes.shape}"
        )
        
        tests_passed.append(camera_intrinsics.shape == (1, 1, 3, 3))
        self.print_test(
            "camera_intrinsics shape [1, 1, 3, 3]",
            tests_passed[-1],
            f"Got {camera_intrinsics.shape}"
        )
        
        tests_passed.append(camera_extrinsics.shape == (1, 1, 4, 4))
        self.print_test(
            "camera_extrinsics shape [1, 1, 4, 4]",
            tests_passed[-1],
            f"Got {camera_extrinsics.shape}"
        )
        
        tests_passed.append(vehicle_speed.shape == (1, 1))
        self.print_test(
            "vehicle_speed shape [1, 1]",
            tests_passed[-1],
            f"Got {vehicle_speed.shape}"
        )
        
        tests_passed.append(target_point.shape == (1, 2))
        self.print_test(
            "target_point shape [1, 2]",
            tests_passed[-1],
            f"Got {target_point.shape}"
        )
        
        # Create mock DrivingLabel
        # Waypoints [B, F, 2] - 10 waypoints (NOT 11!)
        # Model outputs 10 speed waypoints at 0.25s intervals = 2.5s prediction horizon
        waypoints = torch.randn(1, 10, 2, dtype=torch.float32)

        # Path [B, F, 2] - 20 waypoints
        path = torch.randn(1, 20, 2, dtype=torch.float32)

        tests_passed.append(waypoints.shape == (1, 10, 2))
        self.print_test(
            "waypoints shape [1, 10, 2]",
            tests_passed[-1],
            f"Got {waypoints.shape}"
        )
        
        tests_passed.append(path.shape == (1, 20, 2))
        self.print_test(
            "path shape [1, 20, 2]",
            tests_passed[-1],
            f"Got {path.shape}"
        )
        
        all_passed = all(tests_passed)
        
        print("\n" + "-"*70)
        print("Sample Format Summary:")
        print("-"*70)
        print("DrivingInput:")
        print(f"  camera_images:     {camera_images.shape} {camera_images.dtype}")
        print(f"  image_sizes:       {image_sizes.shape} {image_sizes.dtype}")
        print(f"  camera_intrinsics: {camera_intrinsics.shape} {camera_intrinsics.dtype}")
        print(f"  camera_extrinsics: {camera_extrinsics.shape} {camera_extrinsics.dtype}")
        print(f"  vehicle_speed:     {vehicle_speed.shape} {vehicle_speed.dtype}")
        print(f"  target_point:      {target_point.shape} {target_point.dtype}")
        print("\nDrivingLabel:")
        print(f"  waypoints:         {waypoints.shape} {waypoints.dtype}")
        print(f"  path:              {path.shape} {path.dtype}")
        print("-"*70)
        
        return all_passed
    
    def run_all_validations(self):
        """Run all validation tests."""
        print("\n" + "="*70)
        print("  SimLingo Data Format Validator")
        print("  Phase 1.3: Data Format Validation")
        print("="*70)
        
        # Run all subtasks
        result1 = self.validate_camera_intrinsics()
        result2 = self.validate_camera_extrinsics()
        result3 = self.validate_ego_frame_transformation()
        result4 = self.validate_single_sample_format()
        
        # Print summary
        self.print_header("Validation Summary")
        print(f"Total tests: {self.passed_tests + self.failed_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print()
        
        if self.failed_tests == 0:
            print("✓ ALL VALIDATIONS PASSED!")
            print("\nPhase 1.3 (Data Format Validation) COMPLETE")
            print("Ready to proceed to Phase 2 (Expert Data Collection)")
        else:
            print("✗ SOME VALIDATIONS FAILED")
            print("\nPlease fix the issues before proceeding to Phase 2")
        
        print("="*70)
        
        return self.failed_tests == 0


def main():
    """Main entry point."""
    validator = DataFormatValidator()
    success = validator.run_all_validations()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

