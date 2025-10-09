"""
Camera Image Preprocessing Module.
Handles image resizing, normalization, and camera parameter generation.
"""

import sys
import os
import numpy as np
import cv2
import torch
from typing import Tuple
from PIL import Image

# Add simlingo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simlingo'))

from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess


class CameraProcessor:
    """Processes camera images for Simlingo model input."""
    
    def __init__(self, config):
        """
        Initialize camera processor.

        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config

        # Pre-compute camera intrinsics and extrinsics
        self.intrinsics = self.config.get_camera_intrinsics()
        self.extrinsics = self.config.get_camera_extrinsics()

        # Build InternVL2 transform (448x448 images)
        self.transform = build_transform(input_size=448)
        self.image_size = 448
        self.use_global_img = False  # Don't use thumbnail (matches original Simlingo default)
        self.max_num_grid = 2  # Maximum number of image patches
        
    def process_image(self, image: np.ndarray) -> Tuple[torch.Tensor, None]:
        """
        Process raw camera image for Simlingo model using InternVL2 preprocessing.

        Args:
            image: Raw RGB image from QCar2 (H, W, 3) uint8

        Returns:
            Tuple of (processed_image, image_sizes)
            - processed_image: Tensor [1, 1, num_patches, 3, 448, 448] float32
            - image_sizes: None (not used by InternVL2 model)
        """
        # Apply JPEG compression/decompression to match training data
        # The Simlingo model was trained on JPEG-compressed images from CARLA
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Encode as JPEG and decode back
        _, compressed_image = cv2.imencode('.jpg', image_bgr)
        image_bgr = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        # Convert back to RGB
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Crop bottom of image (conservative 10% crop)
        # Original CARLA preprocessing removed 30% to eliminate hood/dashboard
        # QCar2 camera does NOT show hood (verified), so we use minimal 10% crop
        # This gives model 20% more road visibility while maintaining safety margin
        # Formula: 1.6/16 = 10% crop (was 4.8/16 = 30% crop)
        crop_height = int(image.shape[0] - (image.shape[0] * 1.6) // 16)
        image = image[:crop_height, :, :]

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)

        # Apply dynamic preprocessing (splits image into patches)
        # This handles aspect ratio and creates multiple 448x448 patches
        images = dynamic_preprocess(
            pil_image,
            image_size=self.image_size,
            use_thumbnail=self.use_global_img,
            max_num=self.max_num_grid
        )

        # Apply transform to each patch (resize to 448x448, normalize)
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)  # [num_patches, 3, 448, 448]

        # Add batch and temporal dimensions
        # [num_patches, 3, 448, 448] -> [1, 1, num_patches, 3, 448, 448]
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)

        # image_sizes is not used by InternVL2 model (set to None like in original agent)
        image_sizes = None

        return pixel_values, image_sizes
    
    def get_camera_intrinsics_tensor(self) -> torch.Tensor:
        """
        Get camera intrinsics as PyTorch tensor.
        
        Returns:
            Tensor [1, 3, 3] float32
        """
        intrinsics_tensor = torch.from_numpy(self.intrinsics).float()
        # Add batch dimension: (3, 3) -> (1, 3, 3)
        intrinsics_tensor = intrinsics_tensor.unsqueeze(0)
        return intrinsics_tensor
    
    def get_camera_extrinsics_tensor(self) -> torch.Tensor:
        """
        Get camera extrinsics as PyTorch tensor.
        
        Returns:
            Tensor [1, 4, 4] float32
        """
        extrinsics_tensor = torch.from_numpy(self.extrinsics).float()
        # Add batch dimension: (4, 4) -> (1, 4, 4)
        extrinsics_tensor = extrinsics_tensor.unsqueeze(0)
        return extrinsics_tensor
    
    def visualize_processed_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert processed image tensor back to displayable format.
        
        Args:
            image_tensor: Processed image tensor [1, 1, 1, 3, H, W]
            
        Returns:
            RGB image as numpy array (H, W, 3) uint8
        """
        # Remove batch and temporal dimensions
        image = image_tensor.squeeze(0).squeeze(0).squeeze(0)  # (3, H, W)
        
        # Convert to numpy and transpose
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        
        # Denormalize
        denormalized = np.zeros_like(image_np)
        for c in range(3):
            denormalized[:, :, c] = image_np[:, :, c] * self.config.imagenet_std[c] + self.config.imagenet_mean[c]
        
        # Convert to uint8
        denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
        
        return denormalized

