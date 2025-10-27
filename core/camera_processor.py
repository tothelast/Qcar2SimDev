"""Camera image preprocessing for Simlingo model input."""

import sys
from pathlib import Path

# Add simlingo directory to path for simlingo_training imports
simlingo_dir = Path(__file__).parent.parent / 'simlingo'
if str(simlingo_dir) not in sys.path:
    sys.path.insert(0, str(simlingo_dir))

import numpy as np
import cv2
import os
import torch
from typing import Tuple
from PIL import Image

from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess


class CameraProcessor:
    """Processes camera images for Simlingo model input."""
    
    def __init__(self, config):
        """Initialize camera processor."""
        self.config = config

        # Camera intrinsics/extrinsics (legacy - required by DrivingInput but not used by model)
        self.intrinsics = self.config.get_camera_intrinsics()
        self.extrinsics = self.config.get_camera_extrinsics()

        # InternVL2 preprocessing
        self.transform = build_transform(input_size=448)
        self.image_size = 448
        self.use_global_img = False
        self.max_num_grid = 2

        # Debug image saving
        self.first_processed_saved = False
        
    def process_image(self, image: np.ndarray) -> Tuple[torch.Tensor, None]:
        """
        Process raw camera image for Simlingo model.

        Args:
            image: RGB image from QCar2 (H, W, 3) uint8 (already JPEG-compressed by QLabs)

        Returns:
            Tuple of (processed_image, image_sizes)
        """
        height = image.shape[0]
        crop_ratio = max(0.0, min(0.95, float(getattr(self.config, "camera_bottom_crop_ratio", 0.0))))
        if crop_ratio > 0.0:
            cropped_height = max(1, int(height * (1.0 - crop_ratio)))
            image = image[:cropped_height, :, :]


        if getattr(self.config, "resize_input_to_training_resolution", False):
            target_w = int(getattr(self.config, "camera_width", image.shape[1]))
            target_h = int(getattr(self.config, "camera_height", image.shape[0]))
            if (image.shape[1], image.shape[0]) != (target_w, target_h):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Dynamic preprocessing: split into 448x448 patches
        pil_image = Image.fromarray(image)
        images = dynamic_preprocess(pil_image, image_size=self.image_size,
                                   use_thumbnail=self.use_global_img, max_num=self.max_num_grid)

        # Transform and stack patches
        pixel_values = torch.stack([self.transform(img) for img in images])

        # Add batch and temporal dimensions: [N, 3, 448, 448] -> [1, 1, N, 3, 448, 448]
        return pixel_values.unsqueeze(0).unsqueeze(0), None
    
    def get_camera_intrinsics_tensor(self) -> torch.Tensor:
        """Get camera intrinsics tensor (legacy - required by DrivingInput but not used by model)."""
        return torch.from_numpy(self.intrinsics).float().unsqueeze(0)

    def get_camera_extrinsics_tensor(self) -> torch.Tensor:
        """Get camera extrinsics tensor (legacy - required by DrivingInput but not used by model)."""
        return torch.from_numpy(self.extrinsics).float().unsqueeze(0)

    def visualize_processed_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert processed image tensor back to displayable RGB."""
        image = image_tensor.squeeze(0).squeeze(0).squeeze(0)
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        denormalized = image_np * self.config.imagenet_std + self.config.imagenet_mean
        return np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
