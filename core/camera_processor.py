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
            image: Raw RGB image from QCar2 (H, W, 3) uint8

        Returns:
            Tuple of (processed_image, image_sizes)
        """
        # Bottom crop 30% to match CARLA training aspect ratio (2.0:1 -> 2.86:1)
        # height = image.shape[0]
        # cropped_height = int(height * 0.7)
        # image = image[:cropped_height, :, :]

        # Save first processed image (after crop, before JPEG) for debugging
        if not self.first_processed_saved:
            os.makedirs('debug_output', exist_ok=True)
            cv2.imwrite('debug_output/processed_after_crop.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.first_processed_saved = True

        # JPEG compression to match CARLA training artifacts
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, compressed_image = cv2.imencode('.jpg', image_bgr)
        image_bgr = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

