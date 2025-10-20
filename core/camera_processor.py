"""Camera image preprocessing for Simlingo model input."""

import sys
from pathlib import Path

# Add simlingo directory to path for simlingo_training imports
simlingo_dir = Path(__file__).parent.parent / 'simlingo'
if str(simlingo_dir) not in sys.path:
    sys.path.insert(0, str(simlingo_dir))

import numpy as np
import cv2
import torch
from typing import Tuple
from PIL import Image

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
        self.use_global_img = False  
        self.max_num_grid = 2  
        
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
        # JPEG compression/decompression to match CARLA training data
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, compressed_image = cv2.imencode('.jpg', image_bgr)
        image_bgr = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Dynamic preprocessing splits image into patches
        pil_image = Image.fromarray(image)
        images = dynamic_preprocess(
            pil_image,
            image_size=self.image_size,
            use_thumbnail=self.use_global_img,
            max_num=self.max_num_grid
        )

        # Transform each patch and stack
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        # Add batch and temporal dimensions: [N, 3, 448, 448] -> [1, 1, N, 3, 448, 448]
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)

        return pixel_values, None
    
    def get_camera_intrinsics_tensor(self) -> torch.Tensor:
        """Get camera intrinsics as PyTorch tensor [1, 3, 3]."""
        return torch.from_numpy(self.intrinsics).float().unsqueeze(0)

    def get_camera_extrinsics_tensor(self) -> torch.Tensor:
        """Get camera extrinsics as PyTorch tensor [1, 4, 4]."""
        return torch.from_numpy(self.extrinsics).float().unsqueeze(0)
    
    def visualize_processed_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert processed image tensor back to displayable RGB format."""
        # Remove batch/temporal dims and convert to numpy
        image = image_tensor.squeeze(0).squeeze(0).squeeze(0)
        image_np = image.cpu().numpy().transpose(1, 2, 0)

        # Denormalize using ImageNet stats
        denormalized = image_np * self.config.imagenet_std + self.config.imagenet_mean

        # Convert to uint8
        return np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)

