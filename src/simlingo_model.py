"""
Simlingo Model Wrapper Module.
Handles model loading, tokenization, and inference.
"""

import sys
import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

# Add simlingo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simlingo'))

from transformers import AutoTokenizer, AutoProcessor
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from omegaconf import OmegaConf
import hydra


class SimlingoModelWrapper:
    """Wrapper for Simlingo model inference."""
    
    def __init__(self, config, device='cuda'):
        """
        Initialize Simlingo model wrapper.
        
        Args:
            config: SimlingoQCar2Config instance
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.cfg = None  # Hydra config

        # Special token IDs
        self.special_token_ids = {}

        # Custom high-level command (HLC)
        self.custom_hlc = None

    def set_hlc(self, command: str = None):
        """
        Set a custom high-level command to guide the vehicle's behavior.

        Args:
            command: Natural language instruction (e.g., "Drive carefully", "Speed up", "Turn left at the intersection")
                    Set to None to use default behavior

        Examples:
            - "Drive carefully and slow down"
            - "Speed up to match traffic"
            - "Prepare to turn left"
            - "Avoid the obstacle on the right"
        """
        self.custom_hlc = command
        if command:
            print(f"HLC set: '{command}'")
        else:
            print("HLC cleared (using default behavior)")

    def load_model(self, checkpoint_path: str = None):
        """
        Load Simlingo model from DeepSpeed checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint directory (default: from config)
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.model_checkpoint_path

        print(f"Loading Simlingo model from {checkpoint_path}...")

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load Hydra config
        hydra_config_path = self.config.hydra_config_path
        if not os.path.exists(hydra_config_path):
            raise FileNotFoundError(f"Hydra config not found: {hydra_config_path}")

        print(f"Loading Hydra config from {hydra_config_path}...")
        with open(hydra_config_path, 'r') as f:
            self.cfg = OmegaConf.load(f)

        # Set use_global_img flag
        self.cfg.model.vision_model.use_global_img = self.cfg.data_module.use_global_img

        # Load processor (needed for model instantiation)
        if self.processor is None:
            # Use local pretrained model to avoid HF authentication issues
            local_model_path = f"pretrained/{self.cfg.model.vision_model.variant.split('/')[1]}"
            print(f"Loading processor from local path: {local_model_path}...")
            self.processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True
            )

        # Instantiate model architecture using Hydra
        print("Instantiating model architecture...")
        cache_dir = f"pretrained/{self.cfg.model.vision_model.variant.split('/')[1]}"

        # Set default dtype to bfloat16 for model instantiation
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        # Change to simlingo directory for Hydra instantiation
        original_cwd = os.getcwd()
        simlingo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simlingo')
        os.chdir(simlingo_dir)

        try:
            self.model = hydra.utils.instantiate(
                self.cfg.model,
                cfg_data_module=self.cfg.data_module,
                processor=self.processor,
                cache_dir=cache_dir,
                _recursive_=False
            )
        finally:
            # Restore original directory
            os.chdir(original_cwd)
            torch.set_default_dtype(default_dtype)

        # Load state dict from checkpoint
        print(f"Loading weights from checkpoint...")
        if os.path.isdir(checkpoint_path):
            # DeepSpeed ZeRO checkpoint directory - use the pre-converted pytorch_model.pt
            pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.pt")
            if os.path.exists(pytorch_model_path):
                print(f"Loading from pre-converted checkpoint: {pytorch_model_path}")
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"pytorch_model.pt not found in {checkpoint_path}. "
                    "Please convert the DeepSpeed checkpoint first using zero_to_fp32.py"
                )
        else:
            # Single file checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Load state dict into model
        self.model.load_state_dict(state_dict)

        # Resize model embeddings to account for new special tokens
        # This is necessary because we added new tokens to the tokenizer
        self.model.language_model.model.resize_token_embeddings(len(self.tokenizer))

        # CRITICAL: Update the adaptor's embed_tokens reference
        # The LanguageAdaptor stores a reference to the old embedding layer
        # After resizing, we need to update it to point to the new embedding layer
        self.model.adaptors.language.embed_tokens = self.model.language_model.model.embed_tokens
        self.model.adaptors.language.lm_head = self.model.language_model.model.lm_head

        # Update the model's processor reference to use our tokenizer with added special tokens
        # This is critical because the model's replace_placeholder_tokens method uses
        # self.tokenizer.additional_special_tokens_ids[0] to determine which tokens are placeholders
        self.model.processor = self.processor
        self.model.vision_model.image_encoder.processor = self.processor

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")
        
    def load_tokenizer(self):
        """Load tokenizer and add special tokens."""
        # Use local pretrained model to avoid HF authentication issues
        local_model_path = f"pretrained/{self.config.encoder_variant.split('/')[1]}"
        print(f"Loading tokenizer from local path: {local_model_path}...")

        try:
            # Try loading as processor first (for InternVL2)
            self.processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.tokenizer = self.processor.tokenizer
        except:
            # Fall back to tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        
        # Add special tokens
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': self.config.special_tokens
        })

        # Set padding side
        self.tokenizer.padding_side = "left"

        # Store special token IDs
        for token in self.config.special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.special_token_ids[token] = token_id

        print("Tokenizer loaded successfully")
    
    def create_language_label(self, prompt: str, target_points: np.ndarray, num_patches: int = 2) -> LanguageLabel:
        """
        Create LanguageLabel from prompt and target points.

        Args:
            prompt: Prompt string with placeholders
            target_points: Target points array [[x1, y1], [x2, y2]]
            num_patches: Number of image patches (default: 2 for use_thumbnail=True)

        Returns:
            LanguageLabel instance
        """
        # Calculate number of image tokens
        # For InternVL2-1B: image_size=448, patch_size=14, downsample_ratio=0.5
        image_size = 448
        patch_size = 14
        downsample_ratio = 0.5
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        # Create image token string
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN

        # Add image tokens to prompt (prepend with newline)
        prompt_with_image = f"<image>\n{prompt}"
        # Replace <image> with actual image tokens
        prompt_with_image = prompt_with_image.replace('<image>', image_tokens, 1)

        # Create placeholder values dictionary
        # We need to provide values for ALL special tokens that appear in the prompt,
        # even if they don't actually need placeholder values (like image tokens).
        # This is because the Simlingo code will try to look up all special tokens
        # in the placeholder_values dictionary.
        #
        # For image tokens and special mode tokens, we provide empty arrays so the
        # lookup succeeds but no waypoint embeddings are created (since the array is empty).
        placeholder_values = {
            '<TARGET_POINT>': target_points,
            # Image tokens - provide empty arrays to avoid KeyError
            # These won't actually be used because they're replaced by vision embeddings
            '<img>': np.array([]),
            '</img>': np.array([]),
            '<IMG_CONTEXT>': np.array([]),
            # Special mode tokens - provide empty arrays to avoid KeyError
            # These are just text tokens, not placeholders for embeddings
            '<INSTRUCTION_FOLLOWING>': np.array([]),
            '<SAFETY>': np.array([])
        }

        # Convert to token IDs
        placeholder_batch_list = []
        tmp = {}
        for key, value in placeholder_values.items():
            # First try to get from our special_token_ids (for tokens we added)
            token_id = self.special_token_ids.get(key)
            # If not found, try to get from tokenizer (for pre-existing tokens like image tokens)
            if token_id is None:
                token_id = self.tokenizer.convert_tokens_to_ids(key)
                # Skip if it's the unknown token (token doesn't exist)
                if token_id == self.tokenizer.unk_token_id:
                    continue
            tmp[token_id] = value

        placeholder_batch_list.append(tmp)

        # Tokenize prompt
        prompt_batch_list = [prompt_with_image]
        tokenized = self.tokenizer(
            prompt_batch_list,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # Create LanguageLabel
        language_label = LanguageLabel(
            phrase_ids=tokenized['input_ids'].to(self.device),
            phrase_valid=tokenized['attention_mask'].bool().to(self.device),
            phrase_mask=tokenized['attention_mask'].bool().to(self.device),
            placeholder_values=placeholder_batch_list,
            language_string=prompt_batch_list,
            loss_masking=None
        )

        return language_label
    
    def inference(
        self,
        camera_images: torch.Tensor,
        image_sizes: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        vehicle_speed: float,
        target_point: np.ndarray,
        next_target_point: np.ndarray
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
        """
        Run Simlingo model inference.
        
        Args:
            camera_images: Processed camera images [1, 1, 1, 3, H, W]
            image_sizes: Image sizes [1, 2]
            camera_intrinsics: Camera intrinsics [1, 3, 3]
            camera_extrinsics: Camera extrinsics [1, 4, 4]
            vehicle_speed: Current speed in m/s
            target_point: Target point in ego frame [x, y]
            next_target_point: Next target point in ego frame [x, y]
            
        Returns:
            Tuple of (speed_waypoints, route_waypoints, language)
            - speed_waypoints: [1, F, 2] tensor or None
            - route_waypoints: [1, F, 2] tensor or None
            - language: String or None
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Create prompt (with optional custom HLC)
        if self.custom_hlc:
            # Use Dreamer dataset format for instruction following
            # Format: "<INSTRUCTION_FOLLOWING> Current speed: X m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. {instruction}"
            # This matches the Dreamer training data (50% of Dreamer samples)
            base_prompt = f"Current speed: {vehicle_speed:.2f} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. {self.custom_hlc}"
            prompt = f"<INSTRUCTION_FOLLOWING> {base_prompt}"
        else:
            # Use default prompt (matches standard driving data)
            # Format: "Current speed: X m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
            prompt = self.config.get_prompt_template(vehicle_speed)

        # Create target points array
        # EXPERIMENT: Set to zeros to test model without route information
        USE_REAL_TARGET_POINTS = True  # Set to True to use real target points
        if USE_REAL_TARGET_POINTS:
            target_points = np.array([target_point, next_target_point], dtype=np.float32)
        else:
            # Pass dummy zeros - model still expects <TARGET_POINT> tokens but gets no route info
            target_points = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            print("DEBUG: Using ZERO target points (no route information)")

        # Get number of patches from camera_images shape
        # camera_images shape: [1, 1, num_patches, 3, 448, 448]
        num_patches = camera_images.shape[2]

        # Create language label
        language_label = self.create_language_label(prompt, target_points, num_patches=num_patches)
        
        # Create vehicle speed tensor
        speed_tensor = torch.tensor([[vehicle_speed]], dtype=torch.float32).to(self.device)
        
        # Create target point tensor
        target_point_tensor = torch.from_numpy(target_point).unsqueeze(0).float().to(self.device)
        
        # Create DrivingInput
        # Note: camera_images must be bfloat16 to match model dtype
        # image_sizes can be None for InternVL2 model
        driving_input = DrivingInput(
            camera_images=camera_images.to(self.device).bfloat16(),
            image_sizes=image_sizes,  # None for InternVL2
            camera_intrinsics=camera_intrinsics.to(self.device),
            camera_extrinsics=camera_extrinsics.to(self.device),
            vehicle_speed=speed_tensor,
            target_point=target_point_tensor,
            prompt=language_label,
            prompt_inference=language_label
        )
        
        # Run inference
        with torch.no_grad():
            try:
                # Call model
                speed_wps, route_wps, language = self.model(driving_input)

                # Convert to float if not None
                if speed_wps is not None:
                    speed_wps = speed_wps.float()
                if route_wps is not None:
                    route_wps = route_wps.float()

                # Extract language string
                language_str = language[0] if language is not None and len(language) > 0 else None

                return speed_wps, route_wps, language_str
                
            except Exception as e:
                print(f"ERROR: Model inference failed: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
    
    def to(self, device):
        """Move model to device."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)

