"""Simlingo model wrapper for inference."""

import sys
import os
import importlib
from pathlib import Path

# Add simlingo directory to path for simlingo_training imports
simlingo_dir = Path(__file__).parent.parent / 'simlingo'
if str(simlingo_dir) not in sys.path:
    sys.path.insert(0, str(simlingo_dir))

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import defaultdict

from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from omegaconf import OmegaConf
import hydra


class SimlingoModelWrapper:
    """Wrapper for Simlingo model inference."""

    def __init__(self, config, device='cuda', nav_mode='target_point'):
        """Initialize model wrapper."""
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}, Nav mode: {nav_mode}")

        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.cfg = None
        self.conv_module = None
        self.tmp_config = None
        self.num_image_token = None

        # Configuration
        self.nav_mode = nav_mode
        self.task_type = 'driving'  # Default to 'driving' mode (matches fine-tuning data)
        self.safety_enabled = True
        self.user_question = None
        self.user_instruction = None

    def set_task_type(self, task_type: str, question: str = None, instruction: str = None, safety_enabled: bool = True):
        """
        Set task type for inference.

        Args:
            task_type: One of 'driving', 'commentary', 'qa', 'dreamer'
            question: Question text for 'qa' mode
            instruction: Instruction text for 'dreamer' mode
            safety_enabled: Enable safety flag for 'dreamer' mode
        """
        self.task_type = task_type
        self.user_question = question
        self.user_instruction = instruction
        self.safety_enabled = safety_enabled

        print(f"Task: {task_type}")
        if task_type == 'qa' and question:
            print(f"  Q: {question}")
        elif task_type == 'dreamer' and instruction:
            print(f"  Instruction: {instruction}, Safety: {'ON' if safety_enabled else 'OFF'}")



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
        print(f"Simlingo model moved to device: {self.device}")
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

        # CRITICAL: Add special tokens to tokenizer vocabulary
        # Only add tokens that should be in the vocabulary (not <SAFETY> or <INSTRUCTION_FOLLOWING>)
        # These are the tokens from agent_simlingo.py line 159
        special_tokens_to_add = [
            '<WAYPOINTS>',
            '<WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS_DIFF>',
            '<ORG_WAYPOINTS>',
            '<WAYPOINT_LAST>',
            '<ROUTE>',
            '<ROUTE_DIFF>',
            '<TARGET_POINT>'
        ]

        # IMPORTANT: Preserve existing additional_special_tokens
        # InternVL2 already has: <|im_start|>, <|im_end|>, <img>, </img>, <IMG_CONTEXT>, etc.
        # We need to append our tokens to the existing list, not replace it
        existing_tokens = self.tokenizer.additional_special_tokens.copy()
        all_tokens = existing_tokens + special_tokens_to_add

        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': all_tokens
        })

        print(f"Added {num_added} special tokens to tokenizer vocabulary")
        print(f"Total additional_special_tokens: {len(self.tokenizer.additional_special_tokens)}")

        # Set padding side
        self.tokenizer.padding_side = "left"

        # Store token ids for logging
        for token in special_tokens_to_add:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            print(f"  {token}: {token_id}")

        print("Tokenizer loaded successfully")
    
    def load_conversation_template(self):
        """Load conversation template module from pretrained model directory."""
        if self.conv_module is not None:
            return  # Already loaded

        cache_dir = f"pretrained/{self.cfg.model.vision_model.variant.split('/')[1]}"
        model_path = f"{cache_dir}/conversation.py"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Conversation template not found: {model_path}")

        # Import conversation module
        spec = importlib.util.spec_from_file_location('get_conv_template', model_path)
        self.conv_module = importlib.util.module_from_spec(spec)
        sys.modules['get_conv_template'] = self.conv_module
        spec.loader.exec_module(self.conv_module)

        print(f"Loaded conversation template from {model_path}")

    def calculate_num_image_tokens(self):
        """Calculate number of image tokens per patch."""
        if self.num_image_token is not None:
            return  # Already calculated

        if self.tmp_config is None:
            self.tmp_config = AutoConfig.from_pretrained(
                self.cfg.model.vision_model.variant,
                trust_remote_code=True
            )

        image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
        patch_size = self.tmp_config.vision_config.patch_size
        downsample_ratio = self.tmp_config.downsample_ratio

        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        print(f"Image tokens per patch: {self.num_image_token} (image_size={image_size}, patch_size={patch_size}, downsample_ratio={downsample_ratio})")

    def build_prompt(
        self,
        speed: float,
        target_points: Optional[np.ndarray] = None,
        hlc: Optional[int] = None
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """
        Build prompt string based on task type and navigational conditioning.

        Args:
            speed: Current vehicle speed in m/s
            target_points: Target points array [[x1, y1], [x2, y2]] (optional)
            hlc: High-level command (1-6) (optional)

        Returns:
            Tuple of (prompt_string, placeholder_values_dict)
        """
        # Build navigational conditioning (matches agent_simlingo.py lines 472-530)
        nav_conditioning = ""
        placeholder_values = {}

        if self.nav_mode == 'target_point' and target_points is not None:
            # Use target point tokens (matches agent_simlingo.py lines 472-486)
            nav_conditioning = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
            placeholder_values['<TARGET_POINT>'] = target_points

        elif self.nav_mode == 'command' and hlc is not None:
            # Use HLC text only, NO placeholder values (matches agent_simlingo.py lines 488-529)
            command_map = {
                1: 'go left at the next intersection',
                2: 'go right at the next intersection',
                3: 'go straight at the next intersection',
                4: 'follow the road',
                5: 'do a lane change to the left',
                6: 'do a lane change to the right',
            }
            command = command_map.get(hlc, 'follow the road')

            # Calculate distance to command
            dist = int(np.linalg.norm(target_points[0])) if target_points is not None else 10

            # Format command string
            if hlc == 4:  # "follow the road" doesn't need distance
                nav_conditioning = f"Command: {command}."
            else:
                nav_conditioning = f"Command: {command} in {dist} meter."

        # Build task-specific prompt
        if self.task_type == 'driving':
            # Pure driving mode (matches training data format)
            prompt = f"Current speed: {speed:.1f} m/s. {nav_conditioning} Predict the waypoints."

        elif self.task_type == 'commentary':
            # Commentary + Driving mode
            prompt = f"Current speed: {speed:.1f} m/s. {nav_conditioning} What should the ego do next?"

        elif self.task_type == 'qa':
            # Q&A mode
            question = self.user_question if self.user_question else "What is ahead?"
            prompt = f"Current speed: {speed:.1f} m/s. {nav_conditioning} Q: {question}"

        elif self.task_type == 'dreamer':
            # Dreamer mode (instruction following)
            instruction = self.user_instruction if self.user_instruction else "Follow the road"
            base_prompt = f"Current speed: {speed:.1f} m/s. {nav_conditioning} {instruction}"

            # Add safety flag as TEXT (not special token!)
            if self.safety_enabled:
                prompt = f"<SAFETY> {base_prompt}"
            else:
                prompt = f"<INSTRUCTION_FOLLOWING> {base_prompt}"
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        return prompt, placeholder_values

    def create_language_label(
        self,
        speed: float,
        target_points: Optional[np.ndarray] = None,
        hlc: Optional[int] = None,
        num_patches: int = 2
    ) -> LanguageLabel:
        """
        Create LanguageLabel using conversation template (matches agent_simlingo.py).

        Args:
            speed: Current vehicle speed in m/s
            target_points: Target points array [[x1, y1], [x2, y2]] (optional)
            hlc: High-level command (1-6) (optional)
            num_patches: Number of image patches (default: 2 for use_thumbnail=True)

        Returns:
            LanguageLabel instance
        """
        # Ensure conversation template is loaded
        self.load_conversation_template()

        # Ensure num_image_token is calculated
        self.calculate_num_image_tokens()

        # Build prompt and placeholder values
        prompt, placeholder_values = self.build_prompt(speed, target_points, hlc)

        # Debug: Print prompt (can be disabled for production)
        if False:  # Set to True to enable debug output
            print(f"\n[PROMPT] {prompt}")

        # Create conversation structure (matches agent_simlingo.py lines 564-578)
        conversation_all = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Waypoints:"},  # For structure only
                ],
            },
        ]

        # Flatten content (matches agent_simlingo.py lines 582-584)
        for i in range(len(conversation_all)):
            conversation_all[i]['content'] = conversation_all[i]['content'][0]['text']

        # Get conversation template
        template = self.conv_module.get_conv_template('internlm2-chat')

        # Add messages to template (matches agent_simlingo.py lines 616-626)
        question = conversation_all[0]['content']
        if '<image>' not in question:
            question = '<image>\n' + question

        for conv_part in conversation_all:
            if conv_part['role'] == 'assistant':
                # CRITICAL: Set message to None for inference (line 619)
                template.append_message(template.roles[1], None)
            elif conv_part['role'] == 'user':
                if '<image>' not in conv_part['content']:
                    conv_part['content'] = '<image>\n' + conv_part['content']
                template.append_message(template.roles[0], conv_part['content'])

        # Get formatted prompt
        query = template.get_prompt()

        # Remove system prompt to save tokens (matches agent_simlingo.py lines 629-631)
        system_prompt = template.system_template.replace('{system_message}', template.system_message) + template.sep
        query = query.replace(system_prompt, '')

        # Replace <image> with image tokens (matches agent_simlingo.py lines 633-639)
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

        # Convert placeholder values to token IDs (matches agent_simlingo.py lines 470-484)
        placeholder_batch_list = []

        # CRITICAL: Only add placeholder values when using target_point mode
        # When using command mode, placeholder_values is empty, and we pass an empty list
        # The model's replace_placeholder_tokens (internvl2_model.py line 60) checks:
        #   if special_ids.size(0) > 0 and len(placeholder_values) > 0:
        # So when placeholder_batch_list is empty, it skips placeholder replacement entirely

        if placeholder_values:
            # Only provide mappings for tokens with actual placeholder data (e.g., <TARGET_POINT>)
            tmp = defaultdict(lambda: np.array([]))
            for key, value in placeholder_values.items():
                token_id = self.tokenizer.convert_tokens_to_ids(key)
                tmp[token_id] = value
            placeholder_batch_list.append(tmp)
        # else: placeholder_batch_list remains empty (command mode)

        # Tokenize (matches agent_simlingo.py line 642)
        prompt_batch_list = [query]
        tokenized = self.tokenizer(
            prompt_batch_list,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False  # CRITICAL: Template already has special tokens!
        )

        # Create LanguageLabel (matches agent_simlingo.py lines 648-655)
        language_label = LanguageLabel(
            phrase_ids=tokenized['input_ids'].to(self.device),
            phrase_valid=(tokenized['input_ids'] != self.tokenizer.pad_token_id).to(self.device),
            phrase_mask=(tokenized['input_ids'] != self.tokenizer.pad_token_id).to(self.device),
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
        next_target_point: np.ndarray,
        hlc: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
        """
        Run Simlingo model inference.

        Args:
            camera_images: Processed camera images [1, 1, num_patches, 3, H, W]
            image_sizes: Image sizes [1, 2] (can be None for InternVL2)
            camera_intrinsics: Camera intrinsics [1, 3, 3]
            camera_extrinsics: Camera extrinsics [1, 4, 4]
            vehicle_speed: Current speed in m/s
            target_point: Target point in ego frame [x, y]
            next_target_point: Next target point in ego frame [x, y]
            hlc: High-level command (1-6) (optional, only used if nav_mode='command')

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

        # Create target points array
        target_points = np.array([target_point, next_target_point], dtype=np.float32)

        # Get number of patches from camera_images shape
        # camera_images shape: [1, 1, num_patches, 3, 448, 448]
        num_patches = camera_images.shape[2]

        # Create language label using conversation template
        language_label = self.create_language_label(
            speed=vehicle_speed,
            target_points=target_points,
            hlc=hlc,
            num_patches=num_patches
        )

        # Create vehicle speed tensor
        speed_tensor = torch.tensor([[vehicle_speed]], dtype=torch.float32).to(self.device)

        # Create target point tensor
        target_point_tensor = torch.from_numpy(target_point).unsqueeze(0).float().to(self.device)

        # Create DrivingInput (matches agent_simlingo.py lines 657-664)
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

                # Clean up language output: remove trailing "Waypoints:" separator
                # The model was trained to output "Waypoints:" before waypoint tokens,
                # but since we skip special tokens during decoding, we're left with just the separator
                if language_str:
                    language_str = language_str.rstrip()
                    if language_str.endswith("Waypoints:"):
                        language_str = language_str[:-len("Waypoints:")].rstrip()

                return speed_wps, route_wps, language_str

            except Exception as e:
                print(f"\n[ERROR] Model inference failed: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
    
