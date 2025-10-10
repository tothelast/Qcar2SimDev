# SimLingo Files Used in Integration

## Part 1: Files from Original SimLingo Codebase

### simlingo_training/utils/custom_types.py
Defines data structures (DrivingInput, LanguageLabel) used for model input/output formatting and type safety throughout the inference pipeline. Imported in `src/simlingo_model.py` to structure model inputs and handle language prompts.
- **Uses:** `DrivingInput` (camera images, intrinsics, extrinsics, speed, target points → structured model input), `LanguageLabel` (tokenized text → prompt structure)

### simlingo_training/utils/internvl2_utils.py
Provides image preprocessing functions (build_transform, dynamic_preprocess) that handle InternVL2-specific image transformations and patch splitting. Used in `src/camera_processor.py` to preprocess QCar2 camera images before feeding them to the model.
- **Uses:** `build_transform()` (→ torchvision transform), `dynamic_preprocess()` (PIL Image → list of 448x448 patches)

### simlingo_training/models/driving.py
Main DrivingModel class that orchestrates the vision encoder, language model, and adaptors to produce driving predictions from camera inputs. Instantiated via Hydra in `src/simlingo_model.py` when loading the model checkpoint.
- **Uses:** `DrivingModel.forward()` (DrivingInput → waypoints, language tokens)

### simlingo_training/models/encoder/vlm.py
VLMEncoderModel wrapper that initializes and manages the vision-language model encoder with configuration options for freezing and embedding dimensions. Instantiated as part of the DrivingModel architecture via Hydra configuration in `src/simlingo_model.py`.
- **Uses:** `VLMEncoderModel` (image patches → visual embeddings)

### simlingo_training/models/encoder/internvl2_model.py
LingoInternVLModel implementation that wraps the InternVL2 pretrained model and extracts visual embeddings from camera images. Instantiated by VLMEncoderModel in `simlingo_training/models/encoder/vlm.py` which is loaded via Hydra in `src/simlingo_model.py`.
- **Uses:** `LingoInternVLModel` (preprocessed images → vision tokens)

### simlingo_training/models/language_model/llm.py
LLM class that loads and manages the language model component (InternLM2) with LoRA fine-tuning support for generating text and trajectory tokens. Instantiated as part of the DrivingModel architecture via Hydra configuration in `src/simlingo_model.py`.
- **Uses:** `LLM` (vision tokens + text prompts → hidden states)

### simlingo_training/models/adaptors/adaptors.py
Contains DrivingAdaptor and LanguageAdaptor classes that convert between language model hidden states and driving-specific outputs (waypoints, speed predictions). Instantiated by DrivingModel in `simlingo_training/models/driving.py` which is loaded via Hydra in `src/simlingo_model.py`.
- **Uses:** `DrivingAdaptor` (hidden states → waypoints), `LanguageAdaptor` (embeds text tokens)

## What SimLingo is used for

SimLingo provides the AI model that takes camera images and outputs driving predictions (waypoints and speed). Our integration code in `src/` just adapts QCar2 inputs to SimLingo's format and converts SimLingo's outputs to QCar2 control commands. SimLingo is the brain, our code is the interface.

