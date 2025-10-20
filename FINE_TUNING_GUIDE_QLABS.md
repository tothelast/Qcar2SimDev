# Fine-Tuning Simlingo Model for QLabs: Complete Guide

## Overview

Fine-tuning adapts the Simlingo model (trained on CARLA) to work with QLabs simulator data. The process involves collecting QLabs data, adapting the data loader, configuring training, and running the training pipeline.

---

## Critical: Model Architecture & Loading

**What gets fine-tuned:**
- **Language Model (Qwen2-0.5B):** Fine-tuned with LoRA (rank 32, alpha 64)
- **Vision Model (InternViT-300M):** FROZEN (no gradient updates)

**Where models come from:**
- Models are downloaded from HuggingFace **during training** (NOT from `pretrained/` folder)
- `pretrained/InternVL2-1B/` is only used for inference, not training
- `models/simlingo/checkpoints/epoch=013.ckpt/` contains LoRA weights (starting point for fine-tuning)

**Critical for fine-tuning:**
```bash
python train.py \
  --config-name=qlabs_finetune \
  checkpoint=../../models/simlingo/checkpoints/epoch=013.ckpt  # ← REQUIRED
```
Without this checkpoint, training starts from scratch (much slower).

---

## Step 1: Collect QLabs Data

**What you need:** Images, vehicle speed, waypoints, and labels from QLabs

**CRITICAL: Data Directory Structure**

Your QLabs data MUST be organized exactly like CARLA data. The dataset loader uses this glob pattern:
```python
glob.glob(f"{data_path}/data/simlingo/*/*/*/Town*")
```

This requires the following structure:
```
data_path/
└── data/
    └── simlingo/
        ├── routes_training/          # Training split
        │   └── {dataset_name}/       # e.g., "qlabs"
        │       └── {route_id}/       # e.g., "Rep0_0"
        │           └── Town*/        # MUST start with "Town" (e.g., "TownQLabs" or "Town01")
        │               ├── rgb/
        │               │   ├── 0000.jpg
        │               │   ├── 0001.jpg
        │               │   └── ...
        │               └── measurements/
        │                   ├── 0000.json.gz
        │                   ├── 0001.json.gz
        │                   └── ...
        └── routes_validation/        # Validation split
            └── {dataset_name}/
                └── {route_id}/
                    └── Town*/
```

**Example for QLabs:**
```
database/qcar2_simlingo/
└── data/
    └── simlingo/
        ├── routes_training/
        │   └── qlabs/
        │       ├── Rep0_0/
        │       │   └── TownQLabs/
        │       │       ├── rgb/
        │       │       └── measurements/
        │       └── Rep0_1/
        │           └── TownQLabs/
        └── routes_validation/
            └── qlabs/
                └── Rep0_Val/
                    └── TownQLabs/
```

**Important Notes:**
- The directory name MUST start with "Town" (e.g., "TownQLabs", "Town01", "TownCityscape")
- The `boxes/` folder (used in CARLA for bounding boxes) is NOT required for Simlingo training
- The loader filters by 'routes_training' and 'routes_validation' in the path for train/val splits

**CRITICAL: Measurement File Format**

Each `measurements/{seq:04}.json.gz` must contain:
```json
{
  "ego_matrix": [[4x4 transformation matrix]],
  "route_original": [[x1, y1], [x2, y2], ...],
  "route": [[x1, y1], [x2, y2], ...],
  "target_point": [x, y],
  "target_point_next": [x, y],
  "speed": 5.5,
  "augmentation_rotation": 0.0,
  "augmentation_translation": 0.0,
  "command": 4,
  "next_command": 4
}
```

**Field explanations:**
- `ego_matrix`: 4×4 transformation matrix for coordinate conversion
- `route_original`, `route`: Future route waypoints (at least 20 points)
- `target_point`, `target_point_next`: Navigation targets in ego frame
- `speed`: Vehicle speed in m/s
- `augmentation_rotation`, `augmentation_translation`: Set to 0.0 for QLabs (no augmentation)
- `command`, `next_command`: High-level commands (1-6). Only required if `route_as` includes 'command'
  - 1: Turn left, 2: Turn right, 3: Go straight, 4: Follow road, 5: Lane change left, 6: Lane change right

**CRITICAL: Image Format**

- Format: JPEG (.jpg)
- Resolution: 1024×512 (width × height)
- Color space: BGR (will be converted to RGB by loader)
- Location: `rgb/{seq:04}.jpg`

**Image Preprocessing Pipeline (automatic during training):**
1. **Dynamic Preprocessing:** Images are split into 448×448 patches based on aspect ratio
2. **Normalization:** ImageNet normalization applied (MEAN: [0.485, 0.456, 0.406], STD: [0.229, 0.224, 0.225])
3. **Interpolation:** Bicubic interpolation for resizing

**Reference files:**
- `simlingo/simlingo_training/dataloader/dataset_base.py` (lines 444-481) - Image loading
- `simlingo/simlingo_training/dataloader/dataset_base.py` (lines 359-390) - Measurement loading
- `simlingo/simlingo_training/dataloader/dataset_base.py` (lines 484-540) - Prompt generation
- `simlingo/simlingo_training/dataloader/datamodule.py` (lines 249) - Image preprocessing
- `simlingo/simlingo_training/utils/internvl2_utils.py` (lines 179-267) - Preprocessing pipeline

---

## Step 2: Create QLabs Data Loader

**File to modify:** `simlingo/simlingo_training/dataloader/dataset_driving.py`

**What to do:**
1. Create a new class `Data_Driving_QLabs` (inherit from `Data_Driving`)
2. Override `__init__` to load QLabs data structure
3. Override `__getitem__` to return DatasetOutput with QLabs data

```python
class Data_Driving_QLabs(Data_Driving):
    def __init__(self, split="train", bucket_name="all", **cfg):
        super().__init__(**cfg)
        # Parent __init__ scans data_path/data/simlingo/ for routes
        # and populates self.images, self.measurements, self.sample_start

    def __getitem__(self, idx):
        # Load measurements, images, waypoints (use parent class methods)
        # Generate prompt based on route_as configuration

        # Conversation format for VLM
        conversation_all = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"{answer}"}
            ]}
        ]

        # placeholder_values: Used when prompt contains <TARGET_POINT> tokens
        # Example: {'<TARGET_POINT>': [[x1, y1], [x2, y2]]}
        placeholder_values = {'<TARGET_POINT>': target_points} if '<TARGET_POINT>' in prompt else {}

        return DatasetOutput(
            conversation=conversation_all,
            answer=conversation_answer,
            image_ff=image_tensor,  # [1, 3, 1024, 512]
            image_ff_org_size=image_size,
            waypoints=waypoints_list,
            waypoints_1d=waypoints_1d,
            path=route_path,
            target_points=target_points,
            speed=speed_value,
            placeholder_values=placeholder_values,
            measurement_path=measurement_path,
            dataset='driving',
        )
```

**Reference:** `simlingo_training/dataloader/dataset_driving.py` (lines 33-319)

---

## Step 3: Create QLabs Training Configuration

**File to create:** `simlingo/simlingo_training/config/experiment/qlabs_finetune.yaml`

**Copy from:** `simlingo/simlingo_training/config/experiment/simlingo_seed1.yaml`

**Key parameters to modify:**
```yaml
# @package _global_
defaults:
  - /data_module: carla_no_buckets  # ← CHANGE: Use no_buckets for QLabs (no bucket filtering)
  - /model/vision_model: vlm
  - /data_module/base_dataset: dataset

model:
  lr: 3e-5                    # Learning rate (keep same for fine-tuning)
  predict_route_as_wps: True
  speed_wps_mode: 2d
  language_model:
    variant: 'OpenGVLab/InternVL2-1B'
    lora: True
    lora_alpha: 64            # LoRA alpha (keep same)
    lora_r: 32                # LoRA rank (keep same)
    lora_dropout: 0.1
  vision_model:
    variant: 'OpenGVLab/InternVL2-1B'

data_module:
  base_dataset:
    data_path: /path/to/database/qcar2_simlingo  # ← CHANGE: Path to your QLabs data root
                                                  #   The loader will look for: {data_path}/data/simlingo/routes_*/
    bucket_path: null                             # ← CHANGE: Not used for QLabs
    bucket_name: "all"                            # ← CRITICAL: Use "all" to load all data (no filtering)
    route_as: "target_point"                      # ← CRITICAL: Navigation conditioning mode
                                                  #   "target_point" → uses <TARGET_POINT> tokens (recommended for QLabs)
                                                  #   "command" → uses high-level commands (requires command fields)
                                                  #   "target_point_command" → uses both
    use_commentary: false                         # ← CHANGE: Set to false for QLabs (no commentary data)
    use_qa: false                                 # ← CHANGE: Set to false for QLabs (no QA data)
    cut_bottom_quarter: false                     # ← CHANGE: Set based on your image format
    num_route_points: 20
    pred_len: 11
    hist_len: 1
    skip_first_n_frames: 10                       # Skip first N frames of each route
    img_augmentation: false                       # ← CHANGE: Disable for QLabs (no augmented images)
    img_shift_augmentation: false                 # ← CHANGE: Disable for QLabs (no augmented images)
  batch_size: 4                                   # ← CHANGE: Adjust based on GPU memory
  num_workers: 4
  driving_dataset:
    _target_: simlingo_training.dataloader.dataset_driving.Data_Driving_QLabs  # ← CHANGE

max_epochs: 15                # Adjust based on data size
precision: 16-mixed
strategy: deepspeed_stage_2
gpus: 1                       # Adjust based on available GPUs
seed: 42
name: qlabs_finetune
```

**Key Configuration Notes:**

1. **Bucket System:** Set `bucket_name: "all"` to disable bucket filtering (CRITICAL)
2. **Data Path:** Points to root directory (e.g., `/path/to/database/qcar2_simlingo`)
   - Loader will look for: `{data_path}/data/simlingo/routes_training/` and `routes_validation/`
3. **Directory Structure:** MUST match glob pattern `data/simlingo/*/*/*/Town*`
   - Example: `data/simlingo/routes_training/qlabs/Rep0_0/TownQLabs/`
4. **route_as:** Controls prompt format:
   - `"target_point"`: Prompt includes "Target waypoint: <TARGET_POINT><TARGET_POINT>." (recommended)
   - `"command"`: Prompt includes "Command: {command} in {distance} meter." (requires command fields)
   - `"target_point_command"`: Uses both (requires command fields)
5. **Image Augmentation:** Disable for QLabs (no augmented image folders)
6. **Commentary/QA:** Disable for QLabs (no commentary/QA data)

**Reference:** `simlingo_training/config/experiment/simlingo_seed1.yaml`

---

## Step 4: Run Training

**Command:**
```bash
cd simlingo/simlingo_training

python train.py \
  --config-name=qlabs_finetune \
  data_module.base_dataset.data_path=/path/to/qlabs/dataset \
  checkpoint=../../models/simlingo/checkpoints/epoch=013.ckpt
```

**What happens:**
1. Downloads InternVL2-1B from HuggingFace
2. Loads starting LoRA weights from checkpoint
3. Fine-tunes Qwen2-0.5B with LoRA (InternViT frozen)
4. Saves new checkpoint to `checkpoints/`

**Note:** Checkpoint parameter is CRITICAL - without it, training starts from scratch (very slow)

---

## Step 5: Use Fine-Tuned Model

Update checkpoint path in `core/config.py`:
```python
self.model_checkpoint_path = "models/simlingo/checkpoints/qlabs_epoch=XX.ckpt"
```

Inference code (`inference/simlingo_model.py`) automatically loads the new checkpoint.

---

## Quick Reference

| Task | File | Key Info |
|------|------|----------|
| Data structure | `dataset_base.py` (444-481) | Image loading, measurement format |
| Data loader | `dataset_driving.py` (33-319) | DatasetOutput structure |
| Configuration | `config/experiment/simlingo_seed1.yaml` | All parameters |
| Training | `train.py` (32-57) | Model loading, checkpoint |
| Inference | `core/config.py` (16) | Update checkpoint path |

---

## Checklist

- [ ] QLabs data in correct directory structure:
  - `{data_path}/data/simlingo/routes_training/{dataset}/{route_id}/Town*/rgb/`
  - `{data_path}/data/simlingo/routes_training/{dataset}/{route_id}/Town*/measurements/`
  - Directory MUST start with "Town" (e.g., "TownQLabs", "Town01")
- [ ] NO `boxes/` folder needed (not used by Simlingo)
- [ ] Measurement files have all required fields:
  - Always: ego_matrix, route_original, route, target_point, target_point_next, speed, augmentation_rotation, augmentation_translation
  - If route_as includes 'command': command, next_command
- [ ] Images are JPEG, 1024×512, BGR format
- [ ] `Data_Driving_QLabs` class created with proper conversation format
- [ ] `qlabs_finetune.yaml` config created:
  - `data_path` points to root (e.g., `/path/to/database/qcar2_simlingo`)
  - `bucket_name: "all"`
  - `route_as: "target_point"` (or your choice)
  - `use_commentary: false`, `use_qa: false`
  - `img_augmentation: false`, `img_shift_augmentation: false`
- [ ] Checkpoint exists: `models/simlingo/checkpoints/epoch=013.ckpt/`
- [ ] Training runs with checkpoint parameter
- [ ] Loss decreases during training
- [ ] Update `core/config.py` with new checkpoint path

---

## Troubleshooting

| Problem | Solution | File |
|---------|----------|------|
| Data loader error | Check data format matches CARLA | `dataset_driving.py` |
| Config not found | Verify YAML path and syntax | `qlabs_finetune.yaml` |
| Out of memory | Reduce batch_size in config | `qlabs_finetune.yaml` |
| Loss not decreasing | Check learning rate, data quality | `train.py` |
| Inference fails | Verify checkpoint path in config.py | `core/config.py` |

