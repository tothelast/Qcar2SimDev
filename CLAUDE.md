# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QCar2SimDev integrates the **Simlingo** vision-language model with the **QCar2** robot (Quanser) in the QLabs simulator. The system enables:
1. **Data Collection**: Recording expert teleoperated driving demonstrations
2. **Model Training**: Fine-tuning Simlingo (InternVL2-1B vision + Qwen2-0.5B language with LoRA) on QCar2 data
3. **Inference**: Running the trained model at 4 Hz to control QCar2 autonomously

The model outputs **waypoint predictions** (future positions) that are converted to steering/throttle via PID control.

## Common Commands

### Run Inference (Autonomous Driving)
```bash
python inference/main.py --scene roundabout_navigation --nav-mode target_point
```

### Collect Training Data
```bash
python data_collection/collect_data.py --scene roundabout_navigation --route roundabout_navigation --num-runs 5
```

### Train Model
```bash
cd simlingo/simlingo_training
python train.py --config-name=qlabs_finetune \
  checkpoint=../../models/simlingo/checkpoints/epoch=013.ckpt \
  data_module.driving_dataset.data_path=../../database
```

### Evaluate Model
```bash
cd simlingo/simlingo_training
python eval.py checkpoint=path/to/checkpoint.pt data_path=../../database
```

### Install Dependencies
```bash
# PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Architecture

### Control Loop (4 Hz)
```
State Estimation → Model Inference → Control Conversion → QCar2 Actuation
     ↓                    ↓                  ↓
Camera + Position    W_route[20×2]     Steering: PID on heading error
+ Target Point       W_speed[10×2]     Throttle: From waypoint displacement
```

### Key Components
- **`core/config.py`**: All runtime parameters (`SimlingoQCar2Config`)
- **`core/qcar2_interface.py`**: QLabs connection, camera, vehicle control
- **`inference/main.py`**: Main entry point (`SimlingoQCar2Controller`)
- **`inference/control_converter.py`**: Waypoints → steering/throttle (PID)
- **`inference/state_estimator.py`**: Position/velocity tracking
- **`simlingo/simlingo_training/train.py`**: Training entry (Hydra + PyTorch Lightning)
- **`simlingo/simlingo_training/dataloader/dataset_driving.py`**: Data loading
- **`data_collection/collect_data.py`**: Expert data recording

### Data Flow
- **Training data**: `database/data/simlingo/routes_{training,validation}/{dataset}/{run}/TownQLabs/`
- **Format**: `rgb/NNNN.jpg` + `measurements/NNNN.json.gz` + `results.json.gz`
- **Must follow CARLA dataset structure** for dataloader compatibility

## Critical Implementation Details

### Velocity Calculation
Use **instantaneous single-frame delta**, NOT moving average:
```python
velocity = ||position_current - position_previous|| / dt
```
This matches training data collection and avoids train-inference mismatch.

### Speed from Waypoints
```python
desired_speed = ||W_speed[2] - W_speed[0]|| / 0.5s  # 2-waypoint span for stability
```

### Steering Sign Convention
QCar2 uses opposite convention from CARLA/Simlingo:
```python
turn_angle = -steering * (π/9)  # Sign inversion required
```

### Key Parameters (in `core/config.py`)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `carla_fps` | 4 Hz | Control loop frequency |
| `dt` | 0.25s | Timestep |
| `turn_kp/kd` | 12.0/3.5 | Steering PID gains |
| `qcar2_max_speed` | 4.0 m/s | Max velocity |
| `qcar2_max_steering` | π/9 rad | Max turn angle (~20°) |
| `target_point_lookahead` | 7.5m | Route navigation look-ahead |

### Training
- **Frozen**: Vision encoder (InternVL2-1B)
- **Trained**: Language model LoRA adapters only (~327M parameters)
- Uses DeepSpeed ZeRO-2, gradient accumulation (4 steps), 16-bit mixed precision

## Documentation

Detailed docs in `docs/`:
- `qcar2_control_loop.md` - Full control flow with equations
- `TRAINING_FLOW_EXPLAINED.md` - Training pipeline walkthrough
- `FINE_TUNING_GUIDE_QLABS.md` - QLabs fine-tuning setup
- `Simlingo_Model_Deep_Dive.md` - Model architecture details
