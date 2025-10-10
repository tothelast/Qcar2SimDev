# QCar2 SimLingo Integration

Adapting SimLingo Vision-Language-Action (VLA) model from CARLA simulator to QCar2 in QLabs simulation. GPU-accelerated inference on RTX 5070 (CUDA 12.8).

## What is This?

**SimLingo** is a vision-language model for autonomous driving, originally trained on CARLA simulator data. It uses InternVL2-1B (vision-language backbone) to predict driving waypoints from camera images, target points, and optional natural language instructions.

**This project** adapts SimLingo to run on Quanser's QCar2 platform in QLabs simulation, replacing CARLA-specific code with QCar2 equivalents while preserving the core AI model.

## Key Features

- **Real SimLingo Model**: InternVL2-1B backbone with LoRA adapters (epoch=013.ckpt)
- **Route Following**: Predefined waypoint routes with lookahead algorithm
- **Control System**: Lateral PID + Longitudinal Linear Regression + Kinematic Bicycle Model (exact SimLingo parameters)
- **HLC Support**: Natural language instructions via interactive commentary window
- **Full Pipeline**: Camera → Preprocessing → Model Inference → Waypoint Prediction → Control Conversion → QCar2

## Project Structure

```
Qcar2SimDev/
├── src/
│   ├── main.py                  # Main control loop
│   ├── simlingo_model.py        # Model wrapper (loads checkpoint, runs inference)
│   ├── camera_processor.py      # Camera preprocessing (JPEG compression, patching)
│   ├── route_manager.py         # Route following with lookahead algorithm
│   ├── control_converter.py     # PID controllers + bicycle model
│   ├── qcar2_interface.py       # QLabs/QCar2 interface
│   ├── commentary_window.py     # GUI for model commentary + HLC input
│   ├── config.py                # Configuration (PID params, routes, model paths)
│   └── state_estimator.py       # State estimation and filtering
├── simlingo/
│   ├── simlingo_training/       # SimLingo model code (from original repo)
│   └── team_code/               # Reference implementations
├── models/simlingo/
│   └── checkpoints/epoch=013.ckpt/  # Trained model weights
├── pretrained/InternVL2-1B/     # Base vision-language model
└── docs/                        # Technical documentation
```

## Requirements

**System:**
- Ubuntu 24.04, Python 3.12
- NVIDIA GPU with CUDA 12.8 (tested on RTX 5070)
- QLabs with QCar2 environment

**Python Dependencies:**
- PyTorch 2.7+ (CUDA 12.8), transformers, pytorch-lightning, peft
- opencv-python, pillow, numpy
- hydra-core, omegaconf, einops, timm
- Quanser libraries: `qvl`, `pal`

## Setup

### 1) Create and activate virtual environment
```bash
# Create venv with system packages (required for Quanser libraries)
python3 -m venv --system-site-packages simlingo_env
source simlingo_env/bin/activate
```

### 2) Install PyTorch with CUDA support
```bash
# Install PyTorch with CUDA 12.8 support first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3) Install remaining requirements
```bash
# Install all other dependencies
pip install -r requirements.txt
```

### 4) Verify GPU detection
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 5) Model Assets

**SimLingo Checkpoint** (required):
- `models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- `models/simlingo/.hydra/config.yaml`

**InternVL2-1B Base Model** (auto-downloaded from HuggingFace):
- `pretrained/InternVL2-1B/` (~2GB)

**SimLingo Code** (required):
- `simlingo/simlingo_training/` - Model architectures, utilities
- `simlingo/team_code/` - Reference implementations

## Usage

**1. Start QLabs** with QCar2 environment loaded

**2. Run the integration:**
```bash
source simlingo_env/bin/activate
python src/main.py
```

**CLI Options:**
- `--config` - Path to custom config file (optional)
- `--spawn-obstacles` - Spawn obstacle vehicles along the route (optional)

**Note:** Route waypoints are defined in `src/config.py` (hardcoded). Control loop runs until route completion (within 2m of final waypoint) or manual stop (Ctrl+C).

**3. Commentary Window** opens automatically:
- **Left panel**: Model's natural language commentary
- **Right panel**: Current speed + HLC input field
- **To use HLC**: Type instruction (e.g., "Turn left at intersection") and press Enter
- **To clear HLC**: Delete text and press Enter (returns to default mode)

## How It Works

### Complete Pipeline

```
1. Route Manager
   ↓ Finds target waypoints using lookahead algorithm
   ↓ Converts to ego frame (vehicle-centric coordinates)

2. Camera Processing
   ↓ QCar2 camera → JPEG compression → Dynamic preprocessing
   ↓ Splits into 2 patches of 448x448 patches 

3. SimLingo Model Inference
   ↓ Input: camera patches + target points + speed + optional HLC
   ↓ Prompt (default): "Current speed: X m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"
   ↓ Prompt (with HLC): "<INSTRUCTION_FOLLOWING> Current speed: X m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. {instruction}"
   ↓ Output: route_waypoints (predicted path) + speed_waypoints (predicted speeds) + commentary

4. Control Conversion
   ↓ Lateral PID Controller: route_waypoints → steering
   ↓ Longitudinal Linear Regression: speed_waypoints → throttle/brake
   ↓ Kinematic Bicycle Model: converts to QCar2 format

5. QCar2 Execution
   ↓ forward_velocity + turn_angle → vehicle motion
```

### Key Components

**Route Manager** (`route_manager.py`):
- Uses predefined waypoint routes from `config.py` (hardcoded in global coordinates)
- Lookahead algorithm: finds target points at fixed distance ahead
- World-to-ego transformation: converts global waypoints to vehicle frame

**SimLingo Model** (`simlingo_model.py`):
- InternVL2-1B vision-language model with LoRA adapters
- Processes camera + target points + optional HLC
- Predicts waypoints and speeds for next 2 seconds
- Generates natural language commentary

**Control Converter** (`control_converter.py`):
- **Lateral PID**: Kp=3.25, Ki=1.0, Kd=1.0 (exact SimLingo params)
- **Longitudinal Linear Regression**: 7-parameter model (exact SimLingo params)
- **Kinematic Bicycle Model**: Converts to QCar2's forward_velocity + turn_angle

**Camera Processor** (`camera_processor.py`):
- JPEG compression (matches CARLA training data)
- Dynamic preprocessing: splits image into 448x448 patches
- ImageNet normalization

## High-Level Commands (HLC)

This project supports natural language instructions via the commentary window:

**Example Instructions:**
- "Turn left at the intersection"
- "Slow down and prepare to stop"
- "Change lanes to the right"
- "Follow the road carefully"

**How it works:**
- Type instruction in commentary window → Press Enter
- Prompt changes to `<INSTRUCTION_FOLLOWING>` mode
- Model receives instruction along with camera and target points
- Model predicts waypoints that follow both the instruction and the route

**Training:** SimLingo was trained on the Dreamer dataset with instruction following, so it can handle natural language commands while maintaining safe driving behavior.

## Documentation

See `docs/` folder for detailed technical documentation:
- `SIMLINGO_FILES_USED.md` - Which SimLingo files we use and why
- `CUSTOM_IMPLEMENTATION.md` - Our QCar2-specific implementations
- `ROUTE_MANAGER_EXPLAINED.md` - How route following works
- `CAMERA_PROCESSOR_VS_INTERNVL2_UTILS.md` - Camera preprocessing details
- `QCAR2_CONTROL_EXPLAINED.md` - Control system implementation

