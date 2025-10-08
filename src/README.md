# Simlingo-QCar2 Integration

This directory contains the complete integration of the Simlingo autonomous driving model with the Quanser QCar2 vehicle in QLabs simulation environment.

## Overview

The integration maintains **exact feature parity** with the original Simlingo implementation while adapting the hardware interface layer to work with QCar2 in QLabs instead of CARLA.

## Architecture

```
src/
├── main.py                    # Entry point: python src/main.py
├── config.py                  # All configuration parameters
├── qcar2_interface.py         # QCar2 QLabs interface wrapper
├── simlingo_model.py          # Simlingo model wrapper
├── camera_processor.py        # Camera image preprocessing
├── control_converter.py       # Convert Simlingo control to QCar2 control
├── state_estimator.py         # Track vehicle state (position, speed, heading)
└── route_manager.py           # Manage target points and route
```

## Features

### Exact Simlingo Feature Parity

✅ **Camera Input**:
- Resolution: 1024x512 (upscaled from QCar2's 820x410)
- FOV: 110 degrees
- ImageNet normalization (MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225])
- Camera intrinsics/extrinsics matrices

✅ **Model Input**:
- camera_images: [B, T, N, C, H, W] uint8 [0, 255]
- image_sizes: Tensor
- camera_intrinsics: [B, N, 3, 3] float32
- camera_extrinsics: [B, N, 4, 4] float32
- vehicle_speed: [B, S] float32 (m/s)
- target_point: [B, 2] float32 (GPS target in ego frame)
- prompt: LanguageLabel (with placeholders)
- prompt_inference: LanguageLabel

✅ **PID Controllers**:
- Turn PID: kp=3.25, ki=1.0, kd=1.0, n=20
- Speed PID: kp=1.75, ki=1.0, kd=2.0, n=20
- Lateral PID: kp=3.118357247806046, kd=1.3782508892109167, ki=0.6406067986034124
- All exact parameters from Simlingo

✅ **Control Pipeline**:
- Waypoint interpolation (0.1m spacing)
- Desired speed calculation from speed waypoints
- PID control law
- Kinematic bicycle model for speed prediction
- Brake logic (brake_speed=0.4, brake_ratio=1.1)

✅ **Configuration**:
- 20 Hz control loop
- All numerical parameters preserved
- Chain-of-Thought prompts

## Installation

### Prerequisites

1. **QLabs**: Quanser Interactive Labs must be installed and running
2. **Python**: Python 3.8 or higher
3. **Dependencies**:
   ```bash
   pip install numpy opencv-python torch torchvision scipy transformers
   ```

### Simlingo Model

The Simlingo model checkpoint is pre-configured to use the DeepSpeed ZeRO checkpoint at:

```python
self.model_checkpoint_path = "models/simlingo/checkpoints/epoch=013.ckpt"
self.hydra_config_path = "simlingo/.hydra/config.yaml"
```

These paths are already set correctly in `src/config.py` and point to the existing checkpoint in the repository.

## Usage

### Basic Usage

```bash
python src/main.py
```

The system will automatically load the Simlingo model from the configured checkpoint path.

### Configuration

Edit `src/config.py` to customize:

- **Route waypoints**: Define the path for QCar2 to follow
- **Camera settings**: Adjust FOV, resolution, position
- **PID parameters**: Fine-tune control (not recommended - use exact Simlingo values)
- **QLabs connection**: Change host, spawn location, etc.

## Configuration Details

### Camera Configuration

```python
# Target resolution (Simlingo)
self.camera_width = 1024
self.camera_height = 512

# QCar2 native resolution
self.qcar2_camera_width = 820
self.qcar2_camera_height = 410

# Camera FOV
self.camera_fov = 110  # degrees

# Camera position (CARLA coordinates)
self.camera_position = [-1.5, 0.0, 2.0]  # [x, y, z]
```

### PID Controller Parameters

All parameters are exact replicas from Simlingo:

```python
# Turn PID
self.turn_kp = 3.25
self.turn_ki = 1.0
self.turn_kd = 1.0
self.turn_n = 20

# Speed PID
self.speed_kp = 1.75
self.speed_ki = 1.0
self.speed_kd = 2.0
self.speed_n = 20

# Lateral PID (Advanced)
self.lateral_pid_kp = 3.118357247806046
self.lateral_pid_kd = 1.3782508892109167
self.lateral_pid_ki = 0.6406067986034124
```

### Kinematic Bicycle Model

```python
self.front_wheel_base = -0.090769015
self.rear_wheel_base = 1.4178275
self.steering_gain = 0.36848336
self.brake_acceleration = -4.952399
self.throttle_acceleration = 0.5633837
```

### Route Definition

Define waypoints in QLabs global coordinates:

```python
self.route_waypoints = [
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    # ... more waypoints
]
```

## Module Descriptions

### config.py
Contains all configuration parameters with exact Simlingo values. Provides methods for generating camera intrinsics/extrinsics and prompt templates.

### qcar2_interface.py
Handles all communication with QLabs:
- Connection management
- QCar2 spawning
- Camera image capture
- Control command sending
- State feedback

### camera_processor.py
Processes camera images:
- Resize from 820x410 to 1024x512 (bicubic interpolation)
- ImageNet normalization
- Tensor conversion
- Camera parameter generation

### state_estimator.py
Tracks vehicle state:
- Position tracking
- Velocity estimation (from position changes)
- Heading tracking
- Coordinate frame transformations (world ↔ ego)

### route_manager.py
Manages navigation:
- Route waypoint storage
- Target point selection (lookahead)
- World to ego frame conversion
- Progress tracking

### simlingo_model.py
Wraps Simlingo model:
- Model loading
- Tokenizer initialization
- LanguageLabel creation
- Model inference

### control_converter.py
Implements control pipeline:
- PID controllers (exact Simlingo implementation)
- Waypoint interpolation
- Control computation
- Simlingo → QCar2 control conversion
- Kinematic bicycle model

### main.py
Main control loop:
- Component initialization
- 20 Hz control loop
- State management
- Error handling
- Graceful shutdown

## Control Flow

1. **Initialization**:
   - Connect to QLabs
   - Spawn QCar2
   - Load Simlingo model
   - Initialize all components

2. **Control Loop (20 Hz)**:
   - Capture camera image
   - Process image (resize, normalize)
   - Get vehicle state (position, rotation)
   - Estimate velocity
   - Get target point from route
   - Create prompt with speed and target
   - Run Simlingo model inference
   - Extract speed and route waypoints
   - Compute PID control (steer, throttle, brake)
   - Convert to QCar2 control (forward, turn)
   - Send control command
   - Update state

3. **Shutdown**:
   - Stop vehicle
   - Close QLabs connection
   - Cleanup

## Coordinate Systems

### CARLA (Simlingo)
- x: forward
- y: right
- z: up
- Right-handed

### QLabs (QCar2)
- Similar to CARLA
- Body frame at ground level between axles
- Distances are 10x physical QCar size

### Ego Frame
- x: forward
- y: left (note: opposite of world y)
- Origin at vehicle center

## Troubleshooting

### Model Loading Issues

If the model fails to load:
1. Check the checkpoint path in `config.py`
2. Ensure the checkpoint file exists
3. Verify PyTorch version compatibility

The system will continue without the model for testing purposes, using default waypoints.

### QLabs Connection Issues

If QLabs connection fails:
1. Ensure QLabs is running
2. Check the host address in `config.py` (default: "localhost")
3. Verify no firewall blocking

### Camera Image Issues

If camera images are not captured:
1. Check QCar2 is spawned successfully
2. Verify camera constant (CAMERA_CSI_FRONT = 3)
3. Check QLabs rendering is enabled

### Control Issues

If vehicle doesn't move properly:
1. Check velocity estimation (printed in status)
2. Verify route waypoints are defined
3. Check PID parameters (should be exact Simlingo values)
4. Monitor stuck detection and recovery

## Performance

- **Control Frequency**: 20 Hz (50ms per iteration)
- **Model Inference**: Depends on GPU (typically 10-30ms)
- **Image Processing**: ~5ms
- **State Estimation**: <1ms
- **PID Control**: <1ms

If control loop runs slow, check:
1. GPU availability for model inference
2. QLabs rendering performance
3. Network latency (if QLabs is remote)

## Limitations

1. **Camera Resolution**: QCar2 CSI cameras are 820x410, upscaled to 1024x512. This may affect model performance compared to native 1024x512 input.

2. **Coordinate System**: Minor differences between CARLA and QLabs coordinate systems may require calibration.

3. **Model Checkpoint**: Requires a trained Simlingo model checkpoint. The system can run without it for testing.

## Future Enhancements

Potential improvements (not implemented to maintain exact feature parity):

- Multi-camera support (left, right, rear)
- LIDAR integration
- Depth camera utilization
- Dynamic route planning
- Obstacle detection and avoidance
- Real-time visualization overlay

## References

- Simlingo Paper: https://arxiv.org/pdf/2503.09594
- QLabs Documentation: https://qlabs.quanserdocs.com/en/latest/Objects/qcar2_library.html
- QCar2 Python SDK: `python/qvl/qcar2.py`

## License

This integration follows the same license as the Simlingo project.

## Contact

For issues or questions, refer to the Simlingo and QLabs documentation.

