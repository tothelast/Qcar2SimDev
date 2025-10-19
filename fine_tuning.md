# SimLingo Fine-Tuning Roadmap for QLabs QCar2

## Phase 1: Infrastructure Setup & Validation

### 1.1 Environment Verification
- [ ] Verify CUDA/PyTorch installation matches training requirements (ref: `requirements.txt`)
- [ ] Test SimLingo training code runs without errors (`simlingo/simlingo_training/train.py`)
- [ ] Verify QLabs connection and QCar2 spawn (`collect_data.py`)
- [ ] Confirm camera feed resolution (820x410 → 1024x512) matches spec (`src/camera_processor.py`)

### 1.2 Teleop Control Implementation
- [ ] Implement keyboard/joystick teleop control for QCar2 in QLabs
- [ ] Add recording trigger (start/stop data collection)
- [ ] Test teleop responsiveness and control smoothness
- [ ] Verify teleop can navigate predefined routes (`src/config.py` route_waypoints)

### 1.3 Data Format Validation
- [ ] Create single sample validator against `docs/simlingo_training_data_format.md`
- [ ] Verify camera intrinsics calculation (160° FOV, 1024x512) (`src/config.py`)
- [ ] Verify camera extrinsics (QCar2 front camera: [+1.83m, 0.0, +1.10m])
- [ ] Test ego-frame coordinate transformation (world → ego)

## Phase 2: Expert Data Collection

### 2.1 Single Route Collection
- [ ] Collect 50 expert demonstrations on predefined route (Node 13→19→17→20→22)
- [ ] Record: camera images, speed, position, heading, timestamp
- [ ] Compute ground-truth waypoints (11 points, 0.2s spacing, ego-frame)
- [ ] Compute route path (20 points, 1m spacing, ego-frame)
- [ ] Save in SimLingo format (DrivingInput + DrivingLabel)

### 2.2 Diverse Route Collection
- [ ] Design 5 diverse routes covering: straight, curves, roundabout, intersections
- [ ] Collect 30 demonstrations per route (150 total)
- [ ] Ensure speed variation (1.5-3.0 m/s)
- [ ] Include challenging scenarios: tight turns, pedestrian crossings

### 2.3 Data Quality Assurance
- [ ] Visualize collected trajectories vs. route waypoints
- [ ] Check for data corruption (missing frames, invalid coordinates)
- [ ] Verify waypoint spacing consistency (0.2s temporal, 1m spatial)
- [ ] Remove low-quality demonstrations (collisions, off-road)

## Phase 3: Dataset Preparation

### 3.1 Dataset Loader Implementation
- [ ] Create QLabs dataset class inheriting from `simlingo_training/dataloader/dataset_base.py`
- [ ] Implement `__getitem__` returning (DrivingInput, DrivingLabel)
- [ ] Add data augmentation: lateral offset (±0.5m), yaw rotation (±5°)
- [ ] Test dataloader with batch_size=4, verify shapes match spec

### 3.2 Training Configuration
- [ ] Create Hydra config for QLabs dataset (`simlingo/simlingo_training/config/`)
- [ ] Set learning rate: 1e-5 (LoRA fine-tuning)
- [ ] Set batch size: 4-8 (based on GPU memory)
- [ ] Configure LoRA parameters: r=16, alpha=32, target_modules=[q_proj, v_proj]
- [ ] Set training epochs: 10-20 (monitor validation loss)

### 3.3 Validation Split
- [ ] Split dataset: 80% train, 20% validation
- [ ] Ensure validation routes differ from training routes
- [ ] Create validation metrics: waypoint L2 error, collision rate

## Phase 4: Fine-Tuning Execution

### 4.1 Initial Training Run
- [ ] Run 1 epoch on small subset (10 samples) to verify pipeline
- [ ] Monitor GPU memory usage and training speed
- [ ] Verify loss computation (language loss + waypoint loss)
- [ ] Check checkpoint saving (`models/simlingo/checkpoints/`)

### 4.2 Full Training
- [ ] Train for 10 epochs on full dataset
- [ ] Monitor training/validation loss curves (WandB/TensorBoard)
- [ ] Save checkpoints every 2 epochs
- [ ] Early stopping if validation loss plateaus (patience=3)

### 4.3 Hyperparameter Tuning
- [ ] Experiment with learning rates: [5e-6, 1e-5, 2e-5]
- [ ] Test different LoRA ranks: [8, 16, 32]
- [ ] Adjust data augmentation strength
- [ ] Select best checkpoint based on validation metrics

## Phase 5: Evaluation & Iteration

### 5.1 Closed-Loop Testing
- [ ] Test fine-tuned model on validation routes in QLabs
- [ ] Measure: route completion rate, average speed, smoothness
- [ ] Compare against pre-trained model baseline
- [ ] Record failure cases (off-road, collisions, stuck)

### 5.2 Generalization Testing
- [ ] Create 3 completely new routes (unseen during training)
- [ ] Test model performance on new routes
- [ ] Analyze failure modes: tight turns, pedestrians, parked cars
- [ ] Identify data gaps (scenarios missing from training set)

### 5.3 Iterative Improvement
- [ ] Collect additional data for failure scenarios
- [ ] Re-train with augmented dataset
- [ ] Test commentary/Q&A/Dreamer modes on QLabs scenes
- [ ] Document performance improvements and limitations

## Phase 6: Production Readiness

### 6.1 Model Optimization
- [ ] Quantize model to INT8 for faster inference (optional)
- [ ] Benchmark inference speed (target: <100ms per frame)
- [ ] Test on different GPU configurations

### 6.2 Documentation
- [ ] Document training procedure and hyperparameters
- [ ] Create model card with performance metrics
- [ ] Write deployment guide for QLabs integration

### 6.3 Final Validation
- [ ] Run 100 episodes on diverse routes
- [ ] Compute aggregate metrics: success rate, average speed, safety violations
- [ ] Compare against human teleop baseline
- [ ] Release fine-tuned checkpoint

---

**Key Files Referenced:**
- Data collection: `collect_data.py`, `src/visualize_map.py`
- Data format: `docs/DATA_COLLECTION.md`, `docs/simlingo_training_data_format.md`
- Training: `simlingo/simlingo_training/train.py`, `simlingo/simlingo_training/dataloader/`
- Configuration: `src/config.py`, `simlingo/simlingo_training/config/`
- Model: `src/simlingo_model.py`, `models/simlingo/checkpoints/`

