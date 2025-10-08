# SimLingo Directory Structure for Integration

This document outlines the key folders from the original SimLingo repository that are essential for integrating the model into a new simulation environment. The goal is to replicate the model's architecture and functionality for fine-tuning and deployment.

## 1. simlingo_training/
**Purpose: Model Architecture and Training Pipeline**

This is the most critical directory as it contains the complete source code for the SimLingo Vision-Language-Action (VLA) model.

**Core Model Definition:** Within this folder, you will find the Python modules that define the neural network architecture. This includes the integration of the vision encoder, the language model, and the specific output heads that produce waypoints, speed predictions, and language tokens. To replicate the model, these files are non-negotiable.

**Training and Fine-Tuning Logic:** This directory houses the main training script (train.py), loss functions, and the optimization loop. When you collect data from your QCar 2 environment, you will use these scripts to fine-tune the downloaded Hugging Face model on your new dataset.

**Data Handling:** It contains the PyTorch Dataset and DataLoader implementations responsible for preprocessing images and language prompts for the model. You will need to adapt these data loaders to read the data format from your new simulator, but the underlying preprocessing logic (e.g., image tiling, text tokenization) should be preserved to ensure consistency with the original model's training.

**Configuration:** The config/ subdirectory holds the YAML files defining all hyperparameters and model settings. You will modify these files to point to your new dataset paths and adjust training parameters for fine-tuning.

## 2. team_code/
**Purpose: Inference Agent and Controller Logic**

This folder contains the self-contained agent code used for running the trained model in a closed-loop setting. While it was originally built for the CARLA Leaderboard, it serves as a perfect template for your new implementation.

**Inference Pipeline:** The agent script (e.g., simlingo_agent.py) provides the blueprint for the perception-action loop. It details the necessary steps for runtime execution: loading the trained model, capturing sensor data, preprocessing the inputs exactly as was done during training, running the model forward pass, and parsing the output tensors into waypoints and a target speed.

**Model-to-Control Adapter:** This is where the model's abstract outputs are translated into concrete actions. You will find the implementation of the low-level PID controller that takes the predicted waypoints and speed as input and calculates the final steering, throttle, and brake commands. You can reuse this controller logic directly.

**Simulator Integration Point:** Your primary task will be to adapt the main run_step method in the agent. You will replace the CARLA-specific API calls for receiving sensor data and sending vehicle commands with the corresponding API calls for your QCar 2 simulator.

## 3. data/
**Purpose: Language Augmentation Assets**

While most of the data folder in the original repository contains CARLA-specific route files, it also contains a critical asset required for training: the augmented_templates/ subdirectory.

**Training Consistency:** The SimLingo model was trained using a specific set of augmented language templates for its VQA and commentary tasks. To successfully fine-tune the model and maintain its linguistic capabilities, the training pipeline will need access to these files.

**Language Task Replication:** These templates are loaded by the data loaders inside simlingo_training to increase the diversity of the language prompts. Including this folder ensures that the fine-tuning process on your new dataset is consistent with the original training methodology, which is crucial for achieving comparable performance.