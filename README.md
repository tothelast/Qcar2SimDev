# QCar2SimDev Project

This project is designed for developing and testing the QCar2 vehicle in a simulation environment (QLabs/CARLA), including data collection, model training (SimLingo), and inference.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Note: PyTorch with CUDA support is recommended. See `requirements.txt` for specific installation commands.

2.  **Environment**:
    Ensure you have the QLabs and Quanser PAL libraries installed (usually system-level or provided by Quanser).

## Project Structure

### Core Components (`core/`)
Contains the fundamental libraries for interacting with the QLabs simulator and QCar2 vehicle.
-   **`qcar2_interface.py`**: Manages the connection to QLabs, spawns the QCar2 vehicle, handles camera image capture, sends control commands (steering/throttle), and visualizes trajectories.
-   **`config.py`**: Central configuration class (`SimlingoQCar2Config`) defining simulation parameters, vehicle settings, and paths.
-   **`scene_spawner.py` & `scene_loader.py`**: Responsible for loading scene definitions and spawning dynamic actors (traffic, pedestrians) into the simulation.
-   **`camera_processor.py`**: Handles image preprocessing for the model.

### Inference Engine (`inference/`)
Scripts for running the trained model in the simulator (Autonomous Mode).
-   **`main.py`**: The main entry point. It initializes the `SimlingoQCar2Controller`, loads the model, and runs the control loop. It handles sensor data ingestion, model inference, and control execution.
-   **`simlingo_model.py`**: A wrapper around the trained Simlingo model to handle inference calls.
-   **`control_converter.py`**: Converts high-level model outputs (waypoints/speed) into low-level QCar2 control signals (throttle/steering) using PID controllers.
-   **`commentary_window.py`**: A UI widget that displays the model's internal "thought process" or language output during driving.

### Model Training & Logic (`simlingo/`)
This directory contains the core machine learning components and agent implementation.
-   **`team_code/`**: Contains the custom implementation of the autonomous agent (`LingoAgent`) and associated logic.
    -   `agent_simlingo.py`: The main agent class that integrates with the CARLA leaderboard, handling sensor data processing, model inference, and control generation.
    -   `autopilot.py`: Rule-based expert driver used for data collection and benchmarking.
    -   `nav_planner.py`, `lateral_controller.py`, `longitudinal_controller.py`: Path planning and PID control logic for vehicle navigation.
    -   `data_agent.py`: Logic for managing data collection sessions.
    -   `transfuser_utils.py`: Utilities specific to the Transfuser-based model architecture.
-   **`simlingo_training/`**: Code related to training the Simlingo model.
    -   `train.py`: Main training script.
    -   `eval.py` & `eval_qlabs_offline_ade.py`: Evaluation scripts for computing metrics like ADE.
    -   `models/`: Model architecture definitions (e.g., `driving.py`, `language_model/`).
    -   `dataloader/`: Data loading logic (`datamodule.py`, `dataset_driving.py`) for processing training datasets.
    -   `config/`: Configuration files for training experiments.
-   **`pretrained/`**: Storage for pretrained model weights.
-   **`ade_metrics_explainer.md`**: Documentation on evaluation metrics (ADE - Average Displacement Error).

### Configuration (`config/`)
Defines the simulation environment and scenarios.
-   **`scenes/`**: YAML/JSON files defining specific driving scenarios (e.g., "roundabout_north", "urban_parking").
-   **`routes/`**: Predefined paths and waypoints for the ego vehicle.
-   **`actors/`**: Definitions for other dynamic actors in the scene.

### Data & Logs
-   **`debug_output/`**: A runtime directory where the system saves:
    -   **Trajectory Logs**: JSON files (`trajectory_log_*.json`) recording the vehicle's path, speed, and control inputs for every run.
    -   **Debug Images**: Screenshots and raw camera captures for verification.
    -   **CSV Logs**: Low-level control logs.
-   **`database/`**: Storage for large-scale datasets used for training and validation.
    -   `data/simlingo/`:
        -   `routes_training/qlabs/`: Contains subdirectories of recorded training data (replays) for various scenarios (e.g., `Rep_roundabout_navigation`, `Rep_full_circuit`).
        -   `routes_validation/qlabs/`: Contains subdirectories of recorded validation data for evaluating model performance.
-   **`models/`**: Contains model artifacts, checkpoints, and metadata.
    -   **`simlingo/`**: Stores model-specific configurations (`.hydra/`) and training checkpoints (`checkpoints/`).
    -   **`.cache/`**: Caching directory for Hugging Face model weights and other external assets.
    -   **`README.md`**: Information about the specific model version/license (e.g., Hugging Face links).
-   **`data_collection/`**: Tools for gathering expert driving data (teleoperation) to train the model.


## Key Scripts

-   **`analyze_logs.py`**: Analyze simulation logs.
-   **`analyze_expert_data.py`**: Analyze collected expert driving data.
-   **`data_collection/collect_data.py`**: Run this to collect new dataset.
-   **`inference/main.py`**: Run this to test the model in the simulator.
