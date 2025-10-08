---
type: "always_apply"
---

# How to Run and Test This Project

## 1. Setup the Environment
Before running the code, you must activate the Python virtual environment named `simlingo_env`.

**Command:**
```bash
source simlingo_env/bin/activate
```

## 2. Execute the Main Script
The primary entry point for the application is src/main.py. Run this script from the root of the project directory after activating the environment.

**Command:**
```bash
python src/main.py
```

## Optional Arguments
You can control the simulation's duration (in seconds) and hz (frequency in Hertz).
--duration: Sets the total run time in seconds. Default is 30.
--hz: Sets the operating frequency in Hz. Default is 5.

Example with arguments: To run the simulation for 60 seconds at a frequency of 10 Hz, use the following command:

**Command:**
```bash
python src/main.py --duration 60 --hz 10
```

