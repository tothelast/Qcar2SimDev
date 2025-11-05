#!/usr/bin/env python3
"""Move randomly selected datasets from training to validation."""

import os
import shutil
import random
from pathlib import Path

def main():
    training_dir = Path("database/data/simlingo/routes_training/qlabs")
    validation_dir = Path("database/data/simlingo/routes_validation/qlabs")
    
    # Create validation directory if it doesn't exist
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all Rep_kink_street and Rep_traffic_circle directories
    dirs = sorted([d for d in training_dir.iterdir() 
                               if d.is_dir() and d.name.startswith("Rep_roundabout_exit_")])
    
    
    if len(dirs) < 3:
        print(f"ERROR: Need at least 3 Rep_kink_street directories, found {len(dirs)}")
        return
    
    # Randomly select 3 from each
    random.seed(42)  # For reproducibility
    selected_reps = random.sample(dirs, 3)
    
    print("\nSelected for validation:")
    for d in selected_reps:
        print(f"  - {d.name}")
    
    
    # Move directories
    print("\nMoving directories...")
    for d in selected_reps:
        dest = validation_dir / d.name
        print(f"  Moving {d.name} -> {dest}")
        shutil.move(str(d), str(dest))
    
    print("\nDone!")
    print(f"\nValidation directory now contains:")
    validation_contents = sorted([d.name for d in validation_dir.iterdir() if d.is_dir()])
    for name in validation_contents:
        print(f"  - {name}")

if __name__ == "__main__":
    main()

