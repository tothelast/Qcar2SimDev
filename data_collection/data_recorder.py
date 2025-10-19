#!/usr/bin/env python3
"""
Data Recorder Module

Placeholder for Phase 2 (Expert Data Collection).

This module will handle:
- Recording camera images from QCar2
- Recording vehicle state (position, heading, speed)
- Computing ground-truth waypoints in ego-frame
- Computing route path waypoints
- Saving data in SimLingo format
"""


class DataRecorder:
    """
    Records expert driving demonstrations for SimLingo fine-tuning.
    
    This is a placeholder class for Phase 2 implementation.
    """
    
    def __init__(self, config):
        """
        Initialize data recorder.
        
        Args:
            config: SimlingoQCar2Config instance
        """
        self.config = config
        self.recording = False
        self.samples = []
    
    def start_recording(self):
        """Start recording data."""
        self.recording = True
        print("Data recording started (placeholder)")
    
    def stop_recording(self):
        """Stop recording data."""
        self.recording = False
        print("Data recording stopped (placeholder)")
    
    def record_sample(self, camera_image, vehicle_state):
        """
        Record a single data sample.
        
        Args:
            camera_image: Camera image from QCar2
            vehicle_state: Vehicle state dict with position, heading, speed
        """
        if not self.recording:
            return
        
        # Placeholder - will be implemented in Phase 2
        pass
    
    def save_dataset(self, output_path):
        """
        Save recorded dataset to disk.
        
        Args:
            output_path: Path to save dataset
        """
        # Placeholder - will be implemented in Phase 2
        print(f"Dataset save (placeholder): {output_path}")

