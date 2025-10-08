"""
PID Controller implementation matching official SimLingo.

Reference: https://github.com/RenzKa/simlingo/blob/main/simlingo_training/utils/transfuser_utils.py
Lines: 330-348

This is a windowed PID controller that uses a deque to store the last n errors
for computing the integral and derivative terms.
"""

from collections import deque
import numpy as np


class PIDController:
    """
    PID controller that converts errors to control outputs.
    
    Uses a windowed approach where only the last n errors are stored
    for computing integral and derivative terms.
    
    Reference: SimLingo's transfuser_utils.py, lines 330-348
    """

    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, n=20):
        """
        Initialize PID controller.
        
        Args:
            k_p: Proportional gain
            k_i: Integral gain
            k_d: Derivative gain
            n: Window size for error history (default: 20)
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        """
        Compute PID output for given error.
        
        Args:
            error: Current error value
            
        Returns:
            PID control output
        """
        self.window.append(error)

        if len(self.window) >= 2:
            integral = np.mean(self.window)
            derivative = self.window[-1] - self.window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self.k_p * error + self.k_i * integral + self.k_d * derivative
    
    def reset(self):
        """Reset the error history window."""
        n = self.window.maxlen
        self.window = deque([0 for _ in range(n)], maxlen=n)

