"""The expensive objective, served by server.py. Replace with your own function."""
import time

import numpy as np


def objective(x1, x2):
    time.sleep(0.2)  # simulate an expensive evaluation
    return -abs(np.sin(x1) * np.cos(x2) * np.exp(abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))
