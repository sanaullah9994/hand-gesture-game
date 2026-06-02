import numpy as np


class Smoother:
    # EMA smoother to reduce jitter from hand tracking
    # alpha: higher = more responsive, lower = smoother
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        elif isinstance(new_value, (tuple, list, np.ndarray)):
            self.value = tuple(
                self.alpha * nv + (1 - self.alpha) * ov
                for nv, ov in zip(new_value, self.value)
            )
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def angle_between(v1, v2):
    # angle in degrees between two 2D vectors
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))))
