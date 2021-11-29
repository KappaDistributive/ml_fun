import numpy as np


def to_one_hot(action: int, num_actions: int) -> np.ndarray:
    result = np.zeros([num_actions])
    result[action] = 1.0
    return result
