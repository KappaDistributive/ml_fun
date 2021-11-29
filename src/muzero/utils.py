import numpy as np


def to_one_hot(action: int, num_actions: int) -> np.ndarray:
    result = np.zeros([num_actions])
    if 0 <= action < num_actions:
        result[action] = 1.0

    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    energies = np.exp(logits - np.max(logits))

    return energies / energies.sum()
