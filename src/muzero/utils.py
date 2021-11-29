import numpy as np


def to_one_hot(action: int, lookahead_range: int) -> np.ndarray:
    result = np.zeros([lookahead_range])
    if 0 <= action < lookahead_range:
        result[action] = 1.0

    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    energies = np.exp(logits - np.max(logits))

    return energies / energies.sum()
