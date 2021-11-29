import numpy as np


def to_one_hot(action: int, action_size: int) -> np.ndarray:
    """
    Create a one-hot-encoded vector of `action`.
    :param action: Index of the action. If outside of [0, action_size), a zero vector is return.
    :param action_size: Number of actions.
    :return: A one-hot-vector for `action` of shape (action_size,).
    """
    result = np.zeros([action_size])
    if 0 <= action < action_size:
        result[action] = 1.0

    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax.
    :param logits: Logits.
    :return: softmax(logits).
    """
    energies = np.exp(logits - np.max(logits))

    return energies / energies.sum()
