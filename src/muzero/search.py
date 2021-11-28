import itertools
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from src.muzero.model import AbstractMuZeroModel


def softmax(logits: np.ndarray) -> np.ndarray:
    energies = np.exp(logits - np.max(logits))
    return energies / energies.sum()


def get_actions(
    num_actions: int, action_size: int
) -> Tuple[List[Tuple[int, ...]], List[tf.Tensor]]:
    def to_one_hot(action: int, num_actions: int) -> np.ndarray:
        result = np.zeros([num_actions])
        result[action] = 1.0
        return result

    all_action_sequences = list(
        itertools.product(list(range(action_size)), repeat=num_actions)
    )
    action_sequence_tensors: List[tf.Tensor] = []
    for action_step in range(num_actions):
        action_sequence_tensors.append(
            tf.stack(
                [
                    to_one_hot(action_sequence[action_step], action_size)
                    for action_sequence in all_action_sequences
                ],
                axis=0,
            )
        )

    return all_action_sequences, action_sequence_tensors


action_space_cache: Dict[
    Tuple[int, int], Tuple[List[Tuple[int, ...]], List[tf.Tensor]]
] = {}


def naive_search(model: AbstractMuZeroModel, initial_observation: tf.Tensor):
    num_actions = model.num_actions
    action_size = model.action_size

    if (num_actions, action_size) not in action_space_cache:
        action_space_cache[(num_actions, action_size)] = get_actions(
            num_actions, action_size
        )

    all_action_sequences, all_encoded_action_sequences = action_space_cache[
        (num_actions, action_size)
    ]
    initial_observations = tf.repeat(
        tf.reshape(initial_observation, shape=(1, -1)),
        len(all_action_sequences),
        axis=0,
    )  # shape (num_actions, observation_size)

    output = model.mu_function(initial_observations, all_encoded_action_sequences)
    final_values = tf.squeeze(output[2 * (num_actions + 1) - 1]).numpy()
    action_value_pairs = [
        (all_action_sequences[index], final_values[index])
        for index in range(final_values.shape[0])
    ]

    action_values = [0.0] * action_size
    for action_sequence, value in action_value_pairs:
        action_values[action_sequence[0]] += value

    action_values = np.array(action_values).astype(np.float64) / num_actions
    policy = softmax(action_values)

    return policy