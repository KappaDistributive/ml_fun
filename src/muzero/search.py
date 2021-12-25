import itertools
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.muzero.model import AbstractMuZeroModel
from src.muzero.utils import softmax, to_one_hot


def get_actions(
    lookahead_range: int, action_size: int
) -> Tuple[List[Tuple[int, ...]], List[tf.Tensor]]:
    """
    Create all possible action combinations.
    :param lookahead_range: Number of actions (i.e. steps) in each entry.
    :param action_size: Number of possible actions per step.
    :return: All possible combinations of `lookahead_range`-sized action-sequences.
    """
    all_action_sequences = list(
        itertools.product(list(range(action_size)), repeat=lookahead_range)
    )
    action_sequence_tensors: List[tf.Tensor] = []
    for action_step in range(lookahead_range):
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


def naive_search(
    model: AbstractMuZeroModel, initial_observation: tf.Tensor
) -> Tuple[np.ndarray, None]:
    """
    A naive search algorithm.
    :param model: MuZero model.
    :param initial_observation: Initial observation.
    :return: Policy, i.e. a probability distribution over all actions.
    """
    lookahead_range = model.lookahead_range
    assert (
        len(model.action_shape) == 1
    ), f"Lacking support for action_shape : {model.action_shape}"
    action_size = model.action_shape[0]

    if (lookahead_range, action_size) not in action_space_cache:
        action_space_cache[(lookahead_range, action_size)] = get_actions(
            lookahead_range, action_size
        )

    all_action_sequences, all_encoded_action_sequences = action_space_cache[
        (lookahead_range, action_size)
    ]
    initial_observations = tf.repeat(
        tf.expand_dims(initial_observation, axis=0), len(all_action_sequences), axis=0,
    )  # shape (lookahead_range, observation_size)

    output = model.mu_function(
        observation=initial_observations, actions=all_encoded_action_sequences
    )
    final_values = tf.squeeze(output[2 * (lookahead_range + 1) - 1]).numpy()
    action_value_pairs = [
        (all_action_sequences[index], final_values[index])
        for index in range(final_values.shape[0])
    ]

    action_values = [0.0] * action_size
    for action_sequence, value in action_value_pairs:
        action_values[action_sequence[0]] += value

    action_values = np.array(action_values).astype(np.float64) / lookahead_range
    policy = softmax(action_values)

    return policy, None


class Node:
    def __init__(self, internal_state: Optional[tf.Tensor] = None, to_play: int = -1):
        """
        :param internal_state: MuZero's current internal state of a game.
        :param to_play: The player who is to act on `internal_state`. Should be either 1 or -1.
        """
        self.internal_state = internal_state
        self.to_play = to_play
        self.num_visits = 0
        self.total_value = 0.0
        self.children: Dict[int, Node] = {}  # maps actions to children
        self.reward = 0.0

    def is_expanded(self) -> bool:
        """
        :return: True if this node has aldready been expanded.
        """
        return bool(self.children)

    def value(self) -> float:
        """
        :return: The current value of this node.
        """
        if self.num_visits == 0:
            return 0.0
        return self.total_value / self.num_visits


def ucb_score(
    parent: Node, child: Node, exploration_parameter: float = math.sqrt(2.0)
) -> float:
    """
    UCB score. See https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
    :param parent: Parent node.
    :param child: Child node.
    :param exploration_parameter: The exploration parameter. A higher value encourages more exploration.
    :return: UCB score of `child`.
    """
    if child.num_visits == 0:
        return float("inf")

    assert (
        parent.num_visits > 0
    ), "Encountered a parent that has never been visited even though its child has been visited."

    return child.value() + exploration_parameter * math.sqrt(
        math.log(parent.num_visits) / child.num_visits
    )


def select_child(node: Node, exploration_parameter: float = 1.0) -> Tuple[int, Node]:
    """
    Select the best child according to their UCB scores.
    :param node: The parent node.
    :param exploration_parameter: The exploration parameter. A higher value encourages more exploration.
    :return: [action_leading_to_best_child, best_child].
    """
    candidates = [
        (action, child, ucb_score(node, child, exploration_parameter))
        for action, child in node.children.items()
    ]
    max_score = max([candidate[2] for candidate in candidates])

    action, child, _ = random.choice(
        [candidate for candidate in candidates if candidate[2] == max_score]
    )

    return action, child


def mcts(
    model: AbstractMuZeroModel,
    initial_observation: tf.Tensor,
    num_simulations: int,
    discount_factor: float = 1.0,
    ignore_to_play: bool = False,
    softmax_temperature: float = 1.0,
) -> Tuple[np.ndarray, Node]:
    """
    Policy-estimation of a MuZero-model via Monte Carlo Tree Search.
    :param model: MuZero-model.
    :param initial_observation: Initial observation.
    :param num_simulations: The number of simulation steps to be performed.
    :param discount_factor: The discount factor applied to future reward.
    :param ignore_to_play: If true, don't distinguish between player 1/-1 when considering reward.
    :param softmax_temperature: Softmax temperature.
    :return: [policy, root].
    """
    root = Node(
        tf.squeeze(
            model.representation_function(tf.expand_dims(initial_observation, axis=0)),
            axis=0,
        )
    )
    policy, _ = model.prediction_function(tf.expand_dims(root.internal_state, axis=0))
    policy = tf.squeeze(policy).numpy()

    action_size = policy.shape[0]

    for action in range(action_size):
        root.children[action] = Node(policy[action], to_play=-root.to_play)

    for simulation_step in range(num_simulations):
        actions = []
        node = root
        search_path = [node]

        # traverse the tree down to a non-expanded node
        while node.is_expanded():
            action, node = select_child(node)
            actions.append(action)
            search_path.append(node)

        # update the non-expanded node
        parent = search_path[-2]
        reward, internal_state = model.dynamics_function(
            tf.expand_dims(parent.internal_state, axis=0),
            tf.expand_dims(
                tf.convert_to_tensor(to_one_hot(actions[-1], action_size)), axis=0
            ),
        )
        node.reward = float(reward.numpy())
        node.internal_state = tf.squeeze(internal_state, axis=0)
        policy, value = model.prediction_function(
            tf.expand_dims(node.internal_state, axis=0)
        )
        policy = tf.squeeze(policy).numpy()
        value = float(tf.squeeze(value).numpy())

        for action in range(action_size):
            node.children[action] = Node(policy[action], to_play=-node.to_play)

        # backpropagation
        for node in reversed(search_path):
            node.total_value += (
                value if (ignore_to_play or root.to_play == node.to_play) else -value
            )
            node.num_visits += 1
            value = node.reward + discount_factor * value

    logits = []
    for action, child in root.children.items():
        logits.append(child.num_visits)

    policy = softmax(np.array(logits).astype(np.float64) / softmax_temperature)

    return policy, root
