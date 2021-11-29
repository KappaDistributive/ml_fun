from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf

from src.muzero.game import ReplayBuffer, play_game
from src.muzero.model import DenseMuZeroModel
from src.muzero.utils import to_one_hot


def prepare_batch(
    batch: List[Tuple[np.ndarray, List[int], List[Tuple[float, float, np.ndarray]]]],
    action_size: int,
    num_actions: int,
):
    observations_batch, actions_batch, labels_batch = [], [], []

    for entry in batch:
        assert isinstance(entry, tuple)
        assert len(entry) == 3
        initial_observation, actions, targets = entry

        while len(actions) < num_actions:
            actions.append(
                -1
            )  # gets encoded as the zero vector during one-hot-encoding

        while len(targets) < num_actions + 1:
            targets.append((0.0, 0.0, np.array([1.0 / action_size] * action_size)))

        # print("Initial observation", initial_observation)
        # print("Actions", actions)
        # print("Targets", targets)

        observations_batch.append(initial_observation)
        actions_batch.append(
            [to_one_hot(action, action_size) for action in actions[:num_actions]]
        )

        # mu function \mu_{\theta}: (o^{0}, a^{1}, ..., a^{K}) |---> (p^{0}, ..., p^{K}, v^{0}, ..., v^{K}, r^{1}, ..., r^{K})
        values = [np.array([target[0]]) for target in targets]
        rewards = [np.array([target[1]]) for target in targets[1:]]
        policies = [target[2] for target in targets]

        # print(len(values), len(rewards), len(policies))
        # print(len(policies + values + rewards))

        labels_batch.append(policies + values + rewards)

    return (
        tf.stack(observations_batch, axis=0),
        [
            tf.stack([action[index] for action in actions_batch], axis=0)
            for index in range(num_actions)
        ],
        [
            tf.stack([label[index] for label in labels_batch], axis=0)
            for index in range(3 * num_actions + 2)
        ],
    )


if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    observation = environment.reset()
    # print(environment.action_space)
    # print(observation, )
    num_actions = 5
    observation_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n
    state_size = 32
    batch_size = 16
    model = DenseMuZeroModel(
        num_actions=num_actions,
        observation_size=observation_size,
        action_size=action_size,
        state_size=state_size,
        hidden_layer_sizes=[32, 32],
    )
    replay_buffer = ReplayBuffer(
        buffer_size=128, batch_size=batch_size, num_actions=num_actions
    )

    print(model.mu_model.summary())

    for sample_step in range(100):
        game = play_game(environment, model)
        print(sum(game.rewards))
        replay_buffer.save_game(game)
        for train_step in range(20):
            batch = replay_buffer.sample_batch()
            observations, actions, labels = prepare_batch(
                batch, action_size, num_actions
            )
            model.train_on_batch(observations, actions, labels)
