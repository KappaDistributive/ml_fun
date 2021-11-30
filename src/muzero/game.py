import random
from collections import deque
from copy import deepcopy
from typing import Any, List, Tuple

import gym
import numpy as np

from src.muzero.model import AbstractMuZeroModel
from src.muzero.search import mcts, naive_search


class Game:
    def __init__(self, environment: gym.Env, discount_factor: float = 1.0):
        """
        :param environment: A gym environment.
        :param discount_factor: The discoutn factor applied to future rewards.
        """
        self.environment = environment
        self.discount_factor = discount_factor
        self.observations = []
        self.actions = []
        self.rewards = []
        self.policies = []
        self.done = False
        self.observation = self.environment.reset()
        self.total_reward = 0.0
        self.num_steps = 0

    def is_terminal(self) -> bool:
        """
        Return true iff the game is finished.
        :return: True iff the game is finished.
        """
        return self.done

    def get_observation(self, offset: int) -> Any:
        """
        Return the observation at step `offset`.
        :param offset: The step to be considered.
        :return: The observation at step `offset`.
        """
        return deepcopy(self.observations[offset])

    def get_actions(self, offset: int = 0, max_length: int = -1) -> List[int]:
        """
        :param offset: Return actions from step `offset` onwards.
        :param max_length: The maximal number of actions to return. If equal to -1, don't limit the number of actions.
        :return: The list of actions performed from step `offset` onwards.
        """
        actions = deepcopy(self.actions)
        if max_length >= 0:
            actions = actions[offset : offset + max_length]
        else:
            actions = actions[offset:]
        return actions

    def make_target(
        self, offset: int, max_length: int = -1
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Create targets for a MuZero-agent.
        :param offset: Start creating a sample from step `offset` onwards in the game.
        :param max_length: The maximum number of states.
        :return: Each entry has the form (value, last_reward, policy).
        """
        # TODO: This is quite confusing and should be refactored.
        if offset >= self.num_steps:
            return []

        if max_length < 0:
            max_length = self.num_steps - offset
        targets = []

        for index in range(offset, offset + max_length):
            if index >= self.num_steps:
                break
            value = 0.0

            for distance, reward in enumerate(self.rewards[index:]):
                value += reward * (self.discount_factor ** distance)

            if 1 <= index < len(self.rewards):
                last_reward = self.rewards[index - 1]
            else:
                last_reward = 0.0
            targets.append((value, last_reward, self.policies[index]))

        return targets

    def apply(self, action: Any, policy: np.ndarray) -> None:
        """
        Apply `action` and perform some bookkeeping.
        :param action: The action to be applied.
        :param policy: A distribution over all possible actions that `action` has been sampled from.
        :return: None.
        """
        self.observations.append(deepcopy(self.observation))
        self.actions.append(action)
        self.policies.append(policy)
        self.observation, reward, done, _ = self.environment.step(action)
        self.rewards.append(reward)
        self.total_reward += reward
        self.done = done
        self.num_steps += 1

    def act_with_policy(self, policy: np.ndarray) -> None:
        """
        Sample an action from the distribution given by `policy` and apply it.
        :param policy: A distribution over all possible actions.
        :return: None.
        """
        action = np.random.choice(list(range(len(policy))), p=policy)
        self.apply(action, policy)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, lookahead_range: int):
        """
        :param buffer_size: The maximal size of the internal buffer.
        :param batch_size:  Batch size.
        :param lookahead_range: Lookahead range, i.e. the number of actions performed in each sequence.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lookahead_range = lookahead_range
        self.buffer = deque(maxlen=self.buffer_size)

    def save_game(self, game: Game) -> None:
        """
        Add a game to the replay buffer.
        :param game: The game that should be added to the replay buffer.
        :return: None.
        """
        self.buffer.append(game)

    def sample_game(self):
        """
        Sample a random game from the replay buffer.
        Note that the game is not removed from the replay buffer.
        :return: A random game from the replay buffer.
        """
        # TODO: Implement priority-ranked sampling.
        return random.choice(self.buffer)

    def sample_batch(
        self,
    ) -> List[Tuple[np.ndarray, List[int], List[Tuple[float, float, np.ndarray]]]]:
        """
        Create a batch of size `self.batch_size` by sampling (with replacement) from the replay buffer.
        :return: A batch of size `self.batch_size`, where each entry is of the form (observation, actions, targets).
        """
        games = [self.sample_game() for _ in range(self.batch_size)]
        games = [(game, random.randint(0, len(game.actions) - 1)) for game in games]
        return [
            (
                game.get_observation(offset),
                game.get_actions(offset, self.lookahead_range),
                game.make_target(offset, self.lookahead_range + 1),
            )
            for game, offset in games
        ]


def play_game(
    environment: gym.Env,
    model: AbstractMuZeroModel,
    epsilon: float = 0.05,
    num_simulations: int = 10,
) -> Game:
    """
    Play a game in `environment` to completion where actions are sampled from the policy created by `model`.
    :param environment: A gym environment.
    :param model: A MuZero agent.
    :param epsilon: With probabilty `epsilon`, perform a random action rather than sampling from the model policy.
    :param num_simulations: The number of simulations to be performed each step.
    :return: A completed game, played according to the policy created by `model`.
    """
    game = Game(environment)
    while not game.is_terminal():
        if random.random() < epsilon:
            policy = np.array([1 / model.action_size] * model.action_size)
        else:
            naive_policy, _ = naive_search(model, game.observation)
            # TODO: MCTS is broken. Do I need priors?
            policy, _ = mcts(
                model, game.observation, num_simulations, ignore_to_play=True
            )
            # print(f"Naive policy: {naive_policy}")
            # print(f"MCTS policy: {policy}")
            # print()
        game.act_with_policy(policy)
    return game
