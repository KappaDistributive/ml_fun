import random
from collections import deque
from copy import deepcopy
from typing import List, Tuple

import gym
import numpy as np

from src.muzero.model import AbstractMuZeroModel
from src.muzero.search import naive_search


class Game:
    def __init__(self, environment: gym.Env, discount_factor: float = 1.0):
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

    def terminal(self) -> bool:
        return self.done

    def get_observation(self, offset: int):
        return deepcopy(self.observations[offset])

    def get_history(self, offset: int = 0, max_length: int = -1):
        result = deepcopy(self.actions)
        if max_length >= 0:
            result = result[offset : offset + max_length]
        else:
            result = result[offset:]
        return result

    def make_target(
        self, offset: int, max_length: int = -1
    ) -> List[Tuple[float, float, np.ndarray]]:
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

    def apply(self, action, policy) -> None:
        self.observations.append(deepcopy(self.observation))
        self.actions.append(action)
        self.policies.append(policy)
        self.observation, reward, done, _ = self.environment.step(action)
        self.rewards.append(reward)
        self.total_reward += reward
        self.done = done
        self.num_steps += 1

    def act_with_policy(self, policy) -> None:
        action = np.random.choice(list(range(len(policy))), p=policy)
        self.apply(action, policy)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, num_actions: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.buffer = deque(maxlen=self.buffer_size)

    def save_game(self, game):
        self.buffer.append(game)

    def sample_game(self):
        return random.choice(self.buffer)

    def sample_batch(self):
        games = [self.sample_game() for _ in range(self.batch_size)]
        games = [(game, random.randint(0, len(game.actions) - 1)) for game in games]
        return [
            (
                game.get_observation(offset),
                game.get_history(offset, self.num_actions),
                game.make_target(offset, self.num_actions + 1),
            )
            for game, offset in games
        ]


def play_game(environment: gym.Env, model: AbstractMuZeroModel) -> Game:
    game = Game(environment)
    while not game.terminal():
        if random.random() < 0.05:
            policy = np.array([1 / model.action_size] * model.action_size)
        else:
            policy = naive_search(model, game.observation)
        game.act_with_policy(policy)
    return game
