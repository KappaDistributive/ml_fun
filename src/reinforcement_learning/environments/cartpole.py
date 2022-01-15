from typing import Any, Dict, List, Optional, Tuple

import gym

from src.reinforcement_learning.environments.abstract import AbstractEnviroment


class CartPole(AbstractEnviroment):
    def __init__(self):
        self.environment = gym.make("CartPole-v0")
        pass

    def reset(self) -> Any:
        return self.environment.reset()

    def step(self, action: int) -> Tuple[Any, float, bool, Optional[Dict]]:
        return self.environment.step(action)

    def available_actions(self) -> List[int]:
        return [0, 1]

    def render(self) -> Any:
        return self.environment.render()
