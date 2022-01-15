from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.reinforcement_learning.environments.abstract import AbstractEnviroment
from src.reinforcement_learning.graph.abstract import AbstractNode


class MultiArmedBandit(AbstractEnviroment):
    """
    A multi-armed bandit using independent random variables of a common family of distributions.
    See [Asymptotically efficient adaptive allocation rules](https://www.sciencedirect.com/science/article/pii/0196885885900028)
    """

    def __init__(
        self,
        distribution_parameters: List[Any],
        distribution_family: str = "normal",
        max_num_rounds: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self.distribution_parameters = distribution_parameters
        self.distribution_family = distribution_family.lower()
        self.max_num_rounds = max_num_rounds
        self.random_seed = random_seed
        self.observation = 0  # `oberservation` is the number of rounds played thus far

        if self.max_num_rounds is not None:
            assert self.max_num_rounds > 0
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        assert (
            self.distribution_family == "normal"
        ), f"Missing support for distribution family `{self.distribution_family}`"

        if self.distribution_family == "normal":
            assert all(
                isinstance(parameter, tuple)
                for parameter in self.distribution_parameters
            )
            assert all(
                len(parameter) == 2 for parameter in self.distribution_parameters
            )  # parameter = (mean, standard_deviation)
            assert all(
                isinstance(parameter[0], float)
                for parameter in self.distribution_parameters
            )
            assert all(
                isinstance(parameter[1], float) and parameter[1] >= 0
                for parameter in self.distribution_parameters
            )

    def reset(self) -> Any:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.observation = 0

        return self.observation

    def step(self, action: int) -> Tuple[Any, float, bool, Optional[Dict]]:
        assert action in self.available_actions(), f"Invalid action: `{action}`"
        parameter = self.distribution_parameters[action]
        reward = np.random.normal(parameter[0], parameter[1])
        self.observation += 1
        done = False
        if self.max_num_rounds is not None:
            done = self.observation >= self.max_num_rounds

        return self.observation, reward, done, None

    def available_actions(self) -> List[int]:
        return list(range(len(self.distribution_parameters)))

    def restore(self, node: AbstractNode) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        distributions = [
            f"{self.distribution_family.title()}({parameter})"
            for parameter in self.distribution_parameters
        ]
        return f"{distributions}\nObservation: {self.observation}"
