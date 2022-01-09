from typing import Any, Dict

from src.reinforcement_learning.graph.abstract import AbstractNode


class MCTSNode(AbstractNode):
    """
  Intended to be used in the Monte Carlo Tree Search algorithm for one- or two-player games.
  """

    def __init__(
        self, observation: Any, reward: float, is_terminal: bool, to_play: int = 1
    ):
        super().__init__()
        self.observation = observation
        self.reward = reward
        self.is_terminal = is_terminal
        self.to_play = to_play
        self.num_visits = 0
        self.sum_of_values = 0.0
        self.children: Dict[int, MCTSNode] = {}

    def value(self) -> float:
        if self.num_visits == 0:
            return self.to_play * float("-infinity")

        assert self.num_visits > 0
        return self.to_play * (self.sum_of_values / self.num_visits)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        return self.is_terminal
