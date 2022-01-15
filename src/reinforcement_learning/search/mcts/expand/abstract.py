from abc import ABC, abstractmethod

from src.reinforcement_learning.graph.mcts import MCTSNode


class AbstractExpand(ABC):
    @abstractmethod
    def expand(self, node: MCTSNode) -> None:
        pass
