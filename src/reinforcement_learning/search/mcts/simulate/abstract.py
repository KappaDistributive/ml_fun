from abc import ABC, abstractmethod

from src.reinforcement_learning.graph.mcts import MCTSNode


class AbstractSimulate(ABC):
    @abstractmethod
    def simulate(self, node: MCTSNode) -> float:
        pass
