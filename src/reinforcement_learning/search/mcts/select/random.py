import random
from typing import List, Tuple

from src.reinforcement_learning.graph.mcts import MCTSNode
from src.reinforcement_learning.search.mcts.select.abstract import \
    AbstractSelect


class RandomSelect(AbstractSelect):
    def __init__(self):
        pass

    def select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        RandomSelect._check(node)
        action, child = random.choice(list(node.children.items()))

        return action, child

    def select_leaf(self, node: MCTSNode) -> List[Tuple[int, MCTSNode]]:
        RandomSelect._check(node)
        path = []
        current_node = node
        while not current_node.is_leaf():
            path.append(self.select_child(current_node))
            current_node = path[-1][1]

        return path

    @staticmethod
    def _check(node: MCTSNode) -> None:
        if node.is_terminal():
            raise ValueError(f"Cannot select a child of the terminal node: `{node}`")
        if not node.is_expanded():
            raise ValueError(f"Cannot select a child of the unexpanded node: `{node}`")
