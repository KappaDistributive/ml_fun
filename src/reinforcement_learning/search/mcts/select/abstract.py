from abc import ABC, abstractmethod
from typing import List, Tuple

from src.reinforcement_learning.graph.mcts import MCTSNode


class AbstractSelect(ABC):
    @abstractmethod
    def select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """
    Select an immediate child.
    :param node: The node of which we are selecting an immediate child.
    :return: (action, selected_child)
    """
        pass

    @abstractmethod
    def select_leaf(self, node: MCTSNode) -> List[Tuple[int, MCTSNode]]:
        """
    Select a path to a leaf node.
    :param node: The node of which we are selecting an immediate child.
    :return: [(a_0, node_0), ..., (a_k, node_k)], where (letting node_{-1} := node)
      - a_0, ..., a_k are actions,
      - for i = 0, 1, .., k, node_i is a child of node_{i -1},
      - for i = 0, 1, .., k, performing action a_i in node_{i - 1} leads to node_{i},
        i.e. node_{i - 1} -- ( a_i ) --> node_i
      - node_k is a leaf node, and
      - the length of the return list is always > 0.
    """
        pass
