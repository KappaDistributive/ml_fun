from __future__ import annotations

import dataclasses
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple

Action = Any
State = Any


@dataclasses.dataclass
class Node(ABC):
    to_play: int
    state: State
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[Action, Node] = dataclasses.field(default_factory=dict)
    reward: float = 0.0
    is_terminal: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS(ABC):
    def __init__(
        self,
        initial_state: State,
        initial_player: int,
        exploration_parameter: float = math.sqrt(2.0),
    ):
        """
        :param exploration_parameter: Exploration parameter used in the calculation of `ucb_score`.
        """
        self.initial_state = deepcopy(initial_state)
        self.root = Node(to_play=initial_player, state=self.initial_state)
        self.exploration_parameter = exploration_parameter

    def ucb_score(self, parent: Node, child: Node) -> float:
        if child.visit_count == 0:
            return float("inf")

        assert (
            parent.visit_count > 0
        ), "Encountered a parent that has never been visited even though its child has been visited."

        return child.value() + self.exploration_parameter * math.sqrt(
            math.log(parent.visit_count) / child.visit_count
        )

    def select(self, node: Node) -> Tuple[Action, Node]:
        """
        Selects the best child (and the action leading to it) of a given node according as measures by ucb scores.
        :param node: Parent node.
        :return: (action, best_child)
        """
        candidates = [
            (self.ucb_score(node, child), action, child)
            for action, child in node.children.items()
        ]

        max_score = max([candidate[0] for candidate in candidates])

        return random.choice(
            [candidate for candidate in candidates if candidate[0] == max_score]
        )[1:]

    def rollout(self, node: Node):
        path = self._traverse(node)
        action, leaf_node = path[-1]
        self._add_children(leaf_node)
        reward = self._simulate(leaf_node)
        self._backpropagate(path, reward)

    @abstractmethod
    def _random_child(self, node: Node) -> Tuple[Action, Node]:
        raise NotImplemented

    @abstractmethod
    def _add_children(self, node: Node) -> None:
        raise NotImplemented

    @abstractmethod
    def _simulate(self, node: Node) -> float:
        raise NotImplemented

    def _traverse(self, node: Node) -> List[Action, Node]:
        current_action, current_node = (None, node)
        path = []
        while True:
            path.append((current_action, current_node))
            if len(current_node.children) == 0 or current_node.visit_count == 0:
                return path
            current_action, current_node = self.select(current_node)

    @staticmethod
    def _backpropagate(path: List[Tuple[Action, Node]], reward: float) -> None:
        for _, node in reversed(path):
            node.visit_count += 1
            node.value_sum += path[0][1].to_play * reward
