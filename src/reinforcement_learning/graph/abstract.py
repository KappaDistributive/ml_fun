from abc import ABC, abstractmethod


class AbstractNode(ABC):
    @abstractmethod
    def is_expanded(self) -> bool:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        pass
