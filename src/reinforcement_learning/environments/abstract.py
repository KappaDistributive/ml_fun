from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class AbstractEnviroment(ABC):
    """
  Motivated by OpenAI's env class: https://gym.openai.com/
  """

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Optional[Dict]]:
        """
    :param action: The action to be performed.
    :return: (observation, reward, done, info)
    """
        pass

    @abstractmethod
    def reset(self) -> Any:
        """
    :return: Observation
    """
        pass

    @abstractmethod
    def available_actions(self) -> List[int]:
        """
    :return: A list of all available (legal) actions.
    """
        pass
