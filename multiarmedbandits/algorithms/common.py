"""
Common base class for all multiarmed bandit algorithms
"""
from abc import ABC, abstractmethod

import numpy as np

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_positive_integer


class BaseModel(ABC):
    """create a basemodel class for multiarmed bandit models"""

    def __init__(self, bandit_env: BaseBanditEnv) -> None:
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        n_arms = bandit_env.n_arms
        max_steps = bandit_env.max_steps
        assert is_positive_integer(n_arms), f"{n_arms} should be a positive integer"
        assert is_positive_integer(max_steps), f"{n_arms} should be a positive integer"
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)

    @abstractmethod
    def select_arm(self, arm_attrib: ArmAttributes | None) -> int:
        """select arm given the specific multiarmed bandit algorithm

        Returns:
            int: arm to play
        """

    def __str__(self) -> str:
        """return name of class as string representation

        Returns:
            str: _description_
        """
        return self.__class__.__name__

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        # increment the chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        times_played_chosen_arm = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # update via memory trick
        self.values[chosen_arm] = self.memory_trick(times_played=times_played_chosen_arm, old_mean=value, value_to_add=reward)

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)

    @staticmethod
    def memory_trick(times_played: int, old_mean: float, value_to_add: float) -> float:
        """calculate mean value using memory trick

        Args:
            times_played (int): number of times played
            old_mean (float): old mean from `times_played`-1 values
            value_to_add (float): value to add for the mean

        Returns:
            float: updated mean value
        """
        return ((times_played - 1) / times_played) * old_mean + (1 / times_played) * value_to_add
