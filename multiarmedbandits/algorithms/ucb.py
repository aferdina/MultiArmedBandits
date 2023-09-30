"""
ucb algorithm for multi armed bandits
"""
from abc import abstractmethod

import numpy as np

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_float_between_0_and_1, is_positive_float
from .common import BaseModel


class UCB(BaseModel):
    """class for ucb algorithm"""

    def __init__(self, bandit_env: BaseBanditEnv) -> None:
        """initialize upper confidence bound algorithm

        Args:
            bandit_env (BaseBanditEnv): bandit environment to use
        """
        super().__init__(bandit_env=bandit_env)
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select arm to play according to the ucb values

        Returns:
            int: best action based on upper confidence bound
        """
        return np.argmax(self.ucb_values)

    @abstractmethod
    def _calc_bonus(self, chosen_arm: int) -> list[float]:
        """calculate exploration bonus for UCB algorithm

        Args:
            chosen_arm (int): action which was played

        Returns:
            list[float]: exploration bonus
        """

    def update(self, chosen_arm: int, reward: float) -> None:
        """update average rewards, counts and ucb values

        Args:
            chosen_arm (int): action which was played and should be updated
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        super().update(chosen_arm=chosen_arm, reward=reward)
        bonus = self._calc_bonus(chosen_arm)
        self.ucb_values[chosen_arm] = self.values[chosen_arm] + bonus

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        super().reset()
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)


class LectureUCB(UCB):
    """UCB algorithm from RL lecture"""

    def __init__(self, bandit_env: BaseBanditEnv, delta: float):
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(delta), f"{delta} should be a float between 0 and 1"
        self.delta = delta

    def _calc_bonus(self, chosen_arm: int) -> list[float]:
        return np.sqrt(-2 * np.log(self.delta) / self.counts[chosen_arm])


class UCBAlpha(UCB):
    """UCB(alpha) algorithm"""

    def __init__(self, bandit_env: BaseBanditEnv, alpha: float):
        super().__init__(bandit_env=bandit_env)
        assert is_positive_float(alpha), f"{alpha} should be a postive float"
        self.alpha = alpha

    def _calc_bonus(self, chosen_arm: int) -> list[float]:
        """calculate the exploration bonus of the UCB_Alpha algorithm.

        Returns:
            list[float]: exploration bonus
        """
        bonus = np.sqrt(self.alpha * np.log(sum(self.counts)) / (2 * self.counts[chosen_arm]))
        return bonus
