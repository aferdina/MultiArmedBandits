"""
ucb algorithm for multi armed bandits
"""

import numpy as np

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_float_between_0_and_1
from .common import BaseModel


class UCB(BaseModel):
    """class for ucb algorithm"""

    def __init__(self, delta: float, bandit_env: BaseBanditEnv) -> None:
        """initialize upper confidence bound algorithm

        Args:
            n_arms (int): number of arms in the multiarmed bandit model
            delta (float): delta parameter of ucb algorithm
        """
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(delta), f"{delta} should be a float between 0 and 1"
        self.delta = delta
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)
        self._exploration_factor = np.sqrt(-2 * np.log(self.delta)) * np.ones(self.n_arms, dtype=np.float32)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select the best arm given the value estimators and the ucb bound
        Returns:
            int: best action based on upper confidence bound
        """
        return np.argmax(self.ucb_values)

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the ucb bound of the ucb algorithm

        Args:
            chosen_arm (int): action which was played an should be updated
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        super().update(chosen_arm=chosen_arm, reward=reward)
        # update all arms which are played at least one time
        # # pylint: disable=C0301
        _square_counts = np.sqrt(self.counts)
        bonus = np.divide(
            self._exploration_factor,
            _square_counts,
            out=np.full(self.n_arms, np.inf, dtype=np.float32),
            where=_square_counts != 0,
        )
        self.ucb_values = self.values + bonus

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        super().reset()
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)
