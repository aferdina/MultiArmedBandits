""" epsilon greedy algorithm for multi armed bandits
"""
import random

import numpy as np

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_float_between_0_and_1
from .common import BaseLearningRule


class EpsilonGreedy(BaseLearningRule):
    """class for epsilon greedy algorithm"""

    def __init__(self, epsilon: float, bandit_env: BaseBanditEnv) -> None:
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(epsilon), f"{epsilon} should be a float between 0 and 1"
        self.epsilon = epsilon

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select the best arm by using epsilon gready method

        Returns:
            int: best action based on the estimators of the values
        """
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        return random.randrange(self.n_arms)
