"""
module for gradient bandit algorithm
"""

import statistics
from dataclasses import dataclass
from typing import Callable

import numpy as np
from strenum import StrEnum

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_positive_float
from .common import BaseLearningRule


class BaseLinesTypes(StrEnum):
    """different types of baselines for gradient method"""

    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass
class GradientBaseLineAttr:
    """required attributes for gradient bandit method"""

    type: BaseLinesTypes
    mean_reward: float = 0.0
    step_count: int = 0
    median: float = 0
    reward_history = []

    def reset(self):
        """reset statistics"""
        self.mean_reward = 0.0
        self.step_count = 0
        self.median = 0
        self.reward_history = []


class GradientBandit(BaseLearningRule):
    """gradient bandit algorithm"""

    def __init__(
        self,
        alpha: float,
        baseline_attr: GradientBaseLineAttr,
        bandit_env: BaseBanditEnv,
    ) -> None:
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(alpha), "Learning rate has to be a positive float"

        self.alpha: float = alpha
        self.calc_baseline = self._create_calc_baseline(baseline_typ=baseline_attr.type)
        self.baseline_attr = baseline_attr
        # for median calculation
        self.update_for_median = 0

    def _create_calc_baseline(self, baseline_typ: BaseLinesTypes) -> Callable[[GradientBaseLineAttr], float]:
        """create baseline function for given baseline type

        Args:
            baseline_typ (BaseLinesTypes): _description_
        """
        if baseline_typ == BaseLinesTypes.ZERO:

            def _calc_baseline(baseline_att: GradientBaseLineAttr) -> float:
                return 0.0

            return _calc_baseline
        if baseline_typ == BaseLinesTypes.MEAN:

            def _calc_baseline(baseline_att: GradientBaseLineAttr) -> float:
                return baseline_att.mean_reward

            return _calc_baseline
        if baseline_typ == BaseLinesTypes.MEDIAN:

            def _calc_baseline(baseline_att: GradientBaseLineAttr) -> float:
                baseline_att.reward_history.append(self.update_for_median)
                baseline_att.median = statistics.median(baseline_att.reward_history)
                return baseline_att.median

            return _calc_baseline

        raise ValueError("method not implemented")

    def calc_baseline(self, baseline_att: GradientBaseLineAttr) -> float:
        """calculate baseline for gradient algorithm

        Args:
            baseline_att (GradientBaseLineAttr): attributes to calculate baseline

        Returns:
            float: calculated baseline
        """
        return 0.0

    def get_prob(self, action: int) -> float:
        """get probability for a given action

        Args:
            action (int):

        Returns:
            float: probability for a given action
        """
        input_vector = np.exp(self.values)
        return float(input_vector[action] / np.sum(input_vector))

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """choose arm in the gradient bandit algorithmus

        Returns:
            int: sampled action
        """
        input_vector = np.exp(self.values)
        input_vector = input_vector / np.sum(input_vector)
        return np.random.choice(self.n_arms, p=input_vector)

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        action_prob = self.get_prob(chosen_arm)
        # increment the chosen arm
        action_prob_vec = np.array([-1 * action_prob for _ in range(self.n_arms)])
        action_prob_vec[chosen_arm] = 1 - action_prob
        # update for median
        self.update_for_median = reward
        # update via memory trick
        baseline = self.calc_baseline(baseline_att=self.baseline_attr)
        gradients = (self.alpha * (reward - baseline)) * action_prob_vec

        # update values
        self.values = self.values + gradients
        self.baseline_attr.step_count += 1
        # update mean reward
        self.baseline_attr.mean_reward = (
            (self.baseline_attr.step_count - 1) / float(self.baseline_attr.step_count)
        ) * self.baseline_attr.mean_reward + (1 / float(self.baseline_attr.step_count)) * reward

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.baseline_attr.reset()
