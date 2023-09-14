"""
explore then commit algorithm
"""

from numpy import argmax

from ..environments import ArmAttributes, BaseBanditEnv
from .common import BaseModel


class ExploreThenCommit(BaseModel):
    """explore then commit algorithm"""

    def __init__(self, explore: int, bandit_env: BaseBanditEnv) -> None:
        """initialize explore then commit algorithm

        Args:
            explore (int): number of steps to explore each arm
            n_arms (int): number of arms in the multi arm bandit
        """
        super().__init__(bandit_env=bandit_env)
        self.explore = explore

    def select_arm(self, arm_attrib: ArmAttributes) -> int:
        """select the best arm given the estimators of the values

        Args:
            arm_attrib (ArmAttributes): step in the game

        Returns:
            int: best action based on the estimators of the values
        """
        if self.explore * self.n_arms < arm_attrib.step_in_game:
            return argmax(self.values)
        return arm_attrib.step_in_game % self.n_arms
