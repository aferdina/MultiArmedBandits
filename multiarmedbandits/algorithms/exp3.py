"""
exp3 algorithm for multi armed bandits
"""
from abc import abstractmethod

import numpy as np

from multiarmedbandits.environments import BaseBanditEnv

from ..environments import ArmAttributes, BaseBanditEnv
from .common import BaseLearningRule

class exp3(BaseLearningRule):
    """class for exp3 algorithm"""

    def __init__(self, gamma: float, bandit_env: BaseBanditEnv) -> None:
        super().__init__(bandit_env)
        self.gamma = gamma
        self.prb_arms =  [0] * self.n_arms
        self.prb_weights = [1/self.n_arms] * self.n_arms

        def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
         """select arm to play according to the probability vector

         Returns:
            int: sampled arm
         """
        return np.random.choice(self.n_arms, p=self.prb_weights)



    def update(self, chosen_arm: int, reward: float) -> None:
        """update prb_weights and prb_arms

        Args:
            chosen_arm (int): action which was played and should be updated
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        super().update(chosen_arm=chosen_arm, reward=reward)
        self.prb_weights[chosen_arm] = self.prb_weights[chosen_arm] * np.exp(self.gamma*((reward/self.prb_arms[chosen_arm])/self.n_arms))
        # updating the weight of the currently played arm based on the correction factor for currently played arm
        for i in range(self.n_arms): 
         self.prb_arms[i] = (1-self.gamma)* self.prb_weights[i]/sum(self.prb_weights) + self.gamma/self.n_arms
        # probability of action based on probability weights and gamma
       