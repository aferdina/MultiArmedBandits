""" 
Implementation of the Thompson Sampling algorithm.
"""
from typing import Any, Dict

import numpy as np

from multiarmedbandits.algorithms.common import BaseModel
from multiarmedbandits.environments import ArmAttributes, BaseBanditEnv
from multiarmedbandits.utils.posterior import PosteriorFactory


class ThompsonSampling(BaseModel):
    """
    Implementation of the Thompson Sampling algorithm.

    Thompson Sampling is an algorithm that uses the Bayesian approach to decide
    which arm to pull in the multi-armed bandit setting. It samples from the
    posterior distribution over the expected rewards of the arms and selects
    the arm with the highest sample.

    Attributes:
    - posterior (Posterior): The posterior distribution over the arms.

    Methods:
    - select_arm(arm_attrib: ArmAttributes | None) -> int: Returns the index of the selected arm.
    - update(chosen_arm: int, reward: float): Updates the posterior distribution with the received reward.
    - reset(): Resets the posterior distribution to its initial state.
    """

    def __init__(self, bandit_env: BaseBanditEnv, config: Dict[str, Any]) -> None:
        super().__init__(bandit_env)
        self.posterior = PosteriorFactory(bandit_env).create(config)

    def select_arm(self, arm_attrib: ArmAttributes | None):
        sampled_probs = self.posterior.sample()
        return np.argmax(sampled_probs)  # in doubt, always return the first maximal q-value

    def update(self, chosen_arm: int, reward: float) -> None:
        self.posterior.update(chosen_arm, reward)

    def reset(self) -> None:
        self.posterior.reset()
