""" thompson sampling algorithm for multi-armed bandits
"""
from typing import Any, Dict
import numpy as np
from multiarmedbandits.algorithms.common import BaseModel
from multiarmedbandits.environments import ArmAttributes, BaseBanditEnv
from multiarmedbandits.utils.posterior import PosteriorFactory

class ThompsonSampling(BaseModel):
    """
    class for thompson sampling algorithm
    """
    def __init__(self, bandit: BaseBanditEnv, config: Dict[str, Any]) -> None:
        super().__init__(bandit)
        self.posterior = PosteriorFactory(bandit).create(config)

    def select_arm(self, arm_attrib: ArmAttributes | None):
        sampled_probs = self.posterior.sample()
        return np.argmax(sampled_probs) # in doubt, always return the first maximal q-value

    def update(self, chosen_arm: int, reward: float) -> None:
        self.posterior.update(chosen_arm, reward)

    def reset(self) -> None:
        self.posterior.reset()
