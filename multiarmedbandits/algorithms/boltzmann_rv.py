"""
Boltzmann algorithm with random variables
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ..environments import ArmAttributes, BaseBanditEnv
from .boltzmann import BoltzmannConfigs, BoltzmannSimple


@dataclass
class RandomVariable:
    """store properties for random variable in boltzmann algorithms"""

    rv_name: str
    rv_param: dict[str, Any]

    def __post_init__(self):
        assert self.rv_name in dir(stats)


class BoltzmannGeneral(BoltzmannSimple):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(
        self,
        boltzmann_configs: BoltzmannConfigs,
        bandit_env: BaseBanditEnv,
        rv_config: RandomVariable,
    ):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env, boltzmann_configs=boltzmann_configs)
        self.random_variables = self.sample_random_variables(rv_config=rv_config)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """get action from boltzmann gumbel paper"""

        random_variables = self.random_variables[arm_attrib.step_in_game,]
        _betas = self.calc_betas(arm_attrib=arm_attrib)
        _used_parameter = self.values + _betas * random_variables
        return int(np.argmax(_used_parameter))

    def sample_random_variables(self, rv_config: RandomVariable) -> np.ndarray:
        """get realization of random variables for algorithm

        Returns:
            np.ndarray: all required random variables
        """
        _dist = getattr(stats, rv_config.rv_name)
        return _dist(**rv_config.rv_param).rvs(size=(self.max_steps, self.n_arms))
