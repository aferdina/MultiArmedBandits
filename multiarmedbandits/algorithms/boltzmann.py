"""
ucb algorithm for multi armed bandits
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from strenum import StrEnum

from ..environments import ArmAttributes, BaseBanditEnv
from ..utils import is_list_of_floats
from .common import BaseLearningRule


class ExplorationType(StrEnum):
    """different types of Exploration in Boltzmann exploration Algorithms"""

    CONSTANT = "constant"
    LOG = "log"
    SQRT = "sqrt"
    UCB = "ucb"
    BGE = "bge"


@dataclass
class BoltzmannConfigs:
    """configuration for boltzmann exploration"""

    explor_type: ExplorationType
    some_constant: list[float]


class BoltzmannSimple(BaseLearningRule):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(self, boltzmann_configs: BoltzmannConfigs, bandit_env: BaseBanditEnv):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_list_of_floats(boltzmann_configs.some_constant), "The temperature  has to be a positive float"
        assert (
            len(boltzmann_configs.some_constant) == self.n_arms
        ), "temperature parameter should be of same size as number of arms"
        self.some_constant = np.array(boltzmann_configs.some_constant, dtype=np.float32)
        self.calc_betas = self._create_calc_betas(explor_type=boltzmann_configs.explor_type)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """choose an arm from the boltzmann distribution

        Returns:
            int: simulated action
        """
        _betas = self.calc_betas(arm_attrib=arm_attrib)
        _logits = self.values / _betas
        probs = np.exp(_logits) / np.sum(np.exp(_logits))
        return np.random.choice(self.n_arms, p=probs)

    def calc_betas(self, arm_attrib: ArmAttributes | None = None) -> np.ndarray:
        """calculating beta values for boltzmann algorithm

        Args:
            arm_attrib (ArmAttributes): parameter for calculating beta values

        Returns:
            np.ndarray: beta values for algorithm
        """
        return np.ones_like(self.values)

    def _create_calc_betas(self, explor_type: ExplorationType) -> Callable[[ArmAttributes], np.ndarray]:
        """create method to calculate beta values for boltzmann algorithm
        based on configs

        Returns:
            Callable[[ArmAttributes], float]: method to calculate beta values for boltzmann
            algorithm
        """
        match explor_type:
            case ExplorationType.CONSTANT:

                def _calc_betas(arm_attrib: ArmAttributes | None = None) -> np.ndarray:
                    return self.some_constant**2

                return _calc_betas

            case ExplorationType.SQRT:

                def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                    return self.some_constant**2 / np.sqrt(1 + arm_attrib.step_in_game)

                return _calc_betas

            case ExplorationType.LOG:

                def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                    if np.log(1 + arm_attrib.step_in_game) == 0.0:
                        return np.full_like(self.values, np.inf)
                    return self.some_constant**2 / np.log(1 + arm_attrib.step_in_game)

                return _calc_betas

            case ExplorationType.UCB:

                def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                    _sqrt_counts = np.sqrt(self.counts)
                    result = np.divide(
                        self.some_constant,
                        _sqrt_counts,
                        out=np.zeros_like(self.some_constant),
                        where=_sqrt_counts != 0,
                    )
                    result = result * np.sqrt(np.log(1 + arm_attrib.step_in_game))
                    result[result == 0.0] = np.inf
                    return result

                return _calc_betas

            case ExplorationType.BGE:

                def _calc_betas(arm_attrib: ArmAttributes | None = None) -> np.ndarray:
                    _sqrt_counts = np.sqrt(self.counts)
                    result = np.divide(
                        self.some_constant,
                        _sqrt_counts,
                        out=np.zeros_like(self.some_constant),
                        where=_sqrt_counts != 0,
                    )
                    result[result == 0.0] = np.inf
                    return result

                return _calc_betas

            case _:
                raise NotImplementedError(f"{explor_type} not implemented yet")
