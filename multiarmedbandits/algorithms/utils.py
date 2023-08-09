""" helpclasses for algorithms
"""
from strenum import StrEnum
from dataclasses import dataclass


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


class BaseLinesTypes(StrEnum):
    ZERO = "zero"
    MEAN = "mean"


@dataclass
class GradientBaseLineAttr:
    type: BaseLinesTypes
    mean_reward: float = 0.0
    step_count: int = 0

    def reset(self):
        self.mean_reward = 0.0
        self.step_count = 0


__all__ = [
    ExplorationType.__name__,
    BoltzmannConfigs.__name__,
    BaseLinesTypes.__name__,
    GradientBaseLineAttr.__name__,
]
