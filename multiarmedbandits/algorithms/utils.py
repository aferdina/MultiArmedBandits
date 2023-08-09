""" helpclasses for algorithms
"""
from dataclasses import dataclass
from strenum import StrEnum


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
    """different types of baselines for gradient method"""

    ZERO = "zero"
    MEAN = "mean"


@dataclass
class GradientBaseLineAttr:
    """required attributes for gradient bandit method"""

    type: BaseLinesTypes
    mean_reward: float = 0.0
    step_count: int = 0

    def reset(self):
        """reset statistics"""
        self.mean_reward = 0.0
        self.step_count = 0


__all__ = [
    ExplorationType.__name__,
    BoltzmannConfigs.__name__,
    BaseLinesTypes.__name__,
    GradientBaseLineAttr.__name__,
]
