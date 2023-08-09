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


__all__ = [ExplorationType.__name__, BoltzmannConfigs.__name__]
