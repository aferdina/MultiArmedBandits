""" util classes for multiarmed bandit environments
"""
from typing import List
from dataclasses import dataclass
from strenum import StrEnum


class INFODICT(StrEnum):
    """Enum class for information dictionary and multi armed bandit environment"""

    STEPCOUNT = "count"
    REGRET = "regret"
    ARMATTRIBUTES = "arm_attributes"


class ArmDistTypes(StrEnum):
    """types of arm distributions"""

    GAUSSIAN = "gaussian"
    BERNOULLI = "bernoulli"


@dataclass
class DistParameter:
    """distribution parameter for arms in multiarmed bandit problems"""

    dist_type: ArmDistTypes
    mean_parameter: list[float]
    scale_parameter: list[float] | None = None


@dataclass
class BanditStatistics:
    """statistics for bandit models"""

    max_mean: float  # maximal mean
    max_mean_positions: List[int]  # position of the maximal mean
    played_optimal: int = 0  # count number of times optimal played
    regret: float = 0.0  # calculate regret

    def reset_statistics(self) -> None:
        """reset statistics"""
        self.played_optimal = 0
        self.regret = 0.0


@dataclass
class ArmAttributes:
    """class to store all attributes for select arm method"""

    step_in_game: int | None = None


__all__ = [
    ArmAttributes.__name__,
    BanditStatistics.__name__,
    DistParameter.__name__,
    ArmDistTypes.__name__,
    INFODICT.__name__,
]
