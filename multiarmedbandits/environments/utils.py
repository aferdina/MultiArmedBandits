""" util classes for multiarmed bandit environments
"""
from dataclasses import dataclass
from typing import List

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


@dataclass
class SingleArmParams:
    """parameter class for storing parameter for one arm in mab env"""

    arm_type: ArmDistTypes
    mean_parameter: float
    scale_parameter: float | None = None


@dataclass
class GapEnvConfigs:
    """storing all parameter for gab environment"""

    no_of_arms: int
    single_arm_distr: SingleArmParams
    gap_parameter: float
    distr_parameter: DistParameter | None = None

    def __post_init__(self):
        mean_parameter = [self.single_arm_distr.mean_parameter for _ in range(self.no_of_arms)]
        mean_parameter[0] = mean_parameter[0] + self.gap_parameter
        scale_parameter = None
        if self.single_arm_distr.arm_type == ArmDistTypes.GAUSSIAN:
            scale_parameter = [self.single_arm_distr.scale_parameter for _ in range(self.no_of_arms)]
        self.distr_parameter = DistParameter(
            dist_type=self.single_arm_distr.arm_type,
            mean_parameter=mean_parameter,
            scale_parameter=scale_parameter,
        )


__all__ = [
    ArmAttributes.__name__,
    BanditStatistics.__name__,
    DistParameter.__name__,
    ArmDistTypes.__name__,
    INFODICT.__name__,
    SingleArmParams.__name__,
    GapEnvConfigs.__name__,
]
