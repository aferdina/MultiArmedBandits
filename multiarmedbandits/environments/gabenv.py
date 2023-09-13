""" This file contains the implementation of Gap environment from paper `Boltzmann exploration done right`
"""

from dataclasses import dataclass
from .common import ArmDistTypes, DistParameter, BaseBanditEnv


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


class GapEnv(BaseBanditEnv):
    """class for Gap enviroment from paper `Boltzmann exploration done right`"""

    def __init__(self, gap_configs: GapEnvConfigs, max_steps: int) -> None:
        super().__init__(distr_params=gap_configs.distr_parameter, max_steps=max_steps)
