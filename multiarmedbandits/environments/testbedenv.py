""" Include all game environments for multi armed bandits
"""
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from strenum import StrEnum

from .common import ArmDistTypes, BaseBanditEnv, DistParameter


class TestBedSampleType(StrEnum):
    """distribution class to sample arm parameters"""

    GAUSSIAN = "normal"
    BERNOULLI = "binomial"


@dataclass
class TestBedConfigs:
    """configuration for test bed classes"""

    type: TestBedSampleType
    sample_config: dict[str, Any]
    no_arms: int
    arm_type: ArmDistTypes


class TestBed(BaseBanditEnv):
    """test bed implementation of multiarmed bandit environment from sutton"""

    def __init__(self, max_steps: int, testbed_config: TestBedConfigs) -> None:
        self.testbed_config = testbed_config
        distr_param = self.get_distr_from_testbed()
        super().__init__(max_steps=max_steps, distr_params=distr_param)
        self.reset()

    def get_distr_from_testbed(self) -> DistParameter:
        """get distribution parameter from test bed configs

        Args:
            config (TestBedConfigs): configs from testbed

        Returns:
            DistParameter: distribution parameter for multiarmed bandit
        """

        rvs = getattr(np.random, self.testbed_config.type)
        mean_parameter: np.ndarray = rvs(**self.testbed_config.sample_config, size=self.testbed_config.no_arms)
        mean_parameter = mean_parameter.tolist()
        scale_parameter = None
        if self.testbed_config.arm_type == TestBedSampleType.GAUSSIAN:
            scale_parameter = [1.0 for _ in range(self.testbed_config.no_arms)]
        return DistParameter(
            dist_type=self.testbed_config.arm_type,
            mean_parameter=mean_parameter,
            scale_parameter=scale_parameter,
        )

    def reset(self) -> Tuple[int, dict[str, Any]]:
        _state, info = super().reset()
        self.distr_params = self.get_distr_from_testbed()
        self._create_reward_function()
        mean_parameter = self.distr_params.mean_parameter
        self.bandit_statistics.max_mean = max(mean_parameter)
        self.bandit_statistics.max_mean_positions = [
            index for index, value in enumerate(mean_parameter) if value == self.bandit_statistics.max_mean
        ]
        return _state, info
