""" Include all game environments for multi armed bandits
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Callable, List
from strenum import StrEnum
import numpy as np
from multiarmedbandits.utils import (
    is_positive_integer,
    ArmAttributes,
)


class INFODICT(StrEnum):
    """Enum class for infromation dictionary"""

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


class BaseBanditEnv:
    """class for a basic multiarmed bandit model"""

    def __init__(self, distr_params: DistParameter, max_steps: int) -> None:
        """create a multiarm bandit with `len(distr_params.mean_parameter)` arms

        Args:
            distr_params (DistParameter): dataclass containing distribution parameter
            for arms of multiarm bandit
            max_steps (int): maximal number of steps to play in the multi arm bandit
        """
        # TODO: write test file for mab configs
        assert is_positive_integer(
            max_steps
        ), "The number of steps should be a positive integer"
        self.n_arms: int = len(distr_params.mean_parameter)
        self.max_steps: int = max_steps
        self.count: int = 0
        self.done: bool = False
        self.distr_params: DistParameter = distr_params
        # maximal mean and position of maximal mean
        mean_parameter = self.distr_params.mean_parameter
        self.bandit_statistics: BanditStatistics = BanditStatistics(
            max_mean=max(mean_parameter),
            max_mean_positions=[
                index
                for index, value in enumerate(mean_parameter)
                if value == max(mean_parameter)
            ],
        )
        self.bandit_statistics.reset_statistics()

        setattr(self, "get_reward", self._create_reward_function())

    def get_reward(self, action: int) -> float:
        """get reward for a given action

        Args:
            action (int): action which is played

        Returns:
            float: reward for playing an specific action
        """
        return float(action)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """run a step in the multiarmed bandit

        Args:
            action (int): choose arm to play

        Returns:
            Tuple[int, float, bool, Dict[str, Any]]: next state, reward,
            bool if done, information dict
        """
        assert action in range(self.n_arms), f"the action {action} is not valid"
        reward = self.get_reward(action=action)
        self.count += 1

        # check if best action was played
        if action in self.bandit_statistics.max_mean_positions:
            self.bandit_statistics.played_optimal += 1

        # update the regret in the game
        self.bandit_statistics.regret += self.bandit_statistics.max_mean - reward

        # if game is finished `done=True`
        done = bool(self.count >= self.max_steps)
        self.done = done
        return (
            0,
            reward,
            done,
            {
                INFODICT.REGRET: self.bandit_statistics.regret,
                INFODICT.STEPCOUNT: self.count,
                INFODICT.ARMATTRIBUTES: ArmAttributes(step_in_game=self.count),
            },
        )

    def reset(self) -> None:
        """reset all statistics to run a new game"""
        self.count = 0
        self.done = False
        self.bandit_statistics.reset_statistics()
        return (
            0,
            0,
            False,
            {
                INFODICT.REGRET: 0,
                INFODICT.STEPCOUNT: self.count,
                INFODICT.ARMATTRIBUTES: ArmAttributes(step_in_game=self.count),
            },
        )

    def _create_reward_function(self) -> Callable[[int], float]:
        if self.distr_params.dist_type == ArmDistTypes.BERNOULLI:

            def _get_reward(action: int) -> float:
                reward = (
                    1.0
                    if np.random.uniform() < self.distr_params.mean_parameter[action]
                    else 0.0
                )
                return reward

            return _get_reward
        if self.distr_params.dist_type == ArmDistTypes.GAUSSIAN:

            def _get_reward(action: int) -> float:
                return np.random.normal(
                    loc=self.distr_params.mean_parameter[action],
                    scale=self.distr_params.scale_parameter[action],
                    size=None,
                )

            return _get_reward
        raise ValueError("Something went wrong")


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
        mean_parameter: np.ndarray = rvs(
            **self.testbed_config.sample_config, size=self.testbed_config.no_arms
        )
        mean_parameter = mean_parameter.tolist()
        scale_parameter = None
        if self.testbed_config.arm_type == TestBedSampleType.GAUSSIAN:
            scale_parameter = [1.0 for _ in range(self.testbed_config.no_arms)]
        return DistParameter(
            dist_type=self.testbed_config.arm_type,
            mean_parameter=mean_parameter,
            scale_parameter=scale_parameter,
        )

    def reset(self) -> None:
        super().reset()
        self.distr_params = self.get_distr_from_testbed()
        self._create_reward_function()
        mean_parameter = self.distr_params.mean_parameter
        self.bandit_statistics.max_mean = max(mean_parameter)
        self.bandit_statistics.max_mean_positions = [
            index
            for index, value in enumerate(mean_parameter)
            if value == self.bandit_statistics.max_mean
        ]


__all__ = [
    DistParameter.__name__,
    TestBed.__name__,
]

if __name__ == "__main__":
    bernoulli_env = BaseBanditEnv(
        distr_params=DistParameter(
            dist_type=ArmDistTypes.GAUSSIAN,
            mean_parameter=[0.1, 0.2],
            scale_parameter=[1.0, 1.0],
        ),
        max_steps=10,
    )
    testbed_env = TestBed(
        max_steps=10,
        testbed_config=TestBedConfigs(
            type=TestBedSampleType.BERNOULLI,
            sample_config={"n": 1, "p": 0.4},
            no_arms=5,
            arm_type=ArmDistTypes.BERNOULLI,
        ),
    )

    for play_action in range(5):
        _, get_reward, _, _ = testbed_env.step(play_action)
        print("Arm", play_action, "gave a reward of:", get_reward)
        print(
            f"optimal action was {testbed_env.bandit_statistics.played_optimal} times played"
        )
        print(f"the new regret is {testbed_env.bandit_statistics.regret}")
