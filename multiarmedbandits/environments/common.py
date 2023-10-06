""" common class for multi armed bandit environments
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from strenum import StrEnum

from multiarmedbandits.utils import is_positive_integer


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

    def sample(self, action) -> float:
        """sample from distribution at position action"""
        if self.dist_type == ArmDistTypes.GAUSSIAN:
            return np.random.normal(
                    loc=self.mean_parameter[action],
                    scale=self.scale_parameter[action],
                    size=None,
                )
        if self.dist_type == ArmDistTypes.BERNOULLI:
            return 1.0 if np.random.uniform() < self.mean_parameter[action] else 0.0
        raise NotImplementedError



class INFODICT(StrEnum):
    """Enum class for information dictionary and multi armed bandit environment"""

    STEPCOUNT = "count"
    REGRET = "regret"
    ARMATTRIBUTES = "arm_attributes"


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


class BaseBanditEnv:
    """class for a basic multiarmed bandit model"""

    def __init__(self, distr_params: DistParameter, max_steps: int) -> None:
        """create a multiarm bandit with `len(distr_params.mean_parameter)` arms

        Args:
            distr_params (DistParameter): dataclass containing distribution parameter
            for arms of multiarm bandit
            max_steps (int): maximal number of steps to play in the multi arm bandit
        """
        assert is_positive_integer(max_steps), "The number of steps should be a positive integer"
        self.n_arms: int = len(distr_params.mean_parameter)
        self.max_steps: int = max_steps
        self.count: int = 0
        self.done: bool = False
        self.distr_params: DistParameter = distr_params
        # maximal mean and position of maximal mean
        mean_parameter = self.distr_params.mean_parameter
        self.bandit_statistics: BanditStatistics = BanditStatistics(
            max_mean=max(mean_parameter),
            max_mean_positions=[index for index, value in enumerate(mean_parameter) 
                                if value == max(mean_parameter)],
        )
        self.bandit_statistics.reset_statistics()


    def get_reward(self, action: int) -> float:
        """get reward for a given action

        Args:
            action (int): action which is played

        Returns:
            float: reward for playing an specific action
        """
        return self.distr_params.sample(action)

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

    def reset(self) -> Tuple[int, dict[str, Any]]:
        """reset all statistics to run a new game

        Returns:
            Tuple[int, dict[str, Any]]: state and information dictionary
        """
        self.count = 0
        self.done = False
        self.bandit_statistics.reset_statistics()
        return (
            0,
            {
                INFODICT.REGRET: 0,
                INFODICT.STEPCOUNT: self.count,
                INFODICT.ARMATTRIBUTES: ArmAttributes(step_in_game=self.count),
            },
        )

