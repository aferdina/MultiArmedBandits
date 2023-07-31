""" Include all game environments for multi armed bandits
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from abc import ABC, abstractmethod
from strenum import StrEnum
import numpy as np
from multiarmedbandits.utils import (
    is_list_of_positive_floats,
    is_positive_integer,
    check_floats_between_zero_and_one,
)


class INFODICT(StrEnum):
    """Enum class for infromation dictionary"""

    STEPCOUNT = "count"
    REGRET = "regret"


@dataclass
class DistParameter:
    """distribution parameter for arms in multiarmed bandit problems"""

    mean_parameter: list[float]
    scale_parameter: list[float] | None = None


@dataclass
class BanditStatistics:
    """statistics for bandit models"""

    max_mean: float  # maximal mean
    max_mean_position: int  # position of the maximal mean
    played_optimal: int = 0  # count number of times optimal played
    regret: float = 0.0  # calculate regret

    def reset_statistics(self) -> None:
        """reset statistics"""
        self.played_optimal = 0
        self.regret = 0.0


class BaseBanditEnv(ABC):
    """class for a basic multiarmed bandit model"""

    def __init__(self, distr_params: DistParameter, max_steps: int) -> None:
        """create a multiarm bandit with `len(distr_params.mean_parameter)` arms

        Args:
            distr_params (DistParameter): dataclass containing distribution parameter
            for arms of multiarm bandit
            max_steps (int): maximal number of steps to play in the multi arm bandit
        """

        assert is_positive_integer(
            max_steps
        ), "The number of steps should be a positive integer"
        self.n_arms: int = len(distr_params.mean_parameter)
        self.max_steps: int = max_steps
        self.count: int = 0
        self.done: bool = False
        self.mean_parameter: list[float] = distr_params.mean_parameter
        # maximal mean and position of maximal mean
        self.bandit_statistics: BanditStatistics = BanditStatistics(
            max_mean=max(self.mean_parameter),
            max_mean_position=self.mean_parameter.index(max(self.mean_parameter)),
        )
        self.bandit_statistics.reset_statistics()

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
        if action == self.bandit_statistics.max_mean_position:
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
            },
        )

    def reset(self) -> None:
        """reset all statistics to run a new game"""
        self.count = 0
        self.done = False
        self.bandit_statistics.reset_statistics()

    @abstractmethod
    def get_reward(self, action: int) -> float:
        """get reward for playing a given action

        Args:
            action (int):action to play

        Returns:
            float: reward for playing a given action
        """


class GaussianBanditEnv(BaseBanditEnv):
    """class for creating gaussian bandit"""

    def __init__(self, distr_params: DistParameter, max_steps: int) -> None:
        """create a multiarm bandit with `len(p_parameter)` arms

        Args:
            mean_parameter (list): list containing mean parameter of guassian bandit arms
            max_steps (int): number of total steps for the bandit problem
        """
        super().__init__(distr_params=distr_params, max_steps=max_steps)
        assert is_list_of_positive_floats(
            input_list=distr_params.scale_parameter
        ), "scale parameter should be a list of positive floats"
        self.scale_parameter = distr_params.scale_parameter

    def get_reward(self, action: int) -> float:
        """receive gaussian reward for playing a given action

        Args:
            action (int): action to play

        Returns:
            float: reard for playing a specific action
        """
        return np.random.normal(
            loc=self.mean_parameter[action],
            scale=self.scale_parameter[action],
            size=None,
        )


class BernoulliBanditEnv(BaseBanditEnv):
    """Bernoulli game environment from lecture"""

    def __init__(self, distr_params: DistParameter, max_steps: int):
        super().__init__(distr_params=distr_params, max_steps=max_steps)
        assert check_floats_between_zero_and_one(
            self.mean_parameter
        ), "mean parameter has to be a list of floats between zero and one"

    def get_reward(self, action: int) -> float:
        """receive bernoulli reward for playing a given action

        Args:
            action (int): action to play

        Returns:
            float: reard for playing a specific action
        """
        reward = 1.0 if np.random.uniform() < self.mean_parameter[action] else 0.0
        return reward


__all__ = [
    DistParameter.__name__,
    GaussianBanditEnv.__name__,
    BernoulliBanditEnv.__name__,
]
