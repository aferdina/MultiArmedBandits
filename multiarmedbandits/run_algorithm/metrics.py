""" define metrics for multiarmed bandit algorithms
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from strenum import StrEnum

from ..algorithms import (
    UCB,
    BoltzmannGeneral,
    BoltzmannSimple,
    EpsilonGreedy,
    ExploreThenCommit,
    GradientBandit,
    ThompsonSampling,
)


class MetricNames(StrEnum):
    """enum class for metric names"""

    REGRET = "regret"
    OPTIMALITIES = "optimalities"
    OPTIM_PERCENTAGE = "optim_percentage"
    CUMULATIVE_REWARD = "cum_reward"
    EXPLORATION_EXPLOITATION = "exploration_exploitation_tradeoff"
    AVERAGE_REWARD = "average_reward"
    REGRETCONVERGENCE = "regret_convergence"


@dataclass
class MABMetrics:
    """store a desired metrics for running a multiarmed bandit"""

    horizon: int
    no_runs: int = 0
    regret: np.ndarray | None = None
    optimalities: np.ndarray | None = None
    optim_percentage: np.ndarray | None = None
    cum_reward: np.ndarray | None = None
    average_reward: np.ndarray | None = None
    regret_convergence: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.regret is None:
            self.regret = np.zeros(self.horizon)
        if self.optimalities is None:
            self.optimalities = np.zeros(self.horizon)
        if self.optim_percentage is None:
            self.optim_percentage = np.zeros(self.horizon)
        if self.cum_reward is None:
            self.cum_reward = np.zeros(self.horizon)
        if self.average_reward is None:
            self.average_reward = np.zeros(self.horizon)
        if self.regret_convergence is None:
            self.regret_convergence = np.zeros(self.horizon)

    def __add__(self, other: "MABMetrics") -> "MABMetrics":
        """add to metrics together by calculating new average

        Args:
            other (MABMetrics): metrics to add

        Raises:
            TypeError: Error if other is not MABMetrics
            ValueError: Error if horizon is not the same

        Returns:
            MABMetrics: updated metrics
        """
        if not isinstance(other, MABMetrics):
            raise TypeError(f"Unsupported operand type for +: 'MABMetrics' and '{type(other)}'")
        if not self.horizon == other.horizon:
            raise ValueError("Horizon must be the same")
        new_no_runs = self.no_runs + other.no_runs
        new_metric = MABMetrics(horizon=self.horizon, no_runs=new_no_runs)
        for attr_name, attr_value in vars(other).items():
            if attr_name not in ["horizon", "no_runs"]:
                new_value = (other.no_runs * attr_value + getattr(self, attr_name) * self.no_runs) / new_no_runs
                setattr(new_metric, attr_name, new_value)
        return new_metric


class Algorithms(StrEnum):
    """algorithm to use for mab environments"""

    EPSILONGREEDY = EpsilonGreedy.__name__
    EXPLORRETHENCOMMIT = ExploreThenCommit.__name__
    UCBALGO = UCB.__name__
    BOLTZMANNSIMPLE = BoltzmannSimple.__name__
    BOLTZMANNRANDOM = BoltzmannGeneral.__name__
    GRADIENTBANDIT = GradientBandit.__name__
    THOMPSON = ThompsonSampling.__name__

    def __str__(self):
        return self.name.capitalize()


@dataclass
class MultiArmedBanditModel:
    """class to create a multiarmed bandit model"""

    dist_type: Algorithms
    dist_params: dict[str, Any]


@dataclass
class NamedMABMetrics:
    """metrics for a multiarmed bandit model including the algorithm with parameters"""

    algorithm: MultiArmedBanditModel
    metrics: MABMetrics
