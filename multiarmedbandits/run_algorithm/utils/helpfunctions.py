""" helpmodules and function for running multiarmed bandit models
"""
import math
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from strenum import StrEnum


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


def plot_statistics(metrics: MABMetrics, metrics_to_plot: list[MetricNames], title: str = "") -> None:
    """plot metrics from running multiarmed agent module

    Args:
        metrics (MABMetrics): metrics to plot
        metrics_to_plot (list[MetricNames]): list of keys to plot
    """
    no_of_metrics = len(metrics_to_plot)
    _rows_square, rows = next_square(number=no_of_metrics)
    cols = rows if rows * (rows - 1) < no_of_metrics else rows - 1
    fig, axs = plt.subplots(cols, rows, figsize=(10, 8))
    fig.suptitle(title, fontsize=16)
    pos = 0
    index_array = np.arange(metrics.horizon)
    if axs.ndim == 1:
        for row in range(rows):
            axs[row].plot(index_array, getattr(metrics, metrics_to_plot[pos]), color="red")
            axs[row].set_title(f"{metrics_to_plot[pos]}")
            pos += 1
        plt.show()
    else:
        for row in range(rows):
            for col in range(cols):
                if pos < no_of_metrics:
                    axis = axs[row, col]
                    axis.plot(
                        index_array,
                        getattr(metrics, metrics_to_plot[pos]),
                    )
                    axis.set_title(f"{metrics_to_plot[pos]}")
                    pos += 1

        plt.show()


def next_square(number: int) -> Tuple[int, int]:
    """get next square of a number

    Args:
        number (int): number to check

    Returns:
        int: next square of a number
    """
    square_root = math.sqrt(number)
    next_integer = math.ceil(square_root)
    next_square_number = next_integer**2
    return next_square_number, next_integer


__all__ = [
    MetricNames.__name__,
    MABMetrics.__name__,
    plot_statistics.__name__,
    next_square.__name__,
]
