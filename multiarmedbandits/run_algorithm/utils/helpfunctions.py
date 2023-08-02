""" helpmodules and function for running multiarmed bandit models
"""
from dataclasses import dataclass
import numpy as np
from typing import Any
from strenum import StrEnum
import matplotlib.pyplot as plt


class MetricNames(StrEnum):
    REGRET = "regret"
    OPTIM_PERC = "optimality_percentage"
    CUMULATIVE_REWARD = "cumulative_reward"
    EXPLORATION_EXPLOITATION = "exploration_exploitation_tradeoff"
    AVERAGE_REWARD = "average_reward"
    REGRETCONVERGENCE = "regret_convergence"


@dataclass
class MABMetrics:
    algorithm_name: str
    bandit_parameters: dict[str, Any]
    num_games: int
    regret: np.ndarray
    optimalities: np.ndarray
    optim_percentage: np.ndarray
    cum_reward: np.ndarray
    average_reward: np.ndarray
    regret_convergence: np.ndarray


def plot_statistics(metrics: MABMetrics, metrics_to_plot: list[MetricNames]) -> None:
    """plot metrics from running multiarmed agent module

    Args:
        metrics (MABMetrics): metrics to plot
        metrics_to_plot (list[MetricNames]): list of keys to plot
    """
    plt.subplot(4, 1, 1)
    for i, reward in enumerate(prin_rewards):
        plt.plot(range(len(reward)), reward, label=f"reward {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 2)
    for i, action in enumerate(prin_chosen_arms):
        plt.plot(
            range(len(action)), action, label=f"action sequence {i}, {name} {parameter}"
        )
        plt.legend()

    plt.subplot(4, 1, 3)
    for i, action in enumerate(prin_regrets):
        plt.plot(range(len(action)), action, label=f"regrets {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 4)
    for i, action in enumerate(prin_optimalities):
        plt.plot(
            range(len(action)), action, label=f"optimalities {i}, {name} {parameter}"
        )
        plt.legend()

    plt.show()
