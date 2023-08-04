""" run multiple multi armed bandit models
"""
from typing import Any, List
from dataclasses import dataclass
from enum import Enum
from multiarmedbandits.environments import BaseBanditEnv, DistParameter, ArmDistTypes
import multiarmedbandits.algorithms.multiarmed_bandit_models as bandit_algos
from multiarmedbandits.run_algorithm.train_multiarmed_bandits import (
    RunMultiarmedBanditModel,
)
import numpy as np
import matplotlib.pyplot as plt
from multiarmedbandits.run_algorithm.utils import (
    next_square,
    MetricNames,
    MABMetrics,
)

MAX_STEPS = 1000
N_ARMS = 10
USED_EPSILONS = [0.1, 0.2, 0.5]
NUM_GAMES = 3000


class Algorithms(Enum):
    """algorithm to use for mab environments"""

    EPSILONGREEDY = bandit_algos.EpsilonGreedy
    EXPLORRETHENCOMMIT = bandit_algos.ExploreThenCommit
    UCBALGO = bandit_algos.UCB
    BOLTZMANNCONSTANT = bandit_algos.BoltzmannConstant
    GRADIENTBANDIT = bandit_algos.GradientBandit
    GRADIENTNOBASELINE = bandit_algos.GradientBanditnobaseline


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


class CompareMultiArmedBandits:
    """compare multiable multiarmed bandit models"""

    def __init__(
        self, test_env: BaseBanditEnv, mab_algorithms: List[MultiArmedBanditModel]
    ):
        self.mab_env = test_env
        self.mab_algorithms = mab_algorithms

    def train_all_models(self, no_of_runs: int) -> List[NamedMABMetrics]:
        """train all models on environment

        Args:
            no_of_runs (int): number of runs to train all algorithms

        Returns:
            List[NamedMABMetrics]: List with named metrics to plot
        """
        named_mabs_metrics = []
        for mab_algorithms in self.mab_algorithms:
            mab_algo_instance = self.get_mab_algo(
                test_env=self.mab_env, mab_algo=mab_algorithms
            )
            train_algo = RunMultiarmedBanditModel(
                mab_algo=mab_algo_instance, bandit_env=self.mab_env
            )
            named_metrics = NamedMABMetrics(
                algorithm=mab_algorithms,
                metrics=train_algo.get_metrics_from_runs(no_runs=no_of_runs),
            )
            named_mabs_metrics.append(named_metrics)
        return named_mabs_metrics

    def plot_multiple_mabs(
        self, named_metrics: List[NamedMABMetrics], metrics_to_plot: List[MetricNames]
    ):
        """plot metrics from running multiarmed agent module

        Args:
            metrics (MABMetrics): metrics to plot
            metrics_to_plot (list[MetricNames]): list of keys to plot
        """
        no_of_metrics = len(metrics_to_plot)
        _rows_square, rows = next_square(number=no_of_metrics)
        cols = rows if rows * (rows - 1) < no_of_metrics else rows - 1
        fig, axs = plt.subplots(cols, rows, figsize=(10, 8))
        fig.suptitle("Comparision", fontsize=16)
        pos = 0
        index_array = np.arange(named_metrics[0].metrics.horizon)
        if axs.ndim == 1:
            for row in range(rows):
                # add multiple plots
                for named_metric in named_metrics:
                    axs[row].plot(
                        index_array,
                        getattr(named_metric.metrics, metrics_to_plot[pos]),
                        label=f"{named_metric.algorithm.dist_type}:{named_metric.algorithm.dist_params}",
                    )
                    axs[row].legend()
                    axs[row].set_title(f"{metrics_to_plot[pos]}")
                pos += 1
            plt.show()
        else:
            for row in range(rows):
                for col in range(cols):
                    if pos < no_of_metrics:
                        axis = axs[row, col]
                        for named_metric in named_metrics:
                            axs[row].plot(
                                index_array,
                                getattr(named_metric.metrics, metrics_to_plot[pos]),
                                label=f"{named_metric.algorithm.dist_type}:{named_metric.algorithm.dist_params}",
                            )
                            axs[row].legend()
                            axis.set_title(f"{metrics_to_plot[pos]}")
                        pos += 1

            plt.show()

    def get_mab_algo(
        self, test_env: BaseBanditEnv, mab_algo: MultiArmedBanditModel
    ) -> bandit_algos.BaseModel:
        """create instance of mab algorithm from multi armed bandit model and algo configs

        Args:
            test_env (BaseBanditEnv): bandit model to use
            mab_algo (MultiArmedBanditModel): configs from multi armed algorithm

        Returns:
            BaseModel: instance of mab algorithm
        """
        return mab_algo.dist_type.value(bandit_env=test_env, **mab_algo.dist_params)


if __name__ == "__main__":
    bandit_env = BaseBanditEnv(
        distr_params=DistParameter(
            dist_type=ArmDistTypes.GAUSSIAN,
            mean_parameter=[0.1, 0.2, 0.3],
            scale_parameter=[1.0, 1.0, 1.0],
        ),
        max_steps=10000,
    )
    algo_one = MultiArmedBanditModel(
        dist_type=Algorithms.EPSILONGREEDY, dist_params={"epsilon": 0.3}
    )
    algo_two = MultiArmedBanditModel(
        dist_type=Algorithms.EPSILONGREEDY, dist_params={"epsilon": 0.7}
    )
    algo_three = MultiArmedBanditModel(
        dist_type=Algorithms.EPSILONGREEDY, dist_params={"epsilon": 0.1}
    )
    algo_four = MultiArmedBanditModel(
        dist_type=Algorithms.BOLTZMANNCONSTANT, dist_params={"temperature": 2.0}
    )
    compare = CompareMultiArmedBandits(
        test_env=bandit_env, mab_algorithms=[algo_one, algo_two, algo_three, algo_four]
    )
    metrics = compare.train_all_models(no_of_runs=100)
    compare.plot_multiple_mabs(
        named_metrics=metrics,
        metrics_to_plot=[MetricNames.AVERAGE_REWARD, MetricNames.OPTIM_PERCENTAGE],
    )
