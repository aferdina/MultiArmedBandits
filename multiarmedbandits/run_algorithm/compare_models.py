""" run multiple multi armed bandit models
"""
import os
from typing import Any, List
from dataclasses import dataclass
from strenum import StrEnum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from multiarmedbandits.environments import (
    BaseBanditEnv,
    INFODICT,
)
from multiarmedbandits.run_algorithm.utils import (
    plot_statistics,
    MetricNames,
    MABMetrics,
    next_square,
)
import multiarmedbandits.algorithms.multiarmed_bandit_models as bandit_algos

COMPARISON_TITLE = "multiarmed bandit comparisons"
INDEX_AXIS = "Timesteps"


class RunMultiarmedBanditModel:
    """run algorithm for mab on an environment"""

    def __init__(self, mab_algo: bandit_algos.BaseModel, bandit_env: BaseBanditEnv):
        self.mab_algo = mab_algo
        self.bandit_env = bandit_env
        self.metrics: MABMetrics = MABMetrics(horizon=bandit_env.max_steps)
        self.total_runs: int = 0

    def reset_statistics(self) -> None:
        """reset statistics for running a mulitarmed bandit model on a given environment"""
        self.metrics = MABMetrics(horizon=self.bandit_env.max_steps)
        self.total_runs = 0

    def plot_mab_statistics(self, metrics_to_plot: List[MetricNames]) -> None:
        """plot current statistics for a multiarmed bandit model on a given environment

        Args:
            metrics_to_plot (List[MetricNames]): List of metric names which should be plotted
        """
        plot_statistics(
            metrics=self.metrics,
            metrics_to_plot=metrics_to_plot,
            title=f"{self.mab_algo!s}",
        )

    def update_metrics(self, metrics_to_update: MABMetrics) -> None:
        """update metrics for a multiarmed bandit model

        Args:
            metrics_to_update (MABMetrics): metrics using for update
        """
        self.metrics = self.metrics + metrics_to_update

    def get_metrics_from_runs(self, no_runs: int) -> MABMetrics:
        """run multi armed bandit model on environment for `no_runs` rounds

        Args:
            no_runs (int): number of rounds to run multi armed bandit model
        """
        rewards = np.zeros(shape=(no_runs, self.bandit_env.max_steps))
        regrets = np.zeros(shape=(no_runs, self.bandit_env.max_steps))
        optimalities = np.zeros(shape=(no_runs, self.bandit_env.max_steps))
        for game in range(no_runs):
            # reset algorithm and bandit
            _new_state, info = self.bandit_env.reset()
            self.mab_algo.reset()
            done = False
            while not done:
                # playing the game until it is done
                action = self.mab_algo.select_arm(
                    arm_attrib=info[INFODICT.ARMATTRIBUTES]
                )
                _new_state, reward, done, info = self.bandit_env.step(action)
                rewards[game, (self.bandit_env.count - 1)] = reward
                regrets[
                    game, (self.bandit_env.count - 1)
                ] = self.bandit_env.bandit_statistics.regret
                optimalities[
                    game, (self.bandit_env.count - 1)
                ] = self.bandit_env.bandit_statistics.played_optimal

                self.mab_algo.update(action, reward)
        # calculate needed metrics
        mean_cum_rewards_over_runs = np.cumsum(np.mean(rewards, axis=0))
        mean_regret_over_runs = np.mean(regrets, axis=0)
        mean_optimalities_over_runs = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities_over_runs)) + 1
        return MABMetrics(
            horizon=self.bandit_env.max_steps,
            no_runs=no_runs,
            regret=mean_regret_over_runs,
            optimalities=mean_optimalities_over_runs,
            optim_percentage=mean_optimalities_over_runs / index_array,
            cum_reward=mean_cum_rewards_over_runs,
            average_reward=mean_cum_rewards_over_runs / index_array,
            regret_convergence=mean_regret_over_runs / index_array,
        )

    def add_runs_to_metrics(self, metrics_to_add: MABMetrics) -> None:
        """add metric to the current metrics

        Args:
            metrics_to_add (MABMetrics): metrics to add
        """
        self.metrics = self.metrics + metrics_to_add


class Algorithms(StrEnum):
    """algorithm to use for mab environments"""

    EPSILONGREEDY = "EpsilonGreedy"
    EXPLORRETHENCOMMIT = "ExploreThenCommit"
    UCBALGO = "UCB"
    BOLTZMANNSIMPLE = "BoltzmannSimple"
    BOLTZMANNRANDOM = "BoltzmannGeneral"
    GRADIENTBANDIT = "GradientBandit"

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


class CompareMultiArmedBandits:
    """compare multiable multiarmed bandit models"""

    def __init__(
        self,
        test_env: BaseBanditEnv,
        mab_algorithms: List[MultiArmedBanditModel],
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
        fig.suptitle(f"{COMPARISON_TITLE}", fontsize=16)
        pos = 0
        axis: Axes
        if axs.ndim == 1:
            for row in range(rows):
                # add multiple plots
                axis = axs[row]
                metric_to_plot = metrics_to_plot[pos]
                for named_metric in named_metrics:
                    self.plot_named_metric(
                        axis=axis,
                        named_metric=named_metric,
                        metric_to_plot=metric_to_plot,
                    )
                pos += 1
            plt.show()
        else:
            for row in range(rows):
                for col in range(cols):
                    if pos < no_of_metrics:
                        axis = axs[row, col]
                        metric_to_plot = metrics_to_plot[pos]
                        for named_metric in named_metrics:
                            self.plot_named_metric(
                                axis=axis,
                                named_metric=named_metric,
                                metric_to_plot=metric_to_plot,
                            )
                        pos += 1

            plt.show()

    @staticmethod
    def plot_named_metric(
        axis: Axes, named_metric: NamedMABMetrics, metric_to_plot: MetricNames
    ) -> None:
        """plot specific metric on axis in matplotlib

        Args:
            axis (Axes): axis to plot metric on
            named_metric (NamedMABMetrics): values to plot
            metric_to_plot (MetricNames): metrics to plot
        """
        index_array = np.arange(named_metric.metrics.horizon)
        axis.plot(
            index_array,
            getattr(named_metric.metrics, metric_to_plot),
            label=f"{named_metric.algorithm.dist_type!s}:{named_metric.algorithm.dist_params}",
        )
        axis.legend()
        axis.set_ylabel(f"{metric_to_plot} over {named_metric.metrics.no_runs} runs")
        axis.set_xlabel(INDEX_AXIS)

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
        _distr = getattr(bandit_algos, mab_algo.dist_type)
        return _distr(bandit_env=test_env, **mab_algo.dist_params)

    @staticmethod
    def store_metric(
        named_metric: NamedMABMetrics,
        file_path: str,
        metrics_to_store: List[MetricNames],
    ) -> None:
        """store named metric as csv

        Args:
            named_metric (NamedMABMetrics): metric to store
        """
        combined_array = np.column_stack(
            tuple(
                getattr(named_metric.metrics, metric_name)
                for metric_name in metrics_to_store
            )
        )
        store_path = os.path.join(
            file_path,
            str(named_metric.algorithm.dist_type),
            str(named_metric.algorithm.dist_params),
        )
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        header = ",".join(metrics_to_store)
        np.savetxt(
            store_path + "/" + "data.csv",
            combined_array,
            delimiter=",",
            header=header,
            comments="",
        )


__all__ = [
    RunMultiarmedBanditModel.__name__,
    Algorithms.__name__,
    MultiArmedBanditModel.__name__,
    NamedMABMetrics.__name__,
    CompareMultiArmedBandits.__name__,
]
