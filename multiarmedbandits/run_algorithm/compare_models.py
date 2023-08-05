""" run multiple multi armed bandit models
"""
from typing import Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from multiarmedbandits.environments import (
    BaseBanditEnv,
    DistParameter,
    ArmDistTypes,
    INFODICT,
)
from multiarmedbandits.run_algorithm.utils import (
    plot_statistics,
    MetricNames,
    MABMetrics,
    next_square,
)
import multiarmedbandits.algorithms.multiarmed_bandit_models as bandit_algos

MAX_STEPS = 1000
N_ARMS = 10
USED_EPSILONS = [0.1, 0.2, 0.5]
NUM_GAMES = 3000


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
            title=f"{str(self.mab_algo)}",
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
            _new_state, reward, done, info = self.bandit_env.reset()
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


__all__ = [
    RunMultiarmedBanditModel.__all__,
    Algorithms.__all__,
    MultiArmedBanditModel.__all__,
    NamedMABMetrics.__all__,
    CompareMultiArmedBandits.__all__,
]

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
