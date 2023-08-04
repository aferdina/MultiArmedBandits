""" run multiple multi armed bandit models
"""
from typing import Any
from dataclasses import dataclass
from typing import List
from enum import Enum
from multiarmedbandits.environments import BaseBanditEnv
from multiarmedbandits.algorithms.multiarmed_bandit_models import (
    EpsilonGreedy,
    BaseModel,
)
from multiarmedbandits.run_algorithm.train_multiarmed_bandits import (
    RunMultiarmedBanditModel,
)
import numpy as np
import matplotlib.pyplot as plt
from multiarmedbandits.run_algorithm.utils import (
    plot_statistics,
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

    EPSILONGREEDY = EpsilonGreedy


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
                metrics=train_algo.get_metrics_from_runs(no_of_runs=no_of_runs),
            )
            named_mabs_metrics.append(named_metrics)
        return named_mabs_metrics

    def plot_multiple_mabs(
        self, metrics: List[NamedMABMetrics], metrics_to_plot: List[MetricNames]
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
        index_array = np.arange(metrics[0].metrics.horizon)
        if axs.ndim == 1:
            for row in range(rows):
                # add multiple plots
                axs[row].plot(
                    index_array, getattr(metrics, metrics_to_plot[pos]), color="red"
                )
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

    def get_mab_algo(
        self, test_env: BaseBanditEnv, mab_algo: MultiArmedBanditModel
    ) -> BaseModel:
        """create instance of mab algorithm from multi armed bandit model and algo configs

        Args:
            test_env (BaseBanditEnv): bandit model to use
            mab_algo (MultiArmedBanditModel): configs from multi armed algorithm

        Returns:
            BaseModel: instance of mab algorithm
        """
        return mab_algo.dist_type.value(bandit_env=test_env, **mab_algo.dist_params)


def epsilon_greedy_exp(max_steps, n_arms, used_epsilons, num_games, printed):
    statistics_mean = {}
    statistics_cumsum = {}
    statistics_regrets = {}
    statistics_optimalities = {}

    for epsilon in used_epsilons:
        agent = EpsilonGreedy(epsilon=epsilon, n_arms=n_arms)
        rewards = np.zeros(shape=(num_games, max_steps))
        regrets = np.zeros(shape=(num_games, max_steps))
        optimalities = np.zeros(shape=(num_games, max_steps))
        for game in range(num_games):
            mean_parameter = np.random.normal(loc=0.0, scale=1.0, size=n_arms).tolist()
            env = GaussianBanditEnv(mean_parameter=mean_parameter, max_steps=max_steps)
            agent.reset()
            reward, _chosen_arms, regret, optimality = train_multiarmed(
                agent=agent, env=env, num_games=1, parameter="epsilon", printed=False
            )
            rewards[game,] = reward
            regrets[game,] = regret
            optimalities[game,] = optimality

        mean_rewards = np.mean(rewards, axis=0)
        mean_cum_rewards = np.cumsum(mean_rewards)
        mean_regrets = np.mean(regrets, axis=0)
        mean_optimalities = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities))
        mean_optimalities = mean_optimalities / (index_array + 1)

        statistics_mean[str(epsilon)] = mean_rewards
        statistics_cumsum[str(epsilon)] = mean_cum_rewards
        statistics_regrets[str(epsilon)] = mean_regrets
        statistics_optimalities[str(epsilon)] = mean_optimalities

        # print statistics in console
        print(50 * "*")
        print(f"total mean reward with epsilon= {epsilon} is {mean_cum_rewards[-1]}")
        print(f"total regret with epsilon= {epsilon} is {mean_regrets[-1]}")
        print(f"total optimality with epsilon= {epsilon} is {mean_optimalities[-1]}")
        print(50 * "*")

    if printed:
        plt.subplot(4, 1, 1)
        for used_epsi, traj in statistics_mean.items():
            plt.plot(range(len(traj)), traj, label=f"mean reward, epsilon {used_epsi}")
            plt.legend()
        plt.subplot(4, 1, 2)
        for used_epsi, traj in statistics_cumsum.items():
            plt.plot(range(len(traj)), traj, label=f"cumsum reward, ep {used_epsi}")
            plt.legend()
        plt.subplot(4, 1, 3)
        for used_epsi, traj in statistics_regrets.items():
            plt.plot(range(len(traj)), traj, label=f"regrets, ep {used_epsi}")
            plt.legend()
        plt.subplot(4, 1, 4)
        for used_epsi, traj in statistics_optimalities.items():
            plt.plot(range(len(traj)), traj, label=f"optimalities, ep {used_epsi}")
            plt.legend()
        plt.show()

    return (
        statistics_mean,
        statistics_cumsum,
        statistics_regrets,
        statistics_optimalities,
    )


if __name__ == "__main__":
    epsilon_greedy_exp(
        max_steps=MAX_STEPS,
        n_arms=N_ARMS,
        used_epsilons=USED_EPSILONS,
        num_games=NUM_GAMES,
        printed=True,
    )
