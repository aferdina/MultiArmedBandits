""" run multiple multi armed bandit models
"""
from typing import Any
from dataclasses import dataclass
from typing import List
from enum import Enum
from multiarmedbandits.environments import BaseBanditEnv
from multiarmedbandits.algorithms.multiarmed_bandit_models import EpsilonGreedy
from multiarmedbandits.run_algorithm.train_multiarmed_bandits import (
    RunMultiarmedBanditModel,
)
import numpy as np
import matplotlib.pyplot as plt
from multiarmedbandits.run_algorithm.utils import (
    plot_statistics,
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

    algorithm: Algorithms
    metrics: MABMetrics


class CompareMultiArmedBandits:
    """compare multiable multiarmed bandit models"""

    def __init__(
        self, test_env: BaseBanditEnv, mab_algorithm: List[MultiArmedBanditModel]
    ):
        self.mab_env = test_env
        self.mab_algorithm = mab_algorithm

    def train_all_models(self) -> List[NamedMABMetrics]:
        pass

    def plot_multiple_mabs(self, metrics_to_plot: List[MetricNames]):
        pass


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
