""" train/run multiarmed algorithms for an environment
"""
from typing import List
import numpy as np
from multiarmedbandits.run_algorithm.utils import (
    plot_statistics,
    MetricNames,
    MABMetrics,
)
from multiarmedbandits.algorithms.multiarmed_bandit_models import BaseModel
from multiarmedbandits.environments.multiarmed_env import BaseBanditEnv, INFODICT


class RunMultiarmedBanditModel:
    """run algorithm for mab on an environment"""

    def __init__(self, mab_algo: BaseModel, bandit_env: BaseBanditEnv):
        self.mab_algo = mab_algo
        self.bandit_env = bandit_env
        self.metrics: MABMetrics = MABMetrics(horizon=bandit_env.max_steps)
        self.total_runs = 0

    def reset_statistics(self) -> None:
        """reset statistics for running a mulitarmed bandit model on a given environment"""
        self.metrics = MABMetrics(horizon=self.bandit_env.max_steps)
        self.total_runs = 0

    def plot_mab_statistics(self, metrics_to_plot: List[MetricNames]) -> None:
        """plot current statistics for a multiarmed bandit model on a given environment

        Args:
            metrics_to_plot (List[MetricNames]): List of metric names which should be plotted
        """
        plot_statistics(self.metrics, metrics_to_plot=metrics_to_plot)

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
        mean_rewards_over_runs = np.mean(rewards, axis=0)
        mean_cum_rewards_over_runs = np.cumsum(mean_rewards_over_runs)
        mean_regret_over_runs = np.mean(regrets, axis=0)
        mean_optimalities_over_runs = np.mean(optimalities, axis=0)
        index_array = np.arange(len(mean_optimalities_over_runs)) + 1
        optimalities_percentage_over_runs = mean_optimalities_over_runs / index_array
        statistics = MABMetrics(
            horizon=self.bandit_env.max_steps,
            no_runs=no_runs,
            regret=mean_regret_over_runs,
            optimalities=mean_optimalities_over_runs,
            optim_percentage=optimalities_percentage_over_runs,
            cum_reward=mean_cum_rewards_over_runs,
            average_reward=mean_cum_rewards_over_runs / index_array,
            regret_convergence=mean_regret_over_runs / index_array,
        )
        return statistics

    def add_runs_to_metrics(self, metrics_to_add: MABMetrics) -> None:
        """add metric to the current metrics

        Args:
            metrics_to_add (MABMetrics): metrics to add
        """
        self.metrics = self.metrics + metrics_to_add


# def train_multiarmedrandom(agent, env, num_games, parameter, printed):
#     """train/run multiarmed model on a game environment

#     Args:
#         agent (obj): agent, multiarmed model
#         env (obj): game environment
#         num_games (int): number of games to play
#         parameter (str): string with name of releveant parameter
#         printed (bool): bool if metrics should be printed

#     Returns:
#         list: list including all relevant metrics
#     """

#     chosen_arms = np.zeros(shape=(num_games, env.max_steps))
#     rewards = np.zeros(shape=(num_games, env.max_steps))
#     regrets = np.zeros(shape=(num_games, env.max_steps))
#     optimalities = np.zeros(shape=(num_games, env.max_steps))

#     for game in range(num_games):
#         # playing the algo for `num_games` rounds
#         agent.reset()
#         env.reset()
#         done = False
#         while not done:
#             # playing the game until it is done
#             action = agent.select_arm(env.count)
#             _new_state, reward, done, _info = env.step(action)
#             rewards[game, (env.count - 1)] = reward
#             chosen_arms[game, (env.count - 1)] = action
#             regrets[game, (env.count - 1)] = env.regret
#             optimalities[game, (env.count - 1)] = env.played_optimal

#             agent.update(action, reward)

#     if printed:
#         plot_statistics(
#             prin_rewards=rewards,
#             prin_chosen_arms=chosen_arms,
#             prin_regrets=regrets,
#             prin_optimalities=optimalities,
#             parameter=getattr(agent, parameter),
#             name=parameter,
#         )

#     return rewards, chosen_arms, regrets, optimalities
