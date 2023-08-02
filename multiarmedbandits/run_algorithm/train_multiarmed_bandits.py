""" train/run multiarmed algorithms for an environment
"""
from strenum import StrEnum
from dataclasses import dataclass
import numpy as np
from multiarmedbandits.run_algorithm.utils.helpfunctions import plot_statistics
from multiarmedbandits.algorithms.multiarmed_bandit_models import BaseModel
from multiarmedbandits.environments.multiarmed_env import BaseBanditEnv


class MetricNames(StrEnum):
    REGRET = "regret"
    OPTIM_PERC = "optimality_percentage"
    CONVERGENCE_RATE = "convergence_rate"
    CUMULATIVE_REWARD = "cumulative_reward"
    EXPLORATION_EXPLOITATION = "exploration_exploitation_tradeoff"
    AVERAGE_REWARD = "average_reward"


@dataclass
class MABMetrics:
    regret: int = 0


def train_multiarmed(
    mab_algo: BaseModel,
    bandit_env: BaseBanditEnv,
    num_games: int,
    parameter: str,
    printed: bool,
):
    """train/run multiarmed model on a game environment

    Args:
        agent (obj): agent, multiarmed model
        env (obj): game environment
        num_games (int): number of games to play
        parameter (str): string with name of releveant parameter
        printed (bool): bool if metrics should be printed

    Returns:
        list: list including all relevant metrics
    """

    chosen_arms = np.zeros(shape=(num_games, bandit_env.max_steps))
    rewards = np.zeros(shape=(num_games, bandit_env.max_steps))
    regrets = np.zeros(shape=(num_games, bandit_env.max_steps))
    optimalities = np.zeros(shape=(num_games, bandit_env.max_steps))

    for game in range(num_games):
        # playing the algo for `num_games` rounds
        mab_algo.reset()
        bandit_env.reset()
        done = False
        while not done:
            # playing the game until it is done
            action = mab_algo.select_arm()
            _new_state, reward, done, _info = bandit_env.step(action)
            rewards[game, (bandit_env.count - 1)] = reward
            chosen_arms[game, (bandit_env.count - 1)] = action
            regrets[game, (bandit_env.count - 1)] = bandit_env.bandit_statistics.regret
            optimalities[
                game, (bandit_env.count - 1)
            ] = bandit_env.bandit_statistics.played_optimal

            mab_algo.update(action, reward)

    if printed:
        plot_statistics(
            prin_rewards=rewards,
            prin_chosen_arms=chosen_arms,
            prin_regrets=regrets,
            prin_optimalities=optimalities,
            parameter=getattr(mab_algo, parameter),
            name=parameter,
        )

    return rewards, chosen_arms, regrets, optimalities


def train_multiarmedrandom(agent, env, num_games, parameter, printed):
    """train/run multiarmed model on a game environment

    Args:
        agent (obj): agent, multiarmed model
        env (obj): game environment
        num_games (int): number of games to play
        parameter (str): string with name of releveant parameter
        printed (bool): bool if metrics should be printed

    Returns:
        list: list including all relevant metrics
    """

    chosen_arms = np.zeros(shape=(num_games, env.max_steps))
    rewards = np.zeros(shape=(num_games, env.max_steps))
    regrets = np.zeros(shape=(num_games, env.max_steps))
    optimalities = np.zeros(shape=(num_games, env.max_steps))

    for game in range(num_games):
        # playing the algo for `num_games` rounds
        agent.reset()
        env.reset()
        done = False
        while not done:
            # playing the game until it is done
            action = agent.select_arm(env.count)
            _new_state, reward, done, _info = env.step(action)
            rewards[game, (env.count - 1)] = reward
            chosen_arms[game, (env.count - 1)] = action
            regrets[game, (env.count - 1)] = env.regret
            optimalities[game, (env.count - 1)] = env.played_optimal

            agent.update(action, reward)

    if printed:
        plot_statistics(
            prin_rewards=rewards,
            prin_chosen_arms=chosen_arms,
            prin_regrets=regrets,
            prin_optimalities=optimalities,
            parameter=getattr(agent, parameter),
            name=parameter,
        )

    return rewards, chosen_arms, regrets, optimalities
