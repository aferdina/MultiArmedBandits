"""
Collection of continuous posterior distributions.
"""
from typing import Any, Dict, List

import numpy as np
from scipy.stats import invgamma, norm

from multiarmedbandits.utils.abstract_posterior import AbstractPosterior


def check_length_parameters(parameters, n_arms):
    """
    Check whether parameters have the same length as the arms of the multi-armed bandit.
    """
    assert len(parameters) == n_arms, f"There have to be {n_arms} initial parameter values."


class NormalPosterior(AbstractPosterior):
    """
    Class for the posterior of a prior normal distribution (which is again normal distributed).

    Per default we choose the Bayes prior N(0,1) for every arm.
    """

    def __init__(self, n_arms: int, config: Dict[str, Any], bandit_scale: List[float]) -> None:
        self.config = config
        self.n_arms = n_arms
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        # scale parameters of Gaussian bandit (known)
        self.bandit_scale = bandit_scale
        # Initialize the mean parameter of the Normal distribution for each arm
        if "mean" in self.config:
            check_length_parameters(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        # Initialize the scale parameter of the Normal distribution for each arm
        if "scale" in self.config:
            check_length_parameters(self.config["scale"], self.n_arms)
            self.scale = np.array(self.config["scale"])
        else:
            self.scale = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        return norm.rvs(self.mean, self.scale)  # Samples from the normal distribution for each arm

    def update(self, action: int, reward: int) -> None:
        # Update counts and values (via memory trick)
        self.counts[action] = self.counts[action] + 1
        self.values[action] = self.memory_trick(
            times_played=self.counts[action], old_mean=self.values[action], value_to_add=reward
        )
        # self.values[action] = ((times_played - 1) / times_played) * old_value + (1 / times_played) * reward

        # Update the mean parameter for the selected arm, based on the received reward
        old_var = self.scale[action] ** 2
        bandit_var = self.bandit_scale[action] ** 2
        self.mean[action] = self.update_mean(
            old_mean=self.mean[action],
            old_var=old_var,
            bandit_var=bandit_var,
            new_value=self.values[action],
            times_played=self.counts[action],
        )

        # Update the scale parameter for the selected arm, based on the received reward
        self.scale[action] = self.update_scale(old_var=old_var, bandit_var=bandit_var, times_played=self.counts[action])

    @staticmethod
    def memory_trick(times_played: int, old_mean: float, value_to_add: float) -> float:
        """calculate mean value using memory trick

        Args:
            times_played (int): number of times played
            old_mean (float): old mean from `times_played`-1 values
            value_to_add (float): value to add for the mean

        Returns:
            float: updated mean value
        """
        return ((times_played - 1) / times_played) * old_mean + (1 / times_played) * value_to_add

    @staticmethod
    def update_mean(old_mean, old_var, bandit_var, new_value, times_played) -> float:
        """calculate the updated mean parameter of the posterior distribution

        Args:
            old_mean (float): mean parameter value before update
            old_var (float): variance parameter value before update (scale ** 2)
            bandit_var (float): variance parameter of bandit's arm (scale ** 2)
            new_value (float): updated action value (mean of rewards)
            times_played (int): number of times played

        Returns:
            float: updated mean parameter
        """
        return (old_mean * bandit_var + times_played * old_var * new_value) / (bandit_var + times_played * old_var)

    @staticmethod
    def update_scale(old_var, bandit_var, times_played) -> float:
        """calculate the updated scale parameter of the posterior distribution

        Args:
            old_var (float): variance parameter value before update (scale ** 2)
            bandit_var (float): variance parameter of bandit's arm (scale ** 2)
            times_played (int): number of times played

        Returns:
            float: updated scale parameter
        """
        return np.sqrt((old_var * bandit_var) / (bandit_var + times_played * old_var))

    def reset(self):
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        # Reset the mean parameter of the Normal distribution for each arm
        if "mean" in self.config:
            check_length_parameters(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        # Reset the scale parameter of the Normal distribution for each arm
        if "scale" in self.config:
            check_length_parameters(self.config["scale"], self.n_arms)
            self.scale = np.array(self.config["scale"])
        else:
            self.scale = np.ones(self.n_arms)


class NIGPosterior(AbstractPosterior):
    """
    Class for the posterior of a prior normal-inverse-gamma distribution (which is again normal-inverse-gamma distributed).
    NIG has parameters mean, lambda_p, alpha, beta

    Per default we choose the Bayes prior NIG(mean=0, lambda_p=1, alpha=1, beta=1) for every arm.
    """

    def __init__(self, n_arms: int, config: Dict[str, Any]) -> None:
        self.config = config
        self.n_arms = n_arms
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.sqsum: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)  # sum of squares
        # Initialize the parameters of the NIG distribution for each arm
        if "mean" in self.config:
            check_length_parameters(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        if "lambda" in self.config:
            check_length_parameters(self.config["lambda"], self.n_arms)
            self.lambda_p = np.array(self.config["lambda"])
        else:
            self.lambda_p = np.ones(self.n_arms)
        if "alpha" in self.config:
            check_length_parameters(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        if "beta" in self.config:
            check_length_parameters(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        # sample variance from an inverse gamma distribution with parameters alpha and beta
        variances = invgamma.rvs(self.alpha, scale=self.beta)
        # sample from a normal distribution N(mean, variance/lambda_p)
        return norm.rvs(self.mean, variances)

    def update(self, action: int, reward: int) -> None:
        # Update counts and values (via memory trick)
        self.counts[action] = self.counts[action] + 1
        self.values[action] = self.memory_trick(
            times_played=self.counts[action], old_mean=self.values[action], value_to_add=reward
        )
        # Update sum of squared rewards
        self.sqsum[action] = self.sqsum[action] + reward**2

        # Update the parameters of the NIG distribution for the selected arm, based on the received reward
        old_mean = self.mean[action]
        old_lambda = self.lambda_p[action]
        times_played = self.counts[action]

        # lambda
        self.lambda_p[action] = 1 / ((1 / old_lambda) + times_played)
        new_lambda = self.lambda_p[action]
        # mean
        self.mean[action] = new_lambda * (old_mean / old_lambda + times_played * self.values[action])
        new_mean = self.mean[action]
        # alpha
        self.alpha[action] = self.alpha[action] + times_played / 2
        # beta
        self.beta[action] = self.beta[action] + 1 / 2 * (
            (old_mean**2) / old_lambda + self.sqsum[action] - (new_mean**2) / new_lambda
        )

    @staticmethod
    def memory_trick(times_played: int, old_mean: float, value_to_add: float) -> float:
        """calculate mean value using memory trick

        Args:
            times_played (int): number of times played
            old_mean (float): old mean from `times_played`-1 values
            value_to_add (float): value to add for the mean

        Returns:
            float: updated mean value
        """
        return ((times_played - 1) / times_played) * old_mean + (1 / times_played) * value_to_add

    def reset(self):
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.sqsum: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)  # sum of squares
        # Reset the parameters of the NIG distribution for each arm
        if "mean" in self.config:
            check_length_parameters(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        if "lambda" in self.config:
            check_length_parameters(self.config["lambda"], self.n_arms)
            self.lambda_p = np.array(self.config["lambda"])
        else:
            self.lambda_p = np.ones(self.n_arms)
        if "alpha" in self.config:
            check_length_parameters(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        if "beta" in self.config:
            check_length_parameters(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)
