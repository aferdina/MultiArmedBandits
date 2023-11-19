"""
Collection of continuous posterior distributions.
"""
from typing import Any, Dict, List

import numpy as np
from scipy.stats import invgamma, norm

from multiarmedbandits.utils.abstract_posterior import AbstractPosterior


class NormalPosterior(AbstractPosterior):
    """
    Class for the posterior of a prior normal distribution (which is again normal distributed).

    Per default we choose the Bayes prior N(0,1) for every arm.
    """

    def __init__(self, n_arms: int, config: Dict[str, Any], bandit_scale: List[float]) -> None:
        self.config = config
        self.n_arms = n_arms
        # scale parameters of Gaussian bandit (known)
        self.bandit_scale = bandit_scale
        # Initialize the mean parameter of the Normal distribution for each arm
        if "mean" in self.config:
            self.check_len_params(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        # Initialize the scale parameter of the Normal distribution for each arm
        if "scale" in self.config:
            self.check_len_params(self.config["scale"], self.n_arms)
            self.scale = np.array(self.config["scale"])
        else:
            self.scale = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        return norm.rvs(loc=self.mean, scale=self.scale)  # Samples from the normal distribution for each arm

    def update(self, action: int, reward: int) -> None:
        # Update the parameters of the normal distribution for the selected arm, based on the received reward
        old_var = self.scale[action] ** 2
        bandit_var = self.bandit_scale[action] ** 2
        # update mean using old mean and old scale
        self.mean[action] = (self.mean[action] * bandit_var + old_var * reward) / (bandit_var + old_var)
        # update scale using old scale
        self.scale[action] = np.sqrt((bandit_var * old_var) / (bandit_var + old_var)) 

    def reset(self):
        # Reset the mean parameter of the Normal distribution for each arm
        if "mean" in self.config:
            self.check_len_params(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        # Reset the scale parameter of the Normal distribution for each arm
        if "scale" in self.config:
            self.check_len_params(self.config["scale"], self.n_arms)
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
        # Initialize the parameters of the NIG distribution for each arm
        if "mean" in self.config:
            self.check_len_params(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        if "lambda" in self.config:
            self.check_len_params(self.config["lambda"], self.n_arms)
            self.lambda_p = np.array(self.config["lambda"])
        else:
            self.lambda_p = np.ones(self.n_arms)
        if "alpha" in self.config:
            self.check_len_params(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        if "beta" in self.config:
            self.check_len_params(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        # sample variance from an inverse gamma distribution with parameters alpha and beta
        variances = invgamma.rvs(a=self.alpha, scale=self.beta)
        # sample from a normal distribution N(mean, variance/lambda_p)
        return norm.rvs(loc=self.mean, scale=np.sqrt(variances/self.lambda_p))

    def update(self, action: int, reward: int) -> None:
        # Update the parameters of the NIG distribution for the selected arm, based on the received reward
        # update beta using old lambda_p and old mean
        self.beta[action] = self.beta[action] + 0.5 * ((self.mean[action] - reward)**2) / (self.lambda_p[action] + 1)
        # update mean using old lambda_p
        self.mean[action] = (self.mean[action] + self.lambda_p[action] * reward) / (self.lambda_p[action] + 1)
        # update lambda
        self.lambda_p[action] = self.lambda_p[action] / (self.lambda_p[action] + 1)
        # update alpha
        self.alpha[action] = self.alpha[action] + 0.5

    def reset(self):
        # Reset the parameters of the NIG distribution for each arm
        if "mean" in self.config:
            self.check_len_params(self.config["mean"], self.n_arms)
            self.mean = np.array(self.config["mean"])
        else:
            self.mean = np.zeros(self.n_arms)
        if "lambda" in self.config:
            self.check_len_params(self.config["lambda"], self.n_arms)
            self.lambda_p = np.array(self.config["lambda"])
        else:
            self.lambda_p = np.ones(self.n_arms)
        if "alpha" in self.config:
            self.check_len_params(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        if "beta" in self.config:
            self.check_len_params(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)
