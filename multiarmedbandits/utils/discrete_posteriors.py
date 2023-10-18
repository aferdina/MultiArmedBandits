"""
Collection of discrete posterior distributions.
"""
from typing import Any, Dict

import numpy as np
from scipy.stats import beta

from multiarmedbandits.utils.abstract_posterior import AbstractPosterior


class BetaPosterior(AbstractPosterior):
    """
    Class for the posterior of a prior beta distribution (which is again beta distributed).

    Per default we choose the Bayes prior Beta(1,1) for every arm.
    It is mathematically equivalent to a uniform distribution.
    """

    def __init__(self, n_arms: int, config: Dict[str, Any]) -> None:
        self.config = config
        self.n_arms = n_arms
        # Initialize the alpha parameter of the Beta distribution for each arm
        if "alpha" in self.config:
            self.check_len_params(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        # Initialize the beta parameter of the Beta distribution for each arm
        if "beta" in self.config:
            self.check_len_params(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        return beta.rvs(self.alpha, self.beta)  # Samples from the Beta distribution for each arm

    def update(self, action: int, reward: int) -> None:
        # Update the alpha or beta parameter for the selected arm, based on the received reward
        if reward:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def reset(self) -> None:
        # Reset the alpha parameter of the Beta distribution for each arm
        if "alpha" in self.config:
            self.check_len_params(self.config["alpha"], self.n_arms)
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        # Reset the beta parameter of the Beta distribution for each arm
        if "beta" in self.config:
            self.check_len_params(self.config["beta"], self.n_arms)
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)
