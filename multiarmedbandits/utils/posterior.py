""" Sampling from a posterior distribution.
"""
from typing import Any, Dict
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import beta
from strenum import StrEnum


class PriorType(StrEnum):
    """
    Specify different types of prior distributions:
        - the beta distribution is conjugate to the bernoulli likelihood
        - the normal distribution is conjugate to a normal distribution with known variance
        - the normal-inverse-gamma is conjugate to a normal distribution without known variance
    """
    BETA = "beta"
    NORMAL = "normal"
    NIG = "normal-inverse-gamma"


class AbstractPosterior(ABC):
    """
    This implements a general posterior distribution.
    """
    @abstractmethod
    def __init__(self, config) -> None:
        """
        We generally need a config to initialize the parameters of our posterior.
        """
    @abstractmethod
    def sample(self):
        """
        Sample from the posterior distribution.
        """
    @abstractmethod
    def update(self, action, reward) -> None:
        """
        Update our posterior distribution given our knowledge over reward from current action.
        """
    @abstractmethod
    def reset(self) -> None:
        """
        Reset our posterior distribution.
        """


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
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        # Initialize the beta parameter of the Beta distribution for each arm
        if "beta" in self.config:
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)

    def sample(self) -> np.ndarray:
        return beta.rvs(self.alpha, self.beta) # Samples from the Beta distribution for each arm

    def update(self, action: int, reward: int) -> None:
        # Update the alpha or beta parameter for the selected arm, based on the received reward
        if reward:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def reset(self) -> None:
        # Reset the alpha parameter of the Beta distribution for each arm
        if "alpha" in self.config:
            self.alpha = np.array(self.config["alpha"])
        else:
            self.alpha = np.ones(self.n_arms)
        # Reset the beta parameter of the Beta distribution for each arm
        if "beta" in self.config:
            self.beta = np.array(self.config["beta"])
        else:
            self.beta = np.ones(self.n_arms)

class PosteriorFactory():
    """
    Returns an a posteriori distribution object.
    """
    def __init__(self, bandit) -> None:
        self.n_arms = bandit.n_arms
        # Could be used for check whether prior/posterior makes sense for dist_type
        self.dist_type =  bandit.distr_params.dist_type

    def create(self, config: Dict[str, Any]):
        """
        Return a posterior distribution
        """
        assert "prior" in config, "You have to provide a prior."
        prior = config["prior"]
        assert isinstance(prior, PriorType), "Provided prior is not known."
        if prior == PriorType.BETA:
            return BetaPosterior(self.n_arms, config=config)
        if prior == PriorType.NORMAL:
            raise NotImplementedError("This prior is not yet implemented.")
        if prior == PriorType.NIG:
            raise NotImplementedError("This prior is not yet implemented.")
        