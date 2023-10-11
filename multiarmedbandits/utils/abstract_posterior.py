"""
Implement an abstract posterior distribution.
"""
from abc import ABC, abstractmethod


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

    @staticmethod
    def check_len_params(parameters, n_arms):
        """
        Check whether some parameters a length equal to the number of arms.
        """
        assert len(parameters) == n_arms, f"There have to be {n_arms} initial parameter values."
