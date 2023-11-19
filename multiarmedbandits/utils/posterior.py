""" Sampling from a posterior distribution.
"""
from typing import Any, Dict

from strenum import StrEnum

from multiarmedbandits.environments import ArmDistTypes, BaseBanditEnv
from multiarmedbandits.utils.continuous_posteriors import GammaContinuousPosterior, NIGPosterior, NormalPosterior
from multiarmedbandits.utils.discrete_posteriors import BetaPosterior, GammaDiscretePosterior


class PriorType(StrEnum):
    """
    Specify different types of prior distributions:
        - the beta distribution is conjugate to the bernoulli likelihood
        - the gamma distribution is conjugate to the poisson likelihood
        - the normal distribution is conjugate to a normal distribution with known variance
        - the normal-inverse-gamma is conjugate to a normal distribution with unknown variance
    """

    BETA = "beta"
    GAMMA_DISC = "gamma_disc"
    GAMMA_CONT = "gamma_cont"
    NORMAL = "normal"
    NIG = "normal-inverse-gamma"


class PosteriorFactory:
    """
    Returns an a posteriori distribution object.
    """

    def __init__(self, bandit: BaseBanditEnv) -> None:
        self.n_arms = bandit.n_arms
        self.bandit_parameters = bandit.distr_params

    def create(self, config: Dict[str, Any]):
        """
        Return a posterior distribution
        """
        assert "prior" in config, "You have to provide a prior."
        prior = config["prior"]
        if prior == PriorType.BETA:
            assert self.bandit_parameters.dist_type == ArmDistTypes.BERNOULLI, "Bandit is not a Bernoulli bandit."
            return BetaPosterior(self.n_arms, config=config)
        if prior == PriorType.GAMMA_DISC:
            assert self.bandit_parameters.dist_type == ArmDistTypes.POISSON, "Bandit is not a Poisson bandit."
            return GammaDiscretePosterior(self.n_arms, config=config)
        if prior == PriorType.GAMMA_CONT:
            assert self.bandit_parameters.dist_type == ArmDistTypes.EXPONENTIAL, "Bandit is not an Exponential bandit."
            return GammaContinuousPosterior(self.n_arms, config=config)
        if prior == PriorType.NORMAL:
            assert self.bandit_parameters.dist_type == ArmDistTypes.GAUSSIAN, "Bandit is not a Gaussian bandit."
            bandit_scale = self.bandit_parameters.scale_parameter
            return NormalPosterior(n_arms=self.n_arms, config=config, bandit_scale=bandit_scale)
        if prior == PriorType.NIG:
            assert self.bandit_parameters.dist_type == ArmDistTypes.GAUSSIAN, "Bandit is not a Gaussian bandit."
            return NIGPosterior(n_arms=self.n_arms, config=config)
        else:
            raise AssertionError("Provided prior is not known.")
