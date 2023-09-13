""" module contains all algorithms for multiarm bandit problems
"""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats

from multiarmedbandits.algorithms.utils import BaseLinesTypes, BoltzmannConfigs, ExplorationType, GradientBaseLineAttr
from multiarmedbandits.environments import ArmAttributes, BaseBanditEnv
from multiarmedbandits.utils import is_float_between_0_and_1, is_list_of_floats, is_positive_float, is_positive_integer

# TODO: devide into different files, for each algorithm one file
@dataclass
class BaseModel(ABC):
    """create a basemodel class for multiarmed bandit models"""

    def __init__(self, bandit_env: BaseBanditEnv) -> None:
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        n_arms = bandit_env.n_arms
        max_steps = bandit_env.max_steps
        assert is_positive_integer(n_arms), f"{n_arms} should be a positive integer"
        assert is_positive_integer(max_steps), f"{n_arms} should be a positive integer"
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float32)

    @abstractmethod
    def select_arm(self, arm_attrib: ArmAttributes | None) -> int:
        """select arm given the specific multiarmed bandit algorithm

        Returns:
            int: arm to play
        """

    def __str__(self) -> str:
        """return name of class as string representation

        Returns:
            str: _description_
        """
        return self.__class__.__name__

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        # increment the chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        times_played_chosen_arm = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # update via memory trick
        self.values[chosen_arm] = self.memory_trick(times_played=times_played_chosen_arm, old_mean=value, value_to_add=reward)

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)

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


class EpsilonGreedy(BaseModel):
    """class for epsilon greedy algorithm"""

    def __init__(self, epsilon: float, bandit_env: BaseBanditEnv) -> None:
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(epsilon), f"{epsilon} should be a float between 0 and 1"
        self.epsilon = epsilon

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select the best arm by using epsilon gready method

        Returns:
            int: best action based on the estimators of the values
        """
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        return random.randrange(self.n_arms)


class ExploreThenCommit(BaseModel):
    """explore then commit algorithm"""

    def __init__(self, explore: int, bandit_env: BaseBanditEnv) -> None:
        """initialize explore then commit algorithm

        Args:
            explore (int): number of steps to explore each arm
            n_arms (int): number of arms in the multi arm bandit
        """
        super().__init__(bandit_env=bandit_env)
        self.explore = explore

    def select_arm(self, arm_attrib: ArmAttributes) -> int:
        """select the best arm given the estimators of the values

        Args:
            arm_attrib (ArmAttributes): step in the game

        Returns:
            int: best action based on the estimators of the values
        """
        if self.explore * self.n_arms < arm_attrib.step_in_game:
            return np.argmax(self.values)
        return arm_attrib.step_in_game % self.n_arms


class UCB(BaseModel):
    """class for ucb algorithm"""

    def __init__(self, delta: float, bandit_env: BaseBanditEnv) -> None:
        """initialize upper confidence bound algorithm

        Args:
            n_arms (int): number of arms in the multiarmed bandit model
            delta (float): delta parameter of ucb algorithm
        """
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(delta), f"{delta} should be a float between 0 and 1"
        self.delta = delta
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)
        self._exploration_factor = np.sqrt(-2 * np.log(self.delta)) * np.ones(self.n_arms, dtype=np.float32)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select the best arm given the value estimators and the ucb bound
        Returns:
            int: best action based on upper confidence bound
        """
        return np.argmax(self.ucb_values)

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the ucb bound of the ucb algorithm

        Args:
            chosen_arm (int): action which was played an should be updated
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        super().update(chosen_arm=chosen_arm, reward=reward)
        # update all arms which are played at least one time
        # # pylint: disable=C0301
        _square_counts = np.sqrt(self.counts)
        bonus = np.divide(
            self._exploration_factor,
            _square_counts,
            out=np.full(self.n_arms, np.inf, dtype=np.float32),
            where=_square_counts != 0,
        )
        self.ucb_values = self.values + bonus

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        super().reset()
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)


class BoltzmannSimple(BaseModel):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(self, boltzmann_configs: BoltzmannConfigs, bandit_env: BaseBanditEnv):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_list_of_floats(boltzmann_configs.some_constant), "The temperature  has to be a positive float"
        assert (
            len(boltzmann_configs.some_constant) == self.n_arms
        ), "temperature parameter should be of same size as number of arms"
        self.some_constant = np.array(boltzmann_configs.some_constant, dtype=np.float32)
        self.calc_betas = self._create_calc_betas(explor_type=boltzmann_configs.explor_type)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """choose an arm from the boltzmann distribution

        Returns:
            int: simulated action
        """
        _betas = self.calc_betas(arm_attrib=arm_attrib)
        _logits = self.values / _betas
        probs = np.exp(_logits) / np.sum(np.exp(_logits))
        return np.random.choice(self.n_arms, p=probs)

    def calc_betas(self, arm_attrib: ArmAttributes | None = None) -> np.ndarray:
        """calculating beta values for boltzmann algorithm

        Args:
            arm_attrib (ArmAttributes): parameter for calculating beta values

        Returns:
            np.ndarray: beta values for algorithm
        """
        return np.ones_like(self.values)

    def _create_calc_betas(self, explor_type: ExplorationType) -> Callable[[ArmAttributes], np.ndarray]:
        """create method to calculate beta values for boltzmann algorithm
        based on configs

        Returns:
            Callable[[ArmAttributes], float]: method to calculate beta values for boltzmann
            algorithm
        """
        if explor_type == ExplorationType.CONSTANT:

            def _calc_betas(arm_attrib: ArmAttributes | None = None) -> np.ndarray:
                return self.some_constant**2

            return _calc_betas

        if explor_type == ExplorationType.SQRT:

            def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                if np.log(1 + arm_attrib.step_in_game) == 0.0:
                    return np.full_like(self.values, np.inf)
                return self.some_constant**2 / np.sqrt(1 + arm_attrib.step_in_game)

            return _calc_betas

        if explor_type == ExplorationType.LOG:

            def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                if np.log(1 + arm_attrib.step_in_game) == 0.0:
                    return np.full_like(self.values, np.inf)
                return self.some_constant**2 / np.log(1 + arm_attrib.step_in_game)

            return _calc_betas
        if explor_type == ExplorationType.UCB:

            def _calc_betas(arm_attrib: ArmAttributes) -> np.ndarray:
                _square_counts = np.sqrt(self.counts)
                result = np.divide(
                    self.some_constant,
                    _square_counts,
                    out=np.zeros_like(self.some_constant),
                    where=_square_counts != 0,
                )
                result = result * np.log(1 + arm_attrib.step_in_game)
                result[result == 0.0] = np.inf
                return result

            return _calc_betas
        if explor_type == ExplorationType.BGE:

            def _calc_betas(arm_attrib: ArmAttributes | None = None) -> np.ndarray:
                _square_counts = np.sqrt(self.counts)
                result = np.divide(
                    self.some_constant,
                    _square_counts,
                    out=np.zeros_like(self.some_constant),
                    where=_square_counts != 0,
                )
                result[result == 0.0] = np.inf
                return result

            return _calc_betas
        raise NotImplementedError(f"{explor_type} not implemented yet")


@dataclass
class RandomVariable:
    """store properties for random variable in boltzmann algorithms"""

    rv_name: str
    rv_param: dict[str, Any]

    def __post_init__(self):
        assert self.rv_name in dir(stats)


class BoltzmannGeneral(BoltzmannSimple):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(
        self,
        boltzmann_configs: BoltzmannConfigs,
        bandit_env: BaseBanditEnv,
        rv_config: RandomVariable,
    ):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env, boltzmann_configs=boltzmann_configs)
        self.random_variables = self.sample_random_variables(rv_config=rv_config)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """get action from boltzmann gumbel paper"""

        random_variables = self.random_variables[arm_attrib.step_in_game,]
        _betas = self.calc_betas(arm_attrib=arm_attrib)
        _used_parameter = self.values + _betas * random_variables
        return int(np.argmax(_used_parameter))

    def sample_random_variables(self, rv_config: RandomVariable) -> np.ndarray:
        """get realization of random variables for algorithm

        Returns:
            np.ndarray: all required random variables
        """
        _dist = getattr(stats, rv_config.rv_name)
        return _dist(**rv_config.rv_param).rvs(size=(self.max_steps, self.n_arms))


class GradientBandit(BaseModel):
    """gradient bandit algorithm"""

    def __init__(
        self,
        alpha: float,
        baseline_attr: GradientBaseLineAttr,
        bandit_env: BaseBanditEnv,
    ) -> None:
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(alpha), "Learning rate has to be a positive float"

        self.alpha: float = alpha
        self.calc_baseline = self._create_calc_baseline(baseline_typ=baseline_attr.type)
        self.baseline_attr = baseline_attr

    def _create_calc_baseline(self, baseline_typ: BaseLinesTypes) -> Callable[[GradientBaseLineAttr], float]:
        """create baseline function for given baseline type

        Args:
            baseline_typ (BaseLinesTypes): _description_
        """
        if baseline_typ == BaseLinesTypes.ZERO:

            def _calc_baseline(baseline_att: GradientBaseLineAttr) -> float:
                return 0.0

            return _calc_baseline
        if baseline_typ == BaseLinesTypes.MEAN:

            def _calc_baseline(baseline_att: GradientBaseLineAttr) -> float:
                return baseline_att.mean_reward

            return _calc_baseline
        raise ValueError("method not implemented")

    def calc_baseline(self, baseline_att: GradientBaseLineAttr) -> float:
        """calculate baseline for gradient algorithm

        Args:
            baseline_att (GradientBaseLineAttr): attributes to calculate baseline

        Returns:
            float: calculated baseline
        """
        return 0.0

    def get_prob(self, action: int) -> float:
        """get probability for a given action

        Args:
            action (int):

        Returns:
            float: probability for a given action
        """
        input_vector = np.exp(self.values)
        return float(input_vector[action] / np.sum(input_vector))

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """choose arm in the gradient bandit algorithmus

        Returns:
            int: sampled action
        """
        input_vector = np.exp(self.values)
        input_vector = input_vector / np.sum(input_vector)
        return np.random.choice(self.n_arms, p=input_vector)

    def update(self, chosen_arm: int, reward: float) -> None:
        """update the value estimators and counts based on the new observed
         reward and played action

        Args:
            chosen_arm (int): action which was played
            reward (float): reward of the multiarmed bandit, based on playing action `chosen_arm`
        """
        action_prob = self.get_prob(chosen_arm)
        # increment the chosen arm
        action_prob_vec = np.array([-1 * action_prob for _ in range(self.n_arms)])
        action_prob_vec[chosen_arm] = 1 - action_prob
        # update via memory trick
        baseline = self.calc_baseline(baseline_att=self.baseline_attr)
        gradients = (self.alpha * (reward - baseline)) * action_prob_vec

        # update values
        self.values = self.values + gradients
        self.baseline_attr.step_count += 1
        # update mean reward
        self.baseline_attr.mean_reward = (
            (self.baseline_attr.step_count - 1) / float(self.baseline_attr.step_count)
        ) * self.baseline_attr.mean_reward + (1 / float(self.baseline_attr.step_count)) * reward

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.baseline_attr.reset()


__all__ = [
    GradientBandit.__name__,
    UCB.__name__,
    ExploreThenCommit.__name__,
    EpsilonGreedy.__name__,
    BaseModel.__name__,
    BoltzmannSimple.__name__,
    BoltzmannGeneral.__name__,
    RandomVariable.__name__,
]
