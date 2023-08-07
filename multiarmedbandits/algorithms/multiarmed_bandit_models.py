""" module contains all algorithms for multiarm bandit problems
"""
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from multiarmedbandits.utils import (
    is_float_between_0_and_1,
    is_positive_integer,
    is_positive_float,
)
from multiarmedbandits.environments import BaseBanditEnv, ArmAttributes


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
        assert is_positive_integer(n_arms), f"{n_arms} should be a positive integer"
        self.n_arms = n_arms
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
        self.values[chosen_arm] = self.memory_trick(
            times_played=times_played_chosen_arm, old_mean=value, value_to_add=reward
        )

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
        return ((times_played - 1) / times_played) * old_mean + (
            1 / times_played
        ) * value_to_add


class EpsilonGreedy(BaseModel):
    """class for epsilon greedy algorithm"""

    def __init__(self, epsilon: float, bandit_env: BaseBanditEnv) -> None:
        """initialize epsilon greedy algorithm

        Args:
            epsilon (float): epsilon parameter for the epsilon greedy algorithm
            n_arms (int): number of possible arms
        """
        super().__init__(bandit_env=bandit_env)
        assert is_float_between_0_and_1(
            epsilon
        ), f"{epsilon} should be a float between 0 and 1"
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
        assert is_float_between_0_and_1(
            delta
        ), f"{delta} should be a float between 0 and 1"
        self.delta = delta
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)

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
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        times_played_chosen_arm = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = (
            (times_played_chosen_arm - 1) / times_played_chosen_arm
        ) * value + (1 / times_played_chosen_arm) * reward
        self.values[chosen_arm] = new_value
        # update all arms which are played at least one time
        # # pylint: disable=C0301
        for arm in [
            arm_index
            for arm_index, already_played in enumerate(self.counts)
            if already_played != 0
        ]:
            bonus = np.sqrt((2 * np.log(1 / self.delta)) / self.counts[arm])
            self.ucb_values[arm] = self.values[arm] + bonus

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.counts = np.zeros(self.n_arms, dtype=np.float32)
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.ucb_values = np.full(self.n_arms, np.inf, dtype=np.float32)


class BoltzmannConstant(BaseModel):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(self, temperature: float, bandit_env: BaseBanditEnv):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(
            temperature
        ), "The temperature  has to be a positive float"
        self.temperature = temperature

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """choose an arm from the boltzmann distribution

        Returns:
            int: simulated action
        """
        canonical_parameter = self.temperature * self.values
        input_vector = np.exp(canonical_parameter)
        probs = (input_vector / np.sum(input_vector)).tolist()
        unif_distr = np.random.rand()
        cum = 0
        position: int = 0  # fallback variable for return
        for position, probability in enumerate(probs):
            cum += probability
            if unif_distr < cum:
                break
        return position


class BoltzmannGumbel(BoltzmannConstant):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(self, temperature: float, bandit_env: BaseBanditEnv):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(temperature=temperature, bandit_env=bandit_env)

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """select action with respect to gumbel trick

        Returns:
            int: returned action
        """
        _parameter = self.temperature * self.values
        gumbel_rvs = np.random.gumbel(loc=0, scale=1, size=self.n_arms)
        return np.argmax(_parameter + gumbel_rvs)


class BoltzmannGumbelRightWay(BaseModel):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(self, some_constant: float, bandit_env: BaseBanditEnv) -> None:
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(
            some_constant
        ), "The some_constant  has to be a positive float"

        self.some_constant = some_constant

    def select_arm(self, arm_attrib: ArmAttributes | None = None) -> int:
        """get action from boltzmann gumbel paper"""

        gumbel_rvs = np.random.gumbel(loc=0.0, scale=1.0, size=self.n_arms)
        betas = self.some_constant * np.sqrt(1 / self.counts)
        used_parameter = self.values + betas * gumbel_rvs
        return np.argmax(used_parameter)


class BoltzmannGumbelRandomVariable(BaseModel):
    """abstract class for boltzmann gumbel classes with using costum random variables"""

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        some_constant: float,
        randomvariable_dict: dict,
    ) -> None:
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(
            some_constant
        ), "The some_constant  has to be a positive float"
        assert isinstance(randomvariable_dict, dict), "Have to be a dict"

        self.some_constant = some_constant
        _dist = getattr(stats, randomvariable_dict["name"])
        self.random_variable = _dist(**randomvariable_dict["parameter"]).rvs(
            size=(bandit_env.max_steps, bandit_env.n_arms)
        )

    def select_arm(self, arm_attrib: ArmAttributes) -> int:
        """get action from boltzmann gumbel paper"""

        random_variables = self.random_variable[arm_attrib.step_in_game,]
        betas = self.calculate_beta()
        betas = np.nan_to_num(betas, nan=np.inf)
        used_parameter = self.values + betas * random_variables

        return int(np.argmax(used_parameter))

    def calculate_beta(self) -> np.float32:
        """calculate beta value for constant gumbel

        Returns:
            np.float32: beta value for constant gumbel
        """
        return self.some_constant**2


# TODO: boltzmann random variable change
class BoltzmannGumbelRandomVariableSqrt(BoltzmannGumbelRandomVariable):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        some_constant: float,
        randomvariable_dict: dict,
    ):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(
            some_constant=some_constant,
            bandit_env=bandit_env,
            randomvariable_dict=randomvariable_dict,
        )

    def calculate_beta(self) -> np.float32:
        """calculate beta value for given steps"""
        return self.some_constant * np.sum(self.counts) ** (-0.5)


class BoltzmannGumbelRandomVariableLog(BoltzmannGumbelRandomVariable):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        some_constant: float,
        randomvariable_dict: dict,
    ):
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(
            bandit_env=bandit_env,
            some_constant=some_constant,
            randomvariable_dict=randomvariable_dict,
        )

    def calculate_beta(self) -> np.float32:
        """calculate the beta value for the given

        Returns:
            np.float32: _description_
        """
        return self.some_constant**2 * (np.log(np.sum(self.counts)) ** -1)


class BoltzmannGumbelRandomVariableUCB(BoltzmannGumbelRandomVariable):
    """boltzmann exploration algorithm also known as softmax bandit"""

    def __init__(
        self,
        bandit_env: BaseBanditEnv,
        some_constant: float,
        randomvariable_dict: dict,
    ) -> None:
        """initialize boltzmann algorithm with constant temperature

        Args:
            temperature (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(
            some_constant=some_constant,
            bandit_env=bandit_env,
            randomvariable_dict=randomvariable_dict,
        )

    def calculate_beta(self) -> np.float32:
        """calculate the beta value for the given step

        Returns:
            np.float32: beta parameter for the given step
        """
        return self.some_constant * np.sqrt(
            (np.log(np.sum(self.counts))) * self.counts**-1
        )


class GradientBandit(BaseModel):
    """gradient bandit algorithm"""

    def __init__(self, alpha: float, bandit_env: BaseBanditEnv) -> None:
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(bandit_env=bandit_env)
        # init tests
        assert is_positive_float(alpha), "Learning rate has to be a positive float"

        self.alpha: float = alpha
        self.count: int = 0
        self.mean_reward: float = 0.0

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
        gradients = (self.alpha * (reward - self.mean_reward)) * action_prob_vec

        # update values
        self.values = self.values + gradients
        self.count += 1
        # update mean reward
        self.mean_reward = ((self.count - 1) / float(self.count)) * self.mean_reward + (
            1 / float(self.count)
        ) * reward

    def reset(self) -> None:
        """reset agent by resetting all required statistics"""
        self.count = 0
        self.values = np.zeros(self.n_arms, dtype=np.float32)
        self.mean_reward = 0.0


class GradientBanditnobaseline(GradientBandit):
    """gradient bandit algorithm"""

    def __init__(self, alpha: float, bandit_env: BaseBanditEnv) -> None:
        """initialize gradient bandit with learning rate `alpha` and `n_arms`

        Args:
            alpha (float): float describing learning rate
            n_arms (int): number of used arms
        """
        super().__init__(alpha=alpha, bandit_env=bandit_env)

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
        gradients = (self.alpha * (reward)) * action_prob_vec

        # update values
        self.values = self.values + gradients


__all__ = [
    GradientBanditnobaseline.__name__,
    GradientBandit.__name__,
    UCB.__name__,
    ExploreThenCommit.__name__,
    EpsilonGreedy.__name__,
    BaseModel.__name__,
    BoltzmannConstant.__name__,
]
