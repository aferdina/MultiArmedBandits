import pytest
from multiarmedbandits.environments import BaseBanditEnv
from multiarmedbandits.algorithms import EpsilonGreedy


@pytest.mark.parametrize(
    "env, algo",
    [(pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("epsilon_greedy"))],
)
def test_epsilon_greedy_model(env: BaseBanditEnv, algo: EpsilonGreedy) -> None:
    env.reset()
    algo.reset()
    assert algo.epsilon == 0.1
    assert algo.n_arms == 2
