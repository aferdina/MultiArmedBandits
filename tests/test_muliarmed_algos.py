import pytest
import numpy as np
from multiarmedbandits.environments import BaseBanditEnv, INFODICT
from multiarmedbandits.algorithms import EpsilonGreedy


@pytest.mark.parametrize(
    "env, algo",
    [(pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("epsilon_greedy"))],
)
def test_epsilon_greedy_model(env: BaseBanditEnv, algo: EpsilonGreedy) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.epsilon == 0.1
    assert algo.n_arms == 2
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]
    assert _new_state == 0
    assert done == False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))

    # test update method
