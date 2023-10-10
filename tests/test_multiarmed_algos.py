import numpy as np
import pytest

import multiarmedbandits.algorithms as mab_algo
from multiarmedbandits.environments import INFODICT, BaseBanditEnv


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("epsilon_greedy")),
        (
            pytest.lazy_fixture("bernoulli_env"),
            pytest.lazy_fixture("explore_then_commit"),
        ),
    ],
)
def test_general_model(env: BaseBanditEnv, algo: mab_algo.BaseModel) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    # assert algo.epsilon == 0.1
    assert algo.n_arms == 2
    assert algo.max_steps == 10
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))

    # test update method
    algo.reset()
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 1], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 1.0], dtype=np.float32))
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 2], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 1.0], dtype=np.float32))
    reward = 4.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 3], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("epsilon_greedy")),
    ],
)
def test_epsilon_greedy_model(env: BaseBanditEnv, algo: mab_algo.EpsilonGreedy) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    # assert algo.epsilon == 0.1
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
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))

    # test update method
    algo.reset()
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 1], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 1.0], dtype=np.float32))
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 2], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 1.0], dtype=np.float32))
    reward = 4.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([0, 3], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([0, 2.0], dtype=np.float32))

    # test select arm method
    selected_arms = [algo.select_arm(info[INFODICT.ARMATTRIBUTES]) for _ in range(int(1e5))]
    # probability zero, that only one arm is selected
    assert 0 in selected_arms
    assert 1 in selected_arms


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env"),
            pytest.lazy_fixture("explore_then_commit"),
        ),
    ],
)
def test_epsilon_greedy_model2(env: BaseBanditEnv, algo: mab_algo.EpsilonGreedy) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([1, 0], dtype=np.float32))
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([1, 1], dtype=np.float32))


@pytest.mark.parametrize("env, algo",
                         [(pytest.lazy_fixture("bernoulli_env"),
                           pytest.lazy_fixture("thompson_bernoulli_no_info"))])
def test_thompson_bernoulli_no_info(env: BaseBanditEnv, algo: mab_algo.ThompsonSampling) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.n_arms == 2
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))

    # test selected arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]
    assert _new_state == 0
    assert done is False
    old_sums = [algo.posterior.alpha[i] + algo.posterior.beta[i] for i in range(algo.n_arms)]
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] + algo.posterior.beta[action] == old_sums[action] + 1
    assert sum(old_sums) + 1 == sum(algo.posterior.alpha) + sum(algo.posterior.beta)

    # test update method
    algo.reset()
    assert np.array_equal(algo.posterior.alpha, np.ones(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.beta, np.ones(2, dtype=np.float32))
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == 2
    assert algo.posterior.beta[action] == 1
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == 3
    assert algo.posterior.beta[action] == 1
    reward = 0.0
    action = 0
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == 1
    assert algo.posterior.beta[action] == 2


@pytest.mark.parametrize("env, algo",
                         [(pytest.lazy_fixture("gaussian_env"),
                           pytest.lazy_fixture("thompson_gaussian_no_info"))])
def test_thompson_gaussian_no_info(env: BaseBanditEnv, algo: mab_algo.ThompsonSampling) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.n_arms == 2
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.mean, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.scale, np.ones(2, dtype=np.float32))

    # test selected arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert not np.array_equal(algo.posterior.mean, np.zeros(2, dtype=np.float32))
    assert not np.array_equal(algo.posterior.scale, np.ones(2, dtype=np.float32))

    # test update method
    algo.reset()
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.mean, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.scale, np.ones(2, dtype=np.float32))
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 1], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 1.0], dtype=np.float32))
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 2], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 1.0], dtype=np.float32))
    reward = 4.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 3], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 2.0], dtype=np.float32))