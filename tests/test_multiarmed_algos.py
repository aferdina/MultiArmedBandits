import numpy as np
import pytest
from unittest.mock import MagicMock

import multiarmedbandits.algorithms as mab_algo
from multiarmedbandits.environments import INFODICT, BaseBanditEnv


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("bernoulli_env_2arms"), pytest.lazy_fixture("epsilon_greedy")),
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
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
        (pytest.lazy_fixture("bernoulli_env_2arms"), pytest.lazy_fixture("epsilon_greedy")),
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
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("explore_then_commit"),
        ),
    ],
)
def test_epsilon_greedy_model(env: BaseBanditEnv, algo: mab_algo.EpsilonGreedy) -> None:
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


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("simple_boltzmann_const_2arms"),
        ),
    ],
)
def test_boltzmann_const_2arms(env: BaseBanditEnv, algo: mab_algo.BoltzmannSimple) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas - CONSTANT
    betas = algo.calc_betas()
    assert np.array_equal(betas, np.array(2 * [0.5**2], dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]

    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 1
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 2


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("simple_boltzmann_log_2arms"),
        ),
    ],
)
def test_boltzmann_log_2arms(env: BaseBanditEnv, algo: mab_algo.BoltzmannSimple) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 1 - LOG
    step_in_game = 0
    some_constant = np.array(2 * [0.5])
    values = algo.values
    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.log(1 + step_in_game) == 0
    assert np.array_equal(betas, np.full_like(values, np.inf))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]

    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 1
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 2 - LOG
    step_in_game = 1
    values = algo.values
    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.array(some_constant**2 / np.log(1 + step_in_game), dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 2


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("simple_boltzmann_sqrt_2arms"),
        ),
    ],
)
def test_boltzmann_sqrt_2arms(env: BaseBanditEnv, algo: mab_algo.BoltzmannSimple) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 1 - SQRT
    step_in_game = 0
    some_constant = np.array(2 * [0.5])
    values = algo.values
    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.array(some_constant**2 / np.sqrt(1 + step_in_game), dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]

    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 1
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 2 - SQRT
    step_in_game = 1
    values = algo.values
    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.array(some_constant**2 / np.sqrt(1 + step_in_game), dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 2


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("simple_boltzmann_ucb_2arms"),
        ),
    ],
)
def test_boltzmann_ucb_2arms(env: BaseBanditEnv, algo: mab_algo.BoltzmannSimple) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 1 - UCB
    some_constant = np.array(2 * [0.5])
    values = algo.values
    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.full_like(values, np.inf))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]

    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 1
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 2 - UCB
    step_in_game = 1
    expected_result = np.divide(
        some_constant, np.sqrt(algo.counts), out=np.zeros_like(some_constant), where=np.sqrt(algo.counts) != 0
    )
    expected_result = expected_result * np.sqrt(np.log(1 + step_in_game))
    expected_result[expected_result == 0.0] = np.inf

    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.array(expected_result, dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 2


@pytest.mark.parametrize(
    "env, algo",
    [
        (
            pytest.lazy_fixture("bernoulli_env_2arms"),
            pytest.lazy_fixture("simple_boltzmann_bge_2arms"),
        ),
    ],
)
def test_boltzmann_bge_2arms(env: BaseBanditEnv, algo: mab_algo.BoltzmannSimple) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()

    # test select arm method
    action = algo.select_arm()
    assert action in range(2)

    # test calc_betas 1 - BGE
    some_constant = np.array(2 * [0.5])
    betas = algo.calc_betas()
    assert np.array_equal(betas, np.array(2 * [np.inf], dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]

    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 1
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)

    # test calc_betas 2 - BGE
    expected_result = np.divide(
        some_constant, np.sqrt(algo.counts), out=np.zeros_like(some_constant), where=np.sqrt(algo.counts) != 0
    )
    expected_result[expected_result == 0.0] = np.inf

    betas = algo.calc_betas(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert np.array_equal(betas, np.array(expected_result, dtype=np.float32))

    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    algo.update(chosen_arm=action, reward=reward)
    assert algo.counts.sum() == 2
