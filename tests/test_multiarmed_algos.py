from typing import Any, Dict

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
def test_general_model(env: BaseBanditEnv, algo: mab_algo.BaseLearningRule) -> None:
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
def test_epsilon_greedy_model_two(env: BaseBanditEnv, algo: mab_algo.EpsilonGreedy) -> None:
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
            pytest.lazy_fixture("bernoulli_env"),
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
            pytest.lazy_fixture("bernoulli_env"),
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
            pytest.lazy_fixture("bernoulli_env"),
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
            pytest.lazy_fixture("bernoulli_env"),
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
            pytest.lazy_fixture("bernoulli_env"),
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


@pytest.mark.parametrize("env, algo", [(pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("ucb_alpha"))])
def test_ucb_alpha(env: BaseBanditEnv, algo: mab_algo.UCBAlpha) -> None:
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
    action = 0
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([1, 2], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([4.0, 1.0], dtype=np.float32))

    # test select arm method
    selected_arms = [algo.select_arm(info[INFODICT.ARMATTRIBUTES]) for _ in range(int(1e5))]
    # ucb of arm 0 is higher than ucb of arm 1, so only arm 0 should be selected
    assert 0 in selected_arms
    assert 1 not in selected_arms


@pytest.mark.parametrize("env, algo", [(pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("lecture_ucb"))])
def test_lecture_ucb(env: BaseBanditEnv, algo: mab_algo.LectureUCB) -> None:
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
    action = 0
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.counts, np.array([1, 2], dtype=np.float32))
    assert np.array_equal(algo.values, np.array([4.0, 1.0], dtype=np.float32))

    # test select arm method
    selected_arms = [algo.select_arm(info[INFODICT.ARMATTRIBUTES]) for _ in range(int(1e5))]
    # ucb of arm 0 is higher than ucb of arm 1, so only arm 0 should be selected
    assert 0 in selected_arms
    assert 1 not in selected_arms


@pytest.mark.parametrize(
    "env, config",
    [
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("config_unknown_prior")),
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("config_empty")),
        (pytest.lazy_fixture("gaussian_env"), pytest.lazy_fixture("config_beta")),
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("config_normal")),
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("config_nig")),
    ],
)
def test_thompson_wrong_prior(env: BaseBanditEnv, config: Dict[str, Any]) -> None:
    with pytest.raises(AssertionError):
        mab_algo.ThompsonSampling(bandit_env=env, config=config)


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("thompson_beta_without_info")),
        (pytest.lazy_fixture("bernoulli_env"), pytest.lazy_fixture("thompson_beta_with_info")),
    ],
)
def test_thompson_beta(env: BaseBanditEnv, algo: mab_algo.ThompsonSampling) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.n_arms == 2
    assert np.array_equal(algo.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.values, np.zeros(2, dtype=np.float32))

    init_alpha = algo.posterior.alpha.copy()
    init_beta = algo.posterior.beta.copy()

    # test selected arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert reward in [1.0, 0.0]
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] + algo.posterior.beta[action] == init_alpha[action] + init_beta[action] + 1
    assert sum(init_alpha) + sum(init_beta) + 1 == sum(algo.posterior.alpha) + sum(algo.posterior.beta)

    # test update method
    algo.reset()
    assert np.array_equal(algo.posterior.alpha, init_alpha)
    assert np.array_equal(algo.posterior.beta, init_beta)
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == init_alpha[action] + 1
    assert algo.posterior.beta[action] == init_beta[action]
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == init_alpha[action] + 2
    assert algo.posterior.beta[action] == init_beta[action]
    reward = 0.0
    action = 0
    algo.update(chosen_arm=action, reward=reward)
    assert algo.posterior.alpha[action] == init_alpha[action]
    assert algo.posterior.beta[action] == init_beta[action] + 1


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("gaussian_env"), pytest.lazy_fixture("thompson_normal_without_info")),
        (pytest.lazy_fixture("gaussian_env"), pytest.lazy_fixture("thompson_normal_with_info")),
    ],
)
def test_thompson_normal(env: BaseBanditEnv, algo: mab_algo.ThompsonSampling) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.n_arms == 2
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))

    init_mean = algo.posterior.mean.copy()
    init_scale = algo.posterior.scale.copy()

    # test selected arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))

    # test update method
    algo.reset()
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.mean, init_mean)
    assert np.array_equal(algo.posterior.scale, init_scale)
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


@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("gaussian_env"), pytest.lazy_fixture("thompson_nig_without_info")),
        (pytest.lazy_fixture("gaussian_env"), pytest.lazy_fixture("thompson_nig_with_info")),
    ],
)
def test_thompson_nig(env: BaseBanditEnv, algo: mab_algo.ThompsonSampling) -> None:
    # resetting environment and algorithm
    _new_state, info = env.reset()
    algo.reset()
    assert algo.n_arms == 2
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.sqsum, np.zeros(2, dtype=np.float32))

    init_mean = algo.posterior.mean.copy()
    init_lambda = algo.posterior.lambda_p.copy()
    init_alpha = algo.posterior.alpha.copy()
    init_beta = algo.posterior.beta.copy()

    # test selected arm method
    action = algo.select_arm(arm_attrib=info[INFODICT.ARMATTRIBUTES])
    assert action in range(2)
    # test environment step for given action
    _new_state, reward, done, info = env.step(action=action)
    assert _new_state == 0
    assert done is False
    algo.update(chosen_arm=action, reward=reward)
    assert not np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))

    # test update method
    algo.reset()
    assert np.array_equal(algo.posterior.counts, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.sqsum, np.zeros(2, dtype=np.float32))
    assert np.array_equal(algo.posterior.mean, init_mean)
    assert np.array_equal(algo.posterior.lambda_p, init_lambda)
    assert np.array_equal(algo.posterior.alpha, init_alpha)
    assert np.array_equal(algo.posterior.beta, init_beta)
    reward = 1.0
    action = 1

    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 1], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 1.0], dtype=np.float32))
    assert np.array_equal(algo.posterior.sqsum, np.array([0, 1.0], dtype=np.float32))
    reward = 1.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 2], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 1.0], dtype=np.float32))
    assert np.array_equal(algo.posterior.sqsum, np.array([0, 2.0], dtype=np.float32))
    reward = 4.0
    action = 1
    algo.update(chosen_arm=action, reward=reward)
    assert np.array_equal(algo.posterior.counts, np.array([0, 3], dtype=np.float32))
    assert np.array_equal(algo.posterior.values, np.array([0, 2.0], dtype=np.float32))
    assert np.array_equal(algo.posterior.sqsum, np.array([0, 18.0], dtype=np.float32))
