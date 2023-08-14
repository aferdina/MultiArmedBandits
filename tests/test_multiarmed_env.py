from typing import List
import pytest
from multiarmedbandits.environments import BaseBanditEnv


@pytest.mark.parametrize(
    "env, max_steps, max_mean, max_mean_position",
    [
        (
            pytest.lazy_fixture("bernoulli_env"),
            10,
            0.7,
            [
                1,
            ],
        ),
        (
            pytest.lazy_fixture("gaussian_env"),
            10,
            0.2,
            [
                1,
            ],
        ),
        (
            pytest.lazy_fixture("gap_env"),
            10,
            0.6,
            [
                0,
            ],
        ),
    ],
)
def test_env_init(
    env: BaseBanditEnv, max_steps: int, max_mean: int, max_mean_position: List[int]
) -> None:
    env.reset()
    assert env.count == 0, "env count after resetting should be equal to 0"
    assert env.done is False, "env done init as False"
    assert isinstance(env.max_steps, int), "max steps should be an integer"
    assert env.max_steps == max_steps
    # testing bandit statistics
    assert env.bandit_statistics.max_mean == pytest.approx(max_mean)
    assert env.bandit_statistics.regret == 0
    assert env.bandit_statistics.max_mean_positions == max_mean_position
    assert env.bandit_statistics.played_optimal == 0
    # testing bandit step
    for play_action in range(2):
        next_state, reward, done, info = env.step(play_action)
        assert isinstance(reward, float), "reward must be a float variable"
        assert next_state == 0, "next state is always 0"
        assert isinstance(done, bool), "done must be a bool variable"
        assert isinstance(info, dict), "info must be a dict"
    assert env.count == 2, "env count after going two steps should be equal to 2"
    assert (
        env.bandit_statistics.played_optimal == 1
    ), "after playing each arm ones, one time optimal is played"
    env.reset()
    # testing bandit statistics again after step
    assert env.count == 0, "env count after resetting should be equal to 0"
    assert env.done is False, "env done init as False"
    assert env.bandit_statistics.max_mean == pytest.approx(max_mean)
    assert env.bandit_statistics.regret == 0
    assert env.bandit_statistics.max_mean_positions == max_mean_position
    assert env.bandit_statistics.played_optimal == 0


@pytest.mark.parametrize(
    "env",
    [pytest.lazy_fixture("testbed_env")],
)
def test_testbed_init(env: BaseBanditEnv) -> None:
    env.reset()
    assert env.count == 0, "env count after resetting should be equal to 0"
    assert env.done is False, "env done init as False"
    assert env.max_steps == 10, "max steps are also 10"
    # test statistics
    assert env.bandit_statistics.regret == 0
    assert 1 >= env.bandit_statistics.max_mean >= 0, "arm type is bernoulli"
    assert env.bandit_statistics.played_optimal == 0
    # play each arm ones
    for play_action in range(5):
        next_state, reward, done, info = env.step(play_action)
        assert isinstance(reward, float), "reward must be a float variable"
        assert next_state == 0, "next state is always 0"
        assert isinstance(done, bool), "done must be a bool variable"
        assert isinstance(info, dict), "info must be a dict"
    assert (
        env.bandit_statistics.played_optimal >= 1
    ), "after playing each arm ones, at least one optimal"
    assert env.count == 5, "env count after going two steps should be equal to 2"
    env.reset()
    assert env.count == 0, "env count after resetting should be equal to 0"
    assert env.done is False, "env done init as False"
    assert env.bandit_statistics.played_optimal == 0
    assert env.bandit_statistics.regret == 0
