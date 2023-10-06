import numpy as np
import pytest

import multiarmedbandits.algorithms as mab_algo
from multiarmedbandits.environments import INFODICT, BaseBanditEnv

from multiarmedbandits.run_algorithm.metrics import MABMetrics, MetricNames, MultiArmedBanditModel, NamedMABMetrics

from multiarmedbandits.run_algorithm.compare_models import RunMultiarmedBanditModel

# TO DO
# explore then commit ist in der fixture zu schlecht wegen 100% exploration
#


# Test quality of algorithms in bernoulli_env
@pytest.mark.parametrize(
    "env, algo",
    [
        (pytest.lazy_fixture("bernoulli_env_quality"), pytest.lazy_fixture("epsilon_greedy")),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("explore_then_commit"),
        ),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("simple_boltzmann_const_2arms"),
        ),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("simple_boltzmann_log_2arms"),
        ),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("simple_boltzmann_sqrt_2arms"),
        ),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("simple_boltzmann_ucb_2arms"),
        ),
        (
            pytest.lazy_fixture("bernoulli_env_quality"),
            pytest.lazy_fixture("simple_boltzmann_bge_2arms"),
        ),
        (pytest.lazy_fixture("bernoulli_env_quality"), pytest.lazy_fixture("ucb_alpha")),
        (pytest.lazy_fixture("bernoulli_env_quality"), pytest.lazy_fixture("lecture_ucb")),
    ],
)
def test_quality_model(env: BaseBanditEnv, algo: mab_algo.BaseLearningRule) -> None:
    train_algo = RunMultiarmedBanditModel(mab_algo=algo, bandit_env=env)
    metrics = train_algo.get_metrics_from_runs(no_runs=algo.max_steps)
    assert metrics.optim_percentage[-1] > 0.9
