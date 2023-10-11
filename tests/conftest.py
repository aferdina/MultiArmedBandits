import pytest

import multiarmedbandits.algorithms as mab_algos
import multiarmedbandits.environments as mab_envs
from multiarmedbandits.utils.posterior import PriorType


@pytest.fixture(scope="module")
def gaussian_env() -> mab_envs.BaseBanditEnv:
    return mab_envs.BaseBanditEnv(
        distr_params=mab_envs.DistParameter(
            dist_type=mab_envs.ArmDistTypes.GAUSSIAN,
            mean_parameter=[0.1, 0.2],
            scale_parameter=[1.0, 1.0],
        ),
        max_steps=10,
    )


@pytest.fixture(scope="module")
def bernoulli_env() -> mab_envs.BaseBanditEnv:
    return mab_envs.BaseBanditEnv(
        distr_params=mab_envs.DistParameter(
            dist_type=mab_envs.ArmDistTypes.BERNOULLI,
            mean_parameter=[0.1, 0.7],
        ),
        max_steps=10,
    )


@pytest.fixture(scope="module")
def testbed_env() -> mab_envs.BaseBanditEnv:
    return mab_envs.TestBed(
        max_steps=10,
        testbed_config=mab_envs.TestBedConfigs(
            type=mab_envs.TestBedSampleType.BERNOULLI,
            sample_config={"n": 1, "p": 0.4},
            no_arms=5,
            arm_type=mab_envs.ArmDistTypes.BERNOULLI,
        ),
    )


@pytest.fixture(scope="module")
def gap_env() -> mab_envs.GapEnv:
    return mab_envs.GapEnv(
        gap_configs=mab_envs.GapEnvConfigs(
            no_of_arms=2,
            single_arm_distr=mab_envs.SingleArmParams(
                arm_type=mab_envs.ArmDistTypes.GAUSSIAN,
                mean_parameter=0.4,
                scale_parameter=1.0,
            ),
            gap_parameter=0.2,
        ),
        max_steps=10,
    )


@pytest.fixture(scope="module")
def epsilon_greedy(bernoulli_env) -> mab_algos.EpsilonGreedy:
    return mab_algos.EpsilonGreedy(epsilon=0.1, bandit_env=bernoulli_env)


@pytest.fixture(scope="module")
def explore_then_commit(bernoulli_env) -> mab_algos.ExploreThenCommit:
    return mab_algos.ExploreThenCommit(explore=1, bandit_env=bernoulli_env)


# @pytest.fixture(scope="module")
# def config_empty():
#     return {}


@pytest.fixture(scope="module")
def config_beta_no_info():
    return {"prior": PriorType.BETA}


# @pytest.fixture(scope="module")
# def config_beta_with_info():
#     return {"prior": PriorType.BETA,
#                 "alpha": [1, 2],
#                 "beta": [1, 2]}


@pytest.fixture(scope="module")
def config_normal_no_info():
    return {"prior": PriorType.NORMAL}


# @pytest.fixture(scope="module")
# def config_normal_with_info():
#     return {"prior": PriorType.NORMAL,
#                 "mean": [1, 2],
#                 "scale": [1, 2]}

# @pytest.fixture(scope="module")
# def config_nig_no_info():
#     return {"prior": PriorType.NIG}

# @pytest.fixture(scope="module")
# def config_nig_with_info():
#     return {"prior": PriorType.NIG,
#                 "mean": [1, 2],
#                 "lambda": [1, 2],
#                 "alpha": [2, 2],
#                 "beta": [2, 2]}


@pytest.fixture(scope="module")
def thompson_bernoulli_no_info(bernoulli_env, config_beta_no_info) -> mab_algos.ThompsonSampling:
    return mab_algos.ThompsonSampling(bandit_env=bernoulli_env, config=config_beta_no_info)


# @pytest.fixture(scope="module")
# def thompson_bernoulli_with_info(bernoulli_env, config_beta_with_info) -> mab_algos.ThompsonSampling:
#     return mab_algos.ThompsonSampling(bandit_env=bernoulli_env, config=config_beta_with_info)


@pytest.fixture(scope="module")
def thompson_gaussian_no_info(gaussian_env, config_normal_no_info) -> mab_algos.ThompsonSampling:
    return mab_algos.ThompsonSampling(bandit_env=gaussian_env, config=config_normal_no_info)
