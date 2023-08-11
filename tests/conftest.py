import pytest

import multiarmedbandits.environments as mab_envs
import multiarmedbandits.algorithms as mab_algos


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
            no_of_arms=10,
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
