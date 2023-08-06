import pytest

import multiarmedbandits.environments as mab_envs


@pytest.fixture(scope="module")
def bernoulli_env() -> mab_envs.BaseBanditEnv:
    return mab_envs.BaseBanditEnv(
        distr_params=mab_envs.DistParameter(
            dist_type=mab_envs.ArmDistTypes.GAUSSIAN,
            mean_parameter=[0.1, 0.2],
            scale_parameter=[1.0, 1.0],
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
