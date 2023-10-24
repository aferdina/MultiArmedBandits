import pytest

import multiarmedbandits.algorithms as mab_algos
import multiarmedbandits.environments as mab_envs


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


@pytest.fixture(scope="module")
def ucb_alpha(bernoulli_env) -> mab_algos.UCBAlpha:
    return mab_algos.UCBAlpha(bandit_env=bernoulli_env, alpha=2.0)


@pytest.fixture(scope="module")
def lecture_ucb(bernoulli_env) -> mab_algos.LectureUCB:
    return mab_algos.LectureUCB(bandit_env=bernoulli_env, delta=0.1)


@pytest.fixture(scope="module")
def simple_boltzmann_const_2arms(bernoulli_env) -> mab_algos.BoltzmannSimple:
    return mab_algos.BoltzmannSimple(
        boltzmann_configs=mab_algos.boltzmann.BoltzmannConfigs(
            explor_type=mab_algos.boltzmann.ExplorationType.CONSTANT, some_constant=[0.5, 0.5]
        ),
        bandit_env=bernoulli_env,
    )


@pytest.fixture(scope="module")
def simple_boltzmann_log_2arms(bernoulli_env) -> mab_algos.BoltzmannSimple:
    return mab_algos.BoltzmannSimple(
        boltzmann_configs=mab_algos.boltzmann.BoltzmannConfigs(
            explor_type=mab_algos.boltzmann.ExplorationType.LOG, some_constant=[0.5, 0.5]
        ),
        bandit_env=bernoulli_env,
    )


@pytest.fixture(scope="module")
def simple_boltzmann_sqrt_2arms(bernoulli_env) -> mab_algos.BoltzmannSimple:
    return mab_algos.BoltzmannSimple(
        boltzmann_configs=mab_algos.boltzmann.BoltzmannConfigs(
            explor_type=mab_algos.boltzmann.ExplorationType.SQRT, some_constant=[0.5, 0.5]
        ),
        bandit_env=bernoulli_env,
    )


@pytest.fixture(scope="module")
def simple_boltzmann_ucb_2arms(bernoulli_env) -> mab_algos.BoltzmannSimple:
    return mab_algos.BoltzmannSimple(
        boltzmann_configs=mab_algos.boltzmann.BoltzmannConfigs(
            explor_type=mab_algos.boltzmann.ExplorationType.UCB, some_constant=[0.5, 0.5]
        ),
        bandit_env=bernoulli_env,
    )


@pytest.fixture(scope="module")
def simple_boltzmann_bge_2arms(bernoulli_env) -> mab_algos.BoltzmannSimple:
    return mab_algos.BoltzmannSimple(
        boltzmann_configs=mab_algos.boltzmann.BoltzmannConfigs(
            explor_type=mab_algos.boltzmann.ExplorationType.BGE, some_constant=[0.5, 0.5]
        ),
        bandit_env=bernoulli_env,
    )


@pytest.fixture(scope="module")
def gradient_bandit(bernoulli_env) -> mab_algos.GradientBandit:
    return mab_algos.GradientBandit(
        alpha=0.05, baseline_attr=mab_algos.GradientBaseLineAttr(type=mab_algos.BaseLinesTypes.ZERO), bandit_env=bernoulli_env
    )


@pytest.fixture(scope="module")
def gradient_bandit_mean(bernoulli_env) -> mab_algos.GradientBandit:
    return mab_algos.GradientBandit(
        alpha=0.05, baseline_attr=mab_algos.GradientBaseLineAttr(type=mab_algos.BaseLinesTypes.MEAN), bandit_env=bernoulli_env
    )


@pytest.fixture(scope="module")
def gradient_bandit_median(bernoulli_env) -> mab_algos.GradientBandit:
    return mab_algos.GradientBandit(
        alpha=0.05,
        baseline_attr=mab_algos.GradientBaseLineAttr(type=mab_algos.BaseLinesTypes.MEDIAN),
        bandit_env=bernoulli_env,
    )
