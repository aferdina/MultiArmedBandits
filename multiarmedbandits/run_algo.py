from multiarmedbandits.algorithms.multiarmed_bandit_models import (
    EpsilonGreedy,
    BoltzmannGumbel,
    BoltzmannGumbelRightWay,
)
from multiarmedbandits.environments.multiarmed_env import (
    BaseBanditEnv,
    DistParameter,
    ArmDistTypes,
)
from multiarmedbandits.run_algorithm.train_multiarmed_bandits import (
    RunMultiarmedBanditModel,
)
from multiarmedbandits.run_algorithm.utils import MetricNames

env = BaseBanditEnv(
    distr_params=DistParameter(
        dist_type=ArmDistTypes.BERNOULLI,
        mean_parameter=[0.1, 0.7],
        scale_parameter=[1.0, 1.0],
    ),
    max_steps=1000,
)
algo = EpsilonGreedy(epsilon=0.1, bandit_env=env)
algo = BoltzmannGumbel(temperature=2.0, bandit_env=env)
algo = BoltzmannGumbelRightWay(some_constant=2.0, bandit_env=env)

run_training = RunMultiarmedBanditModel(mab_algo=algo, bandit_env=env)

run_training.reset_statistics()

metrics = run_training.get_metrics_from_runs(no_runs=1000)
run_training.add_runs_to_metrics(metrics)
run_training.plot_mab_statistics(
    metrics_to_plot=[
        MetricNames.AVERAGE_REWARD,
        MetricNames.REGRETCONVERGENCE,
        MetricNames.OPTIM_PERCENTAGE,
        MetricNames.REGRET,
    ]
)
