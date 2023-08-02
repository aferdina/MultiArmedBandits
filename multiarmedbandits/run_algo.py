from multiarmedbandits.algorithms.multiarmed_bandit_models import EpsilonGreedy, BoltzmannGumbel
from multiarmedbandits.environments.multiarmed_env import (
    GaussianBanditEnv,
    DistParameter,
)
from multiarmedbandits.run_algorithm.train_multiarmed_bandits import (
    RunMultiarmedBanditModel,
)
from multiarmedbandits.run_algorithm.utils import MetricNames

algo = EpsilonGreedy(epsilon=0.1, n_arms=2)
algo = BoltzmannGumbel(temperature=2.0,n_arms = 2)
env = GaussianBanditEnv(
    distr_params=DistParameter(mean_parameter=[0.1, 0.7], scale_parameter=[1.0, 1.0]),
    max_steps=1000,
)

run_training = RunMultiarmedBanditModel(mab_algo=algo, bandit_env=env)

run_training.reset_statistics()

metrics = run_training.get_metrics_from_runs(no_runs=1000)
run_training.add_runs_to_metrics(metrics)
run_training.plot_mab_statistics(
    metrics_to_plot=[MetricNames.AVERAGE_REWARD, MetricNames.REGRETCONVERGENCE,MetricNames.OPTIM_PERCENTAGE, MetricNames.REGRET]
)
