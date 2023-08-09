from multiarmedbandits.run_algorithm import (
    CompareMultiArmedBandits,
    Algorithms,
    MultiArmedBanditModel,
)
from multiarmedbandits.environments import (
    BaseBanditEnv,
    DistParameter,
    ArmDistTypes,
)
from multiarmedbandits.run_algorithm.utils import MetricNames
import multiarmedbandits.algorithms.multiarmed_bandit_models as bandit_algos

bandit_env = BaseBanditEnv(
    distr_params=DistParameter(
        dist_type=ArmDistTypes.GAUSSIAN,
        mean_parameter=[0.1, 0.2, 0.3],
        scale_parameter=[1.0, 1.0, 1.0],
    ),
    max_steps=10000,
)
algo_one = MultiArmedBanditModel(
    dist_type=Algorithms.UCBALGO,
    dist_params={"delta": 0.2},
)
algo_two = MultiArmedBanditModel(
    dist_type=Algorithms.UCBALGO,
    dist_params={"delta": 0.3},
)
algo_three = MultiArmedBanditModel(
    dist_type=Algorithms.UCBALGO,
    dist_params={"delta": 0.4},
)
algo_four = MultiArmedBanditModel(
    dist_type=Algorithms.UCBALGO,
    dist_params={"delta": 0.5},
)
compare = CompareMultiArmedBandits(
    test_env=bandit_env, mab_algorithms=[algo_one, algo_two, algo_three, algo_four]
)
metrics = compare.train_all_models(no_of_runs=100)
compare.plot_multiple_mabs(
    named_metrics=metrics,
    metrics_to_plot=[MetricNames.AVERAGE_REWARD, MetricNames.OPTIM_PERCENTAGE],
)
