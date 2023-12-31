from multiarmedbandits.environments import ArmDistTypes, BaseBanditEnv, DistParameter
from multiarmedbandits.run_algorithm import Algorithms, CompareMultiArmedBandits, MetricNames, MultiArmedBanditModel

# Define experiment configurations

## env
test_environment = BaseBanditEnv(
    distr_params=DistParameter(
        dist_type=ArmDistTypes.GAUSSIAN,
        mean_parameter=[0.1, 0.7, 0.5],
        scale_parameter=[1.0, 2.0, 3.0],
    ),
    max_steps=1000,
)

## Algorithms to compare
algo_one = MultiArmedBanditModel(dist_type=Algorithms.EXPLORRETHENCOMMIT, dist_params={"explore": 10})
algo_two = MultiArmedBanditModel(dist_type=Algorithms.EXPLORRETHENCOMMIT, dist_params={"explore": 50})
algo_three = MultiArmedBanditModel(dist_type=Algorithms.EXPLORRETHENCOMMIT, dist_params={"explore": 100})
algo_four = MultiArmedBanditModel(dist_type=Algorithms.EXPLORRETHENCOMMIT, dist_params={"explore": 200})
explorethencommit_compare = CompareMultiArmedBandits(
    test_env=test_environment,
    mab_algorithms=[algo_one, algo_two, algo_three, algo_four],
)
metrics = explorethencommit_compare.train_all_models(no_of_runs=100)

explorethencommit_compare.plot_multiple_mabs(
    metrics_to_plot=[
        MetricNames.AVERAGE_REWARD,
        MetricNames.OPTIM_PERCENTAGE,
        MetricNames.REGRETCONVERGENCE,
    ],
    named_metrics=metrics,
)
