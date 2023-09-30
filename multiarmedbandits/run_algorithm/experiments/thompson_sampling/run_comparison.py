from multiarmedbandits.environments import ArmDistTypes, BaseBanditEnv, DistParameter
from multiarmedbandits.run_algorithm import Algorithms, CompareMultiArmedBandits, MetricNames, MultiArmedBanditModel

bandit_env = BaseBanditEnv(
    distr_params=DistParameter(
        dist_type=ArmDistTypes.BERNOULLI,
        mean_parameter=[0.1, 0.2, 0.3],
        scale_parameter=[1.0, 1.0, 1.0],
    ),
    max_steps=10000,
)
algo_one = MultiArmedBanditModel(
    dist_type=Algorithms.THOMPSON,
    dist_params={
        "config": {
            "prior": "beta",
        }
    },
)
algo_two = MultiArmedBanditModel(
    dist_type=Algorithms.THOMPSON,
    dist_params={
        "config": {
            "prior": "beta",
            "alpha": [10.0, 10.0, 10.0]
        }
    },
)

compare = CompareMultiArmedBandits(test_env=bandit_env, mab_algorithms=[algo_one, algo_two])
metrics = compare.train_all_models(no_of_runs=100)
compare.plot_multiple_mabs(
    named_metrics=metrics,
    metrics_to_plot=[MetricNames.AVERAGE_REWARD, MetricNames.OPTIM_PERCENTAGE],
)
