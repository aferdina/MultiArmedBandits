import multiarmedbandits.algorithms.multiarmed_bandit_models as bandit_algos
from multiarmedbandits.environments import ArmDistTypes, BaseBanditEnv, DistParameter
from multiarmedbandits.run_algorithm import Algorithms, CompareMultiArmedBandits, MultiArmedBanditModel
from multiarmedbandits.run_algorithm.utils import MetricNames

bandit_env = BaseBanditEnv(
    distr_params=DistParameter(
        dist_type=ArmDistTypes.GAUSSIAN,
        mean_parameter=[0.1, 0.2, 0.3],
        scale_parameter=[1.0, 1.0, 1.0],
    ),
    max_steps=10000,
)
algo_one = MultiArmedBanditModel(
    dist_type=Algorithms.BOLTZMANNRANDOM,
    dist_params={
        "boltzmann_configs": bandit_algos.BoltzmannConfigs(
            explor_type=bandit_algos.ExplorationType.BGE,
            some_constant=[1.0, 1.0, 1.0],
        ),
        "rv_config": bandit_algos.RandomVariable(rv_name="gumbel_r", rv_param={"scale": 1.0, "loc": 0.0}),
    },
)
algo_two = MultiArmedBanditModel(
    dist_type=Algorithms.BOLTZMANNRANDOM,
    dist_params={
        "boltzmann_configs": bandit_algos.BoltzmannConfigs(
            explor_type=bandit_algos.ExplorationType.SQRT,
            some_constant=[1.0, 1.0, 1.0],
        ),
        "rv_config": bandit_algos.RandomVariable(rv_name="gumbel_r", rv_param={"scale": 1.0, "loc": 0.0}),
    },
)
algo_three = MultiArmedBanditModel(
    dist_type=Algorithms.BOLTZMANNRANDOM,
    dist_params={
        "boltzmann_configs": bandit_algos.BoltzmannConfigs(
            explor_type=bandit_algos.ExplorationType.CONSTANT,
            some_constant=[1.0, 1.0, 1.0],
        ),
        "rv_config": bandit_algos.RandomVariable(rv_name="gumbel_r", rv_param={"scale": 1.0, "loc": 0.0}),
    },
)
algo_four = MultiArmedBanditModel(
    dist_type=Algorithms.BOLTZMANNRANDOM,
    dist_params={
        "boltzmann_configs": bandit_algos.BoltzmannConfigs(
            explor_type=bandit_algos.ExplorationType.LOG,
            some_constant=[1.0, 1.0, 1.0],
        ),
        "rv_config": bandit_algos.RandomVariable(rv_name="gumbel_r", rv_param={"scale": 1.0, "loc": 0.0}),
    },
)
compare = CompareMultiArmedBandits(test_env=bandit_env, mab_algorithms=[algo_one, algo_two, algo_three, algo_four])
metrics = compare.train_all_models(no_of_runs=100)
compare.plot_multiple_mabs(
    named_metrics=metrics,
    metrics_to_plot=[MetricNames.AVERAGE_REWARD, MetricNames.OPTIM_PERCENTAGE],
)
