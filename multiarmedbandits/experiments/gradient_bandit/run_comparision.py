from multiarmedbandits.run_algorithm.compare_models import (
    CompareMultiArmedBandits,
    MultiArmedBanditModel,
    Algorithms,
)
from multiarmedbandits.environments import (
    BaseBanditEnv,
    DistParameter,
    ArmDistTypes,
)
from multiarmedbandits.algorithms import GradientBaseLineAttr, BaseLinesTypes
from multiarmedbandits.run_algorithm.utils import (
    MetricNames,
)

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
algo_one = MultiArmedBanditModel(
    dist_type=Algorithms.GRADIENTBANDIT,
    dist_params={
        "alpha": 0.1,
        "baseline_attr": GradientBaseLineAttr(type=BaseLinesTypes.MEAN),
    },
)
algo_two = MultiArmedBanditModel(
    dist_type=Algorithms.GRADIENTBANDIT,
    dist_params={
        "alpha": 0.1,
        "baseline_attr": GradientBaseLineAttr(type=BaseLinesTypes.ZERO),
    },
)
algo_three = MultiArmedBanditModel(
    dist_type=Algorithms.GRADIENTBANDIT,
    dist_params={
        "alpha": 0.2,
        "baseline_attr": GradientBaseLineAttr(type=BaseLinesTypes.MEAN),
    },
)
algo_four = MultiArmedBanditModel(
    dist_type=Algorithms.GRADIENTBANDIT,
    dist_params={
        "alpha": 0.2,
        "baseline_attr": GradientBaseLineAttr(type=BaseLinesTypes.ZERO),
    },
)
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
