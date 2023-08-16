"""run multiarmed bandit comparisons"""
import argparse
from dataclasses import dataclass
from typing import Any, List, Tuple

import yaml
from strenum import StrEnum

import multiarmedbandits.algorithms as mab_algos
import multiarmedbandits.environments as mab_envs
from multiarmedbandits.run_algorithm.compare_models import Algorithms, CompareMultiArmedBandits, MultiArmedBanditModel
from multiarmedbandits.run_algorithm.config_utils import SEQUENCETAG, add_constructors, sequence_constructor
from multiarmedbandits.run_algorithm.utils import MetricNames


class EnvAlgoConfigs(StrEnum):
    """dictionary names for env algo configurations"""

    MAB_ENV = "mab_env"
    MAB_ALGOS = "mab_algos"
    NO_OF_RUNS = "no_of_runs"
    METRICS_TO_PLOT = "metrics_to_plot"


def read_mab_env_and_algos(
    configs_path: str,
) -> Tuple[str, mab_envs.BaseBanditEnv, List[mab_algos.BaseModel], int, List[MetricNames]]:
    """read multi armed bandit environment and multiarmed bandit models

    Args:
        configs_path (str): path to configs for multiarmed bandit models

    Returns:
        Tuple[mab_envs.BaseBanditEnv, List[mab_algos.BaseModel, int]]: Environment to run
        algorithms, algorithms to run, max number of runs
    """
    print(f"Loading hyperparameters from: {configs_path}")
    yaml.add_constructor(SEQUENCETAG, sequence_constructor)
    add_constructors(
        [
            mab_algos.EpsilonGreedy,
            mab_envs.BaseBanditEnv,
            mab_envs.DistParameter,
            mab_envs.ArmDistTypes,
            MetricNames,
            Algorithms,
            MultiArmedBanditModel,
        ]
    )
    if configs_path.endswith(".yml") or configs_path.endswith(".yaml"):
        # Load hyperparameters from yaml file
        with open(configs_path, encoding="utf-8") as file:
            experiment: dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)
    exp_name = next(iter(experiment.keys()))
    mab_env = experiment[exp_name][EnvAlgoConfigs.MAB_ENV]
    mab_algorithm = experiment[exp_name][EnvAlgoConfigs.MAB_ALGOS]
    no_of_runs = experiment[exp_name][EnvAlgoConfigs.NO_OF_RUNS]
    metrics_to_plot = experiment[exp_name][EnvAlgoConfigs.METRICS_TO_PLOT]
    return exp_name, mab_env, mab_algorithm, no_of_runs, metrics_to_plot


@dataclass
class MabArgs:
    """dataclass to store information from argument parser"""

    plot_metrics: bool
    save_metrics: bool
    store_path: str
    config_path: str


def train() -> None:
    """main file"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="multiarmedbandits/run_algorithm/trainings_configs/epsilon.yaml",
        help="config path for models",
    )
    parser.add_argument(
        "--plot-metrics",
        type=bool,
        default=False,
        help="bool if to run the plot metrics",
    )
    parser.add_argument(
        "--save-metrics",
        type=bool,
        default=True,
        help="bool if to store the metrics",
    )
    parser.add_argument(
        "--store-path",
        type=str,
        default="result",
        help="path to store the metrics",
    )
    args: MabArgs = MabArgs(**vars(parser.parse_args()))
    (
        exp_name,
        mab_env,
        mab_algorithm,
        no_of_runs,
        metrics_to_plot,
    ) = read_mab_env_and_algos(configs_path=args.config_path)
    print(f"experiment to run: {exp_name}")
    compare_models = CompareMultiArmedBandits(test_env=mab_env, mab_algorithms=mab_algorithm)

    metrics = compare_models.train_all_models(no_of_runs=no_of_runs)
    if args.plot_metrics:
        compare_models.plot_multiple_mabs(named_metrics=metrics, metrics_to_plot=metrics_to_plot)
    if args.save_metrics:
        for metric in metrics:
            compare_models.store_metric(
                named_metric=metric,
                file_path=args.store_path,
                metrics_to_store=metrics_to_plot,
            )


if __name__ == "__main__":
    train()
