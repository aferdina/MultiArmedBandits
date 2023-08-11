"""run multiarmed bandit comparisons"""
from dataclasses import dataclass
import argparse
from multiarmedbandits.run_algorithm.compare_models import CompareMultiArmedBandits
import multiarmedbandits.environments as mab_envs


@dataclass
class MabArgs:
    env_dist_type: str
    max_steps: int
    env_dist_params: dict
    no_of_runs: int
    plot_metrics: bool
    save_metrics: bool


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-dist-type",
        help="distribution type for environment",
        default="gaussian",
        type=str,
        required=False,
        choices=[member.value for member in mab_envs.ArmDistTypes],
    )
    parser.add_argument(
        "--env-dist-parameter",
        type=dict,
        default="LogisticGame",
        help="parameter for distribution arms",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="maximal steps to run in multiarmedbandit",
    )
    parser.add_argument(
        "--no-of-runs",
        type=int,
        default=1000,
        help="runs to iterate over the trainings process",
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
        default=False,
        help="bool if to store the metrics",
    )
    args: MabArgs = MabArgs(vars(**parser.parse_args()))

    bandit_env = mab_envs.BaseBanditEnv(
        distr_params=mab_envs.DistParameter(dist_type=args.env_dist_type)
    )
    compare_models = CompareMultiArmedBandits(test_env=bandit_env, mab_algorithms=[])

    metrics = compare_models.train_all_models(no_of_runs=args.no_of_runs)
    if args.plot_metrics:
        compare_models.plot_multiple_mabs(named_metrics=metrics)


if __name__ == "__main__":
    train()
