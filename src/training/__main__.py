"""Run one configured experiment, or a loss-ablation series across several configs."""

import argparse

from src.config import load_experiment_config
from src.training.ablation import LossAblation
from src.training.engine import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='*', help='one or more experiment config paths')
    parser.add_argument('--config', help='single experiment config path (legacy alias)')
    parser.add_argument('--data-path', help="override paths.data_path for every config")
    parser.add_argument('--csv', help='write the ablation comparison table to this path')
    args = parser.parse_args()

    paths = list(args.configs) + ([args.config] if args.config else [])
    if not paths:
        parser.error('provide at least one config path, or --config')

    if len(paths) == 1 and not args.csv and not args.data_path:
        ExperimentRunner(load_experiment_config(paths[0])).run()
        return

    ablation = LossAblation(paths, data_path=args.data_path)
    results = ablation.run()
    table = LossAblation.compare(results)
    if args.csv:
        table.to_csv(args.csv, index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
