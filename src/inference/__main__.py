import argparse

from .predict import ThresholdConfig
from .submission import create_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a submission from a saved run")
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--data-path")
    parser.add_argument("--template-path")
    parser.add_argument("--device")
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--cls-threshold", type=float)
    parser.add_argument("--min-area", type=float)
    parser.add_argument("--area-cap", type=float, default=0.0,
                        help="0 zeroes a gated frame; above 0 clips it below this area instead")
    args = parser.parse_args()
    values = (args.mask_threshold, args.cls_threshold, args.min_area)
    if any(value is not None for value in values) and not all(value is not None for value in values):
        parser.error("provide all three thresholds, or omit them to use the run summary")
    if args.area_cap and values[0] is None:
        parser.error("--area-cap only applies with explicit thresholds; the run summary carries its own")
    thresholds = ThresholdConfig(*values, area_cap=args.area_cap) if values[0] is not None else None
    path = create_submission(args.run_dir, args.output_dir, thresholds=thresholds,
                             data_path=args.data_path, template_path=args.template_path, device=args.device)
    print(path)


if __name__ == "__main__":
    main()
