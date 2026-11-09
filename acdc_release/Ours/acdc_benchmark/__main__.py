"""Build and evaluate the ACDC trustworthy segmentation comparison."""
import argparse
import json
from pathlib import Path


def smoke(output_dir):
    import numpy as np
    from .metrics import FixedECEAccumulator, fixed_support, score

    labels = np.array([[[0, 1, 2, 3, 0, 0]]], dtype=np.uint8)
    probabilities = np.array([[[[.95, .1, .95, .1, .1, .1]]],
                              [[[.02, .7, .02, .1, .7, .7]]],
                              [[[.02, .1, .02, .1, .1, .1]]],
                              [[[.01, .1, .01, .7, .1, .1]]]], dtype=np.float32)
    valid = np.array([[[True, True, True, True, True, False]]])
    reference = np.array([[[True, True, False, True, False, True]]])
    observed = score(probabilities, labels, valid, reference)
    expected = FixedECEAccumulator()
    expected.add(probabilities, labels, fixed_support(labels, valid, reference))
    for key in ("ECE", "fixed_foreground_NLL", "fixed_foreground_Brier",
                "high_confidence_error_fraction"):
        assert observed[key] == expected.result()[key], key
    assert observed["ece_voxels"] == 4
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_smoke.json").write_text(
        json.dumps({"synthetic": True, **observed}, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", required=True, type=Path)
    from . import evaluate, table, train

    evaluate.add_arguments(commands.add_parser("evaluate"))
    train.add_arguments(commands.add_parser("train"))
    table.add_arguments(commands.add_parser("table"))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.output_dir)
    elif args.command == "evaluate":
        evaluate.run(args)
    elif args.command == "train":
        train.run(args)
    elif args.command == "table":
        table.run(args)


if __name__ == "__main__":
    main()
