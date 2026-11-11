"""Summarize frozen and adapted ACDC perturbation runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate import write_csv
from .perturbation import VARIANTS


def paired_auc(current, reference):
    if current.keys() != reference.keys():
        raise ValueError("Paired AUROC requires the same patient cohort")
    differences = np.asarray([float(current[p]["native_AUROC"]) - float(reference[p]["native_AUROC"])
                              for p in sorted(current)
                              if current[p]["native_AUROC"] and reference[p]["native_AUROC"]])
    if len(differences) < 2:
        raise ValueError("Too few paired patients for an AUROC interval")
    rng = np.random.default_rng(42)
    sample = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(sample, [.025, .975])
    return dict(delta_native_AUROC=float(differences.mean()), ci_low=float(low),
                ci_high=float(high), paired_patients=len(differences))


def fixed_adapter_ablation(control_dir, online_dir):
    control, online = [json.loads((directory / "protocol.json").read_text())
                       for directory in (control_dir, online_dir)]
    expected = {**online["comparison"], "tta": False, "fixed_adapter_mc": True}
    if (online["comparison"]["tta"] is not True or control["comparison"] != expected
            or any(control[key] != online[key] for key in
                   ("method", "checkpoint_sha256", "reference_checkpoint_sha256", "frames"))):
        raise ValueError("Fixed adapter and online TTA must share weights, cohort and prediction protocol")
    if control["domain"]["tta_optimizer_updates"] != 0 or online["domain"]["tta_optimizer_updates"] < 1:
        raise ValueError("Expected only online TTA to perform updates")
    metrics = ("Dice", "ECE", "HD95", "AURC", "native_AUROC", "error_AUROC", "head1_Dice", "head2_Dice")
    rows = []
    for mode, saved in (("fixed_adapter_mc", control), ("online_tta", online)):
        rows.append(dict(mode=mode, **{metric: saved["domain"][metric] for metric in metrics}))
    rows.append(dict(mode="online_minus_fixed", **{metric: rows[1][metric] - rows[0][metric]
                     for metric in metrics if rows[0][metric] is not None and rows[1][metric] is not None}))
    write_csv(control_dir / "test_fixed_adapter_ablation.csv", rows)


def build(root, split, variants=VARIANTS, control_variant="dropout"):
    values, patients = {}, {}
    reference = None
    for variant in variants:
        for mode in ("frozen", "tta"):
            key = variant, mode
            path = root / split / variant / mode
            manifest = json.loads((path / "protocol.json").read_text())
            protocol = manifest["comparison"].copy()
            if protocol.pop("head_perturbation") != variant or protocol.pop("tta") != (mode == "tta"):
                raise ValueError(f"Unexpected evaluation protocol at {path}")
            identity = protocol, manifest["reference_checkpoint_sha256"], manifest["frames"]
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError(f"Fixed evaluation reference changed at {path}")
            with (path / "patient_metrics.csv").open(newline="") as stream:
                patients[key] = {row["patient_id"]: row for row in csv.DictReader(stream)}
            values[key] = dict(manifest["domain"],
                               tta_batches=manifest["domain"].get("tta_batches", 0),
                               tta_optimizer_updates=manifest["domain"].get("tta_optimizer_updates", 0))
            if "ueo_threshold" in manifest:
                values[key]["ueo_threshold"] = manifest["ueo_threshold"]
    rows, effects = [], []
    for (variant, mode), metrics in values.items():
        control = control_variant, mode
        row = dict(variant=variant, mode=mode, **metrics)
        if control in values:
            dice_delta = metrics["Dice"] - values[control]["Dice"]
            row.update({f"dice_delta_vs_{control_variant}": dice_delta, "dice_cost": dice_delta < -0.01})
            if variant != control_variant:
                effects.append(dict(variant=variant, comparison=f"{mode}_vs_{control_variant}",
                                    **paired_auc(patients[variant, mode], patients[control])))
        rows.append(row)
        if mode == "tta":
            effects.append(dict(variant=variant, comparison="tta_vs_frozen",
                                **paired_auc(patients[variant, mode], patients[variant, "frozen"])))
    write_csv(root / f"{split}_table.csv", rows)
    write_csv(root / f"{split}_paired_auc.csv", effects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("split")
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--control", default="dropout")
    parser.add_argument("--online-tta-dir", type=Path,
                        help="Compare the fixed-adapter MC evaluation at ROOT with this online run")
    args = parser.parse_args()
    if args.online_tta_dir is None:
        build(args.root, args.split, args.variants or VARIANTS, args.control)
    else:
        fixed_adapter_ablation(args.root, args.online_tta_dir)
