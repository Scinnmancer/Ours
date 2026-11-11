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
    args = parser.parse_args()
    build(args.root, args.split, args.variants or VARIANTS, args.control)
