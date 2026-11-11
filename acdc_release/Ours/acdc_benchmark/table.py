"""Assemble the paper table from completed, paired ACDC evaluations."""
import csv
import json
import math
from pathlib import Path

from .evaluate import METHODS, PROTOCOL, SPLIT_SHA256, aggregate, validate_frames, write_csv
from .metrics import ECE_CHANNELS, ECE_DEFINITION, ECE_SOURCE_SHA256, EXTRA_METRICS, METRICS

IDENTITY_FIELDS = ("ece_support_sha256", "ece_target_sha256", "valid_image_sha256",
                   "reference_foreground_sha256")


def read_metrics(path, metrics=METRICS):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for metric in metrics:
            row[metric] = float(row[metric]) if row[metric] else None
        ece = row["ECE"]
        channels = [row[name] for name in ECE_CHANNELS if name in metrics]
        if channels:
            valid = (all(value is not None and 0 <= value <= 1 for value in channels)
                     and ece is not None and math.isclose(ece, sum(channels) / len(channels), abs_tol=1e-12))
        else:
            valid = ece is not None and ece == float(row["fixed_foreground_ECE"])
        if not valid or not 0 <= ece <= 1:
            raise ValueError(f"{path}/{row['case']}: expected matching raw ECE in [0, 1]")
    return rows


def manifest(path, method):
    result = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    if result["method"] != method:
        raise ValueError(f"{path}: expected method {method}")
    return result


def build_table(results_dir):
    baseline = manifest(results_dir / "baseline", "baseline")
    protocol = baseline["comparison"]
    metrics = METRICS + EXTRA_METRICS if protocol.get("extended_metrics") else METRICS
    identity_fields = IDENTITY_FIELDS
    if "ece_definition" not in protocol:
        metrics = tuple(metric for metric in metrics if metric not in ECE_CHANNELS)
    elif protocol["ece_definition"] == ECE_DEFINITION:
        identity_fields += ("fixed_foreground_support_sha256",)
    else:
        raise ValueError("Unknown ECE definition in the baseline protocol")
    if protocol.get("ueo"):
        metrics += ("UEO",)
    if protocol.get("error_auroc"):
        metrics += ("error_AUROC",)
    frames = read_metrics(results_dir / "baseline/frame_metrics.csv", metrics)
    expected = {(r["case"], r["patient_id"], r["frame"]) for r in frames}
    expected_size = {"val": 26, "test": 66}.get(protocol["split"])
    if (protocol["protocol"] != PROTOCOL or protocol["ece_source_sha256"] != ECE_SOURCE_SHA256
            or protocol["split_sha256"] != SPLIT_SHA256
            or len(expected) != expected_size or {r[0] for r in expected} != set(baseline["frames"])):
        raise ValueError("The baseline protocol or cohort does not match this evaluator")
    results = []
    for method in METHODS:
        directory = results_dir / method
        saved = manifest(directory, method)
        if (saved["comparison"] != protocol or saved["frames"] != baseline["frames"]
                or saved["reference_checkpoint_sha256"] != baseline["checkpoint_sha256"]):
            raise ValueError(f"{method}: protocol, reference checkpoint or frame identities differ")
        rows = read_metrics(directory / "frame_metrics.csv", metrics)
        validate_frames(rows, expected)
        for row in rows:
            identity = {name: row[name] for name in identity_fields}
            if identity != baseline["frames"][row["case"]]:
                raise ValueError(f"{method}/{row['case']}: CSV and fixed frame identity differ")
        _, domain = aggregate(rows, metrics)
        extra = {"ueo_threshold": saved["ueo_threshold"]} if protocol.get("ueo") else {}
        results.append({"method": method, **domain, **extra})
    return results


def run(args):
    rows = build_table(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)


def add_arguments(parser):
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
