"""Evaluate ACDC checkpoints on the same fixed foreground and geometry."""
import argparse
import csv
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .metrics import (ECE_SOURCE_SHA256, EXTRA_METRICS, METRICS, UEO_THRESHOLDS,
                      average, error_auroc, extended_score, fixed_support, rank_auroc, score, support_sha256,
                      uncertainty_error_overlap)

ACDC_ROOT = Path(__file__).resolve().parents[2] / "ACDC"
METHODS = ("baseline", "nacl", "ml1_ace", "mbls", "edl", "devis", "fgrm", "ours")
EXISTING_ADAPTERS = ("ml1_ace", "devis", "fgrm")
PROTOCOL = "acdc_fixed_reference_foreground_v1"
SPLIT_SHA256 = "eedc8967964d2b71719738efb2ecd8e834efc222c5b55a2fe70939b2c9d29420"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_frames(rows, expected):
    keys = [(r["case"], r["patient_id"], r["frame"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Frame keys differ from the locked split or contain duplicates")


def aggregate(rows, metrics=METRICS):
    groups = {}
    for row in rows:
        groups.setdefault(row["patient_id"], []).append(row)
    patients = []
    for patient_id, frames in sorted(groups.items()):
        if len(frames) != 2 or {r["frame"] for r in frames} != {"ED", "ES"}:
            raise ValueError(f"{patient_id}: expected exactly one ED and one ES frame")
        patients.append({"patient_id": patient_id, "n_frames": len(frames),
                         **{metric: average(r[metric] for r in frames) for metric in metrics}})
    result = {"n_frames": len(rows), "n_patients": len(patients),
              **{metric: average(p[metric] for p in patients) for metric in metrics},
              **{f"{metric}_valid_patients": sum(p[metric] is not None for p in patients)
                 for metric in metrics}}
    if "HD95" in metrics:
        for name in ("RV", "MYO", "LV"):
            result[f"miss_{name}_frames"] = sum(r[f"miss_{name}"] or 0 for r in rows)
            result[f"hd95_{name}_valid_frames"] = sum(r[f"hd95_{name}"] is not None for r in rows)
    if "error_AUROC" in metrics:
        scores = [p["error_AUROC"] for p in patients if p["error_AUROC"] is not None]
        result["error_AUROC_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
        result["error_AUROC_valid_frames"] = sum(r["error_AUROC"] is not None for r in rows)
    return patients, result


def read_reference(directory, protocol, expected):
    manifest = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if manifest["method"] != "baseline" or manifest["comparison"] != protocol:
        raise ValueError("Frozen reference protocol differs from the requested evaluation")
    if set(manifest["frames"]) != {r[0] for r in expected}:
        raise ValueError("Frozen reference frame set differs from the locked split")
    return manifest


def score_frame(probability, labels, valid, record, output, reference, manifest,
                geometry=None, ueo_thresholds=None, include_error_auroc=False):
    case = record["subject_id"]
    if manifest is None:
        foreground = valid & (probability.argmax(0) > 0)
        path = output / "support" / f"{case}.npz"
        np.savez_compressed(path, foreground=foreground)
    else:
        path = reference / "support" / f"{case}.npz"
        with np.load(path, allow_pickle=False) as saved:
            foreground = saved["foreground"]
    values = score(probability, labels, valid, foreground)
    if values["ECE"] is None:
        raise ValueError(f"{case}: empty fixed ECE support; cannot publish a complete cohort")
    if geometry is not None:
        values.update(extended_score(probability, labels, valid, foreground, record, *geometry))
    if ueo_thresholds is not None or include_error_auroc:
        support = fixed_support(labels, valid, foreground)
    if ueo_thresholds is not None:
        values["UEO"] = uncertainty_error_overlap(
            probability, labels, support, ueo_thresholds)
    if include_error_auroc:
        values["error_AUROC"] = error_auroc(probability, labels, support)
    identity = {"ece_support_sha256": values["ece_support_sha256"],
                "ece_target_sha256": values["ece_target_sha256"],
                "valid_image_sha256": support_sha256(valid),
                "reference_foreground_sha256": support_sha256(foreground)}
    if manifest is not None and identity != manifest["frames"][case]:
        raise ValueError(f"{case}: fixed reference support, labels or valid image changed")
    row = {"case": case, "patient_id": record["patient_id"], "frame": record["frame"],
           "domain": record["domain"], **values, **identity}
    return row, identity


def split_sha256(path):
    content = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if digest != SPLIT_SHA256:
        raise ValueError(f"ACDC split differs from the locked 1.5T/3T manifest: {path}")
    return digest


def acdc_config(args):
    args.acdc_root = args.acdc_root.expanduser().resolve()
    sys.path.insert(0, str(args.acdc_root))
    from ours.config import load_config

    args.data_root = args.data_root or args.acdc_root / "database"
    args.split_json = args.split_json or args.acdc_root / "baseline/jsons/acdc_domain_shift.json"
    return load_config(str(args.acdc_root / "ours/configs/acdc.yaml"), [
        f"paths.data_root={args.data_root.resolve()}",
        f"paths.split_json={args.split_json.resolve()}",
    ])


def valid_field(image_path, shape, roi, spacing):
    import SimpleITK as sitk
    from ours.data import _center_crop_xy, _padding

    header = sitk.ImageFileReader()
    header.SetFileName(str(image_path))
    header.ReadImageInformation()
    resampled = tuple(reversed(tuple(int(round(size * old / new))
                      for size, old, new in zip(header.GetSize(), header.GetSpacing(), spacing))))
    valid = np.pad(np.ones(resampled, dtype=np.bool_), _padding(resampled, roi))
    valid, _ = _center_crop_xy(valid, valid)
    if valid.shape != shape:
        raise ValueError(f"Valid-field shape differs from preprocessed image: {image_path}")
    return valid


def model_options(config):
    return dict(roi_size=tuple(config["data"]["roi_size"]),
                feature_size=config["model"]["feature_size"],
                use_checkpoint=config["model"]["use_checkpoint"],
                dropout_rate=config["model"]["dropout_rate"],
                attn_drop_rate=config["model"]["attention_dropout_rate"],
                dropout_path_rate=config["model"]["dropout_path_rate"])


def new_method_predictor(model, method, roi, device, amp, acdc_root):
    from ours.inference import autocast_context
    from torch.nn.functional import softmax
    from .losses import edl_probability

    path = acdc_root / "sota/ml1_ace/inference.py"
    spec = spec_from_file_location("acdc_adapter_inference", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    def transform(logits):
        return edl_probability(logits) if method == "edl" else softmax(logits.float(), dim=1)

    def predict(image):
        with autocast_context(device, amp):
            return module.sliding_window_inference(model, image, roi, transform)
    return predict


def load_predictor(args, config, device):
    from ours.inference import autocast_context

    if args.method == "ours":
        from ours.checkpoint import load_checkpoint
        from ours.inference import infer_volume
        from ours.model import DualHeadSwinUNETR

        model = DualHeadSwinUNETR(config).to(device)
        payload = load_checkpoint(str(args.checkpoint), model, strict=True)
        model.eval()
        if getattr(args, "perturbation", None):
            from .episodic import predict_frame

            trained = payload["config"]["model"].get("head_perturbation", "dropout")
            if trained != args.perturbation:
                raise ValueError(f"Checkpoint used {trained}, requested {args.perturbation}")
            return lambda image: predict_frame(model, image, config, args.tta, infer_volume)

        def predict(image):
            return infer_volume(model, image, config, compute_uncertainty=False).base_class_probability
        return predict

    # The legacy standalone projects use top-level imports; one method per CLI process.
    method_root = (args.acdc_root / "sota" / args.method if args.method in EXISTING_ADAPTERS
                   else args.acdc_root / "baseline")
    sys.path.insert(0, str(method_root))
    options = model_options(config)
    parser = argparse.ArgumentParser()
    if args.method in EXISTING_ADAPTERS:
        from method.adapter import MethodAdapter
        from checkpoint import evaluation_state, safe_load
        from inference import sliding_window_inference

        MethodAdapter.add_method_args(parser)
        checkpoint = safe_load(str(args.checkpoint), map_location="cpu")
        method_options = vars(parser.parse_args([]))
        saved_options = checkpoint.get("args", {})
        if not isinstance(saved_options, dict):
            saved_options = vars(saved_options)
        method_options.update({k: saved_options[k] for k in method_options if k in saved_options})
        adapter = MethodAdapter(argparse.Namespace(**options, **method_options))
        adapter.validate_method_state(checkpoint.get("method_state"))
        if args.method == "fgrm" and checkpoint.get("phase") != "fgrm":
            raise ValueError("FGRM needs a checkpoint after reward fine-tuning")
        model = adapter.model
        model.load_state_dict(evaluation_state(checkpoint, prefer_ema=True), strict=True)
        infer = partial(sliding_window_inference, model, window_size=options["roi_size"],
                        output_transform=adapter.logits_to_inference)
    else:
        from checkpoint import evaluation_state, safe_load
        from model import build_model

        checkpoint = safe_load(str(args.checkpoint), map_location="cpu")
        if args.method != "baseline" and checkpoint.get("method_state", {}).get("method") != args.method:
            raise ValueError("Checkpoint does not belong to the requested benchmark method")
        model = build_model(argparse.Namespace(**options))
        model.load_state_dict(evaluation_state(checkpoint, prefer_ema=True), strict=True)
        if args.method != "baseline":
            return new_method_predictor(model.to(device).eval(), args.method,
                                        options["roi_size"], device, args.amp, args.acdc_root)
        from inference import sliding_window_inference
        infer = partial(sliding_window_inference, model, window_size=options["roi_size"])
    model.to(device).eval()

    def predict(image):
        # Convert patch probabilities to FP32 before fusion for existing evidential adapters.
        with autocast_context(device, args.amp):
            return infer(image)[:, :4].float()
    return predict


def comparison_protocol(args, config):
    return {"protocol": PROTOCOL, "split": args.split, "split_sha256": split_sha256(args.split_json),
            "ece_source_sha256": ECE_SOURCE_SHA256,
            "roi_size": list(config["data"]["roi_size"]),
            "target_spacing": list(config["data"]["target_spacing"]),
            "class_order": ["BG", "RV", "MYO", "LV"],
            "amp_requested": args.amp}


def cohort(args):
    import torch

    config = acdc_config(args)
    from ours.data import preprocess_case, read_records
    protocol = comparison_protocol(args, config)
    records = read_records(str(args.split_json), str(args.data_root), args.split)
    expected = {(r["subject_id"], r["patient_id"], r["frame"]) for r in records}
    if len(expected) != len(records) or len({r[0] for r in expected}) != len(records):
        raise ValueError("The split must contain unique frame identifiers")
    roi, spacing = tuple(protocol["roi_size"]), tuple(protocol["target_spacing"])

    def cases():
        for record in records:
            image, label, _ = preprocess_case(record["image"], record["label"], roi, spacing)
            labels = label.numpy().astype(np.uint8)
            valid = valid_field(record["image"], labels.shape, roi, spacing)
            yield record, image[None, None].to(dtype=torch.float32), labels, valid
    return config, protocol, expected, cases


def run(args):
    import torch
    if args.tta and not args.perturbation:
        raise ValueError("--tta requires --perturbation")
    if args.perturbation and (args.method != "ours" or not args.error_auroc):
        raise ValueError("--perturbation requires --method ours --error-auroc")

    if args.method == "baseline" and args.reference is not None:
        raise ValueError("The baseline command produces the reference; omit --reference")
    if args.method != "baseline" and args.reference is None:
        raise ValueError("All comparator methods require --reference from the baseline evaluation")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu only for a compatible local run")
    config, protocol, expected, cases = cohort(args)
    geometry = None
    if args.extended_metrics:
        protocol["extended_metrics"] = "native_hd95_tie_grouped_fixed_support_aurc_v1"
        geometry = (tuple(protocol["roi_size"]), tuple(protocol["target_spacing"]))
    if args.error_auroc:
        protocol["error_auroc"] = "fixed_support_top_label_midranks_patient_macro_v1"
    config["training"]["amp"] = args.amp
    if args.perturbation:
        config["model"]["head_perturbation"] = args.perturbation
        protocol.update(head_perturbation=args.perturbation, tta=args.tta)
        torch.manual_seed(42)
    predict = load_predictor(args, config, device)
    checkpoint_hash = sha256(args.checkpoint)
    ueo_thresholds = None
    if args.ueo:
        if geometry is None:
            raise ValueError("--ueo requires --extended-metrics")
        protocol["ueo"] = "fixed_support_max_probability_val_grid_v1"
        if args.split == "val" and args.ueo_validation is None:
            ueo_thresholds = UEO_THRESHOLDS
        elif args.split == "test" and args.ueo_validation is not None:
            fit = json.loads(args.ueo_validation.read_text())
            if (fit["comparison"] != {**protocol, "split": "val"} or fit["method"] != args.method
                    or fit["checkpoint_sha256"] != checkpoint_hash
                    or fit["ueo_threshold"] not in UEO_THRESHOLDS):
                raise ValueError("UEO requires a matching 1.5T threshold fit")
            ueo_thresholds = np.array([fit["ueo_threshold"]])
        else:
            raise ValueError("Provide --ueo-validation only for UEO on test")
    reference_protocol = {k: v for k, v in protocol.items() if k not in ("head_perturbation", "tta")}
    reference = None if args.reference is None else read_reference(args.reference, reference_protocol, expected)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if reference is None:
        (args.output_dir / "support").mkdir()
    rows, frame_identity = [], {}
    inference_context = torch.no_grad if args.tta else torch.inference_mode
    with inference_context():
        for record, image, labels, valid in cases():
            prediction = predict(image.to(device))
            if args.perturbation:
                output, steps = prediction
                prediction = output.base_class_probability
            probability = prediction[0].float().cpu().numpy()
            row, identity = score_frame(probability, labels, valid, record, args.output_dir,
                                        args.reference, reference, geometry, ueo_thresholds,
                                        args.error_auroc)
            if args.perturbation:
                with np.load(args.reference / "support" / f'{record["subject_id"]}.npz',
                             allow_pickle=False) as stored:
                    support = fixed_support(labels, valid, stored["foreground"])
                native = output.uncertainty[0, 0].float().cpu().numpy()
                head1, head2 = output.head_class_probabilities
                difference = (head1 - head2).abs().mean(1)[0].float().cpu().numpy()
                row.update(native_AUROC=rank_auroc(native[support],
                                                   probability[:, support].argmax(0) != labels[support]),
                           head_probability_l1=float(difference[support].mean()), tta_steps=steps)
            rows.append(row)
            frame_identity[row["case"]] = identity
            print(f"{args.method} {len(rows)}/{len(expected)} {row['case']} ECE={row['ECE']}", flush=True)
    validate_frames(rows, expected)
    metrics = METRICS + EXTRA_METRICS if geometry else METRICS
    if args.error_auroc:
        metrics += ("error_AUROC",)
    if args.perturbation:
        metrics += ("native_AUROC", "head_probability_l1", "tta_steps")
    ueo_metadata = {}
    if ueo_thresholds is not None:
        # The locked split has exactly two frames per patient, so frame means
        # and patient means have identical weights for choosing the threshold.
        means = np.stack([row["UEO"] for row in rows]).mean(axis=0)
        index = len(means) - 1 - int(np.argmax(means[::-1]))
        ueo_metadata["ueo_threshold"] = float(ueo_thresholds[index])
        for row in rows:
            row["UEO"] = float(row["UEO"][index])
        metrics += ("UEO",)
    patients, domain = aggregate(rows, metrics)
    if args.tta:
        domain["tta_skipped_frames"] = sum(row["tta_steps"] == 0 for row in rows)
    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "patient_metrics.csv", patients)
    write_csv(args.output_dir / "domain_metrics.csv", [{"method": args.method, **domain}])
    write_json(args.output_dir / "protocol.json", {"method": args.method, "comparison": protocol,
               "checkpoint_sha256": checkpoint_hash, "checkpoint": str(args.checkpoint.resolve()),
               "reference_checkpoint_sha256": checkpoint_hash if reference is None else reference["checkpoint_sha256"],
               "frames": frame_identity, "domain": domain, **ueo_metadata})


def add_acdc_args(parser):
    parser.add_argument("--acdc-root", type=Path, default=ACDC_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)


def add_arguments(parser):
    add_acdc_args(parser)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extended-metrics", action="store_true")
    parser.add_argument("--ueo", action="store_true")
    parser.add_argument("--ueo-validation", type=Path)
    parser.add_argument("--error-auroc", action="store_true")
    from .perturbation import VARIANTS
    parser.add_argument("--perturbation", choices=VARIANTS)
    parser.add_argument("--tta", action="store_true")
