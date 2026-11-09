"""Train NACL, MbLS or standard EDL using the existing ACDC recipe."""
import argparse
from copy import deepcopy
import random
import sys

from . import evaluate


def run(args):
    import numpy as np
    import torch
    from .losses import TrainingAdapter

    if min(args.max_epochs, args.val_every, args.iter_per_epoch, args.batch_size,
           args.edl_annealing_epochs) <= 0:
        raise ValueError("Training schedules and batch sizes must be positive")
    if min(args.margin, args.margin_weight, args.nacl_weight) < 0:
        raise ValueError("Margin and calibration weights must be nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The ACDC training recipe requires CUDA")
    torch.cuda.set_device(device)
    args.split = "val"
    config, protocol, expected, cases = evaluate.cohort(args)
    reference = evaluate.read_reference(args.reference, protocol, expected)
    sys.path.insert(0, str(args.acdc_root / "sota/ml1_ace"))
    from checkpoint import save_checkpoint
    from data import create_loader
    from model import build_swin_unetr
    from trainer import make_scaler, set_learning_rate, train_epoch

    options = evaluate.model_options(config)
    for name, value in options.items():
        setattr(args, name, value)
    args.target_spacing = tuple(config["data"]["target_spacing"])
    args.data_dir, args.json_list = str(args.data_root), str(args.split_json)
    args.device = device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_swin_unetr(args, 4).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    adapter = TrainingAdapter(model, args)
    predictor = evaluate.new_method_predictor(ema, args.method, args.roi_size, device, args.amp,
                                               args.acdc_root)
    loader = create_loader(args, "train")
    optimizer = torch.optim.AdamW(adapter.trainable_parameters(), lr=args.base_lr,
                                   betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-5)
    scaler = make_scaler(args.amp) if args.amp else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    method_state = {"method": args.method, "implementation_version": 1}
    best = None
    history = []
    for epoch in range(args.max_epochs):
        set_learning_rate(optimizer, args.base_lr, epoch, args.max_epochs)
        train_epoch(adapter, model, ema, loader, optimizer, scaler, args, epoch)
        if (epoch + 1) % args.val_every and epoch + 1 != args.max_epochs:
            continue
        rows = []
        with torch.inference_mode():
            for record, image, labels, valid in cases():
                p = predictor(image.to(device))[0].float().cpu().numpy()
                row, _ = evaluate.score_frame(p, labels, valid, record, args.output_dir,
                                               args.reference, reference)
                rows.append(row)
        evaluate.validate_frames(rows, expected)
        _, metrics = evaluate.aggregate(rows)
        dice, ece = metrics["Dice"], metrics["ECE"]
        if dice is None or ece is None or metrics["ECE_valid_patients"] != metrics["n_patients"]:
            raise ValueError("Source validation requires complete patient-level Dice and ECE")
        key = (dice, -ece)
        history.append({"epoch": epoch + 1, **metrics})
        evaluate.write_csv(args.output_dir / "validation_metrics.csv", history)
        print(f"Source validation epoch {epoch + 1}: Dice={dice:.6f} ECE={ece:.6f}", flush=True)
        if best is None or key > best:
            best = key
            save_checkpoint(args.output_dir / "model_best.pt", model, ema, optimizer, scaler,
                            epoch + 1, dice, ece, args, method_state, "dice_best")
    evaluate.write_json(args.output_dir / "training_protocol.json", {
        "method": args.method, "selection": "source validation Dice, then fixed foreground ECE",
        "comparison": protocol, "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "seed": args.seed, "max_epochs": args.max_epochs,
    })


def add_arguments(parser):
    from pathlib import Path

    evaluate.add_acdc_args(parser)
    parser.add_argument("--method", choices=("nacl", "mbls", "edl"), required=True)
    parser.add_argument("--reference", required=True, type=Path, help="Frozen Swin val reference")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--iter-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--ema-alpha", type=float, default=0.99)
    parser.add_argument("--nacl-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument("--edl-annealing-epochs", type=int, default=10)
