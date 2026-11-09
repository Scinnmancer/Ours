"""Run the real ACDC training and adaptation paths before launching the grid."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .episodic import predict_frame
from .evaluate import add_acdc_args, cohort
from .metrics import score
from .perturbation import VARIANTS, configure_heads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_acdc_args(parser)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    args.split = "val"
    config, _, _, cases = cohort(args)
    from ours.checkpoint import load_checkpoint
    from ours.inference import infer_volume
    from ours.model import DualHeadSwinUNETR

    torch.manual_seed(42)
    device = torch.device(args.device)
    model = DualHeadSwinUNETR(config).to(device)
    load_checkpoint(args.checkpoint, model, strict=True)
    record, image, labels, valid = next(cases())
    case = record["subject_id"]
    model.eval()
    config["training"]["amp"] = args.amp
    with torch.no_grad():
        p = infer_volume(model, image.to(device), config, compute_uncertainty=False)
        probability = p.base_class_probability[0].float().cpu().numpy()
    with np.load(args.reference / "support" / f"{case}.npz", allow_pickle=False) as saved:
        observed = score(probability, labels.numpy().astype(np.uint8), valid, saved["foreground"])
    with (args.reference.parent / "ours/frame_metrics.csv").open(newline="") as stream:
        expected = next(row for row in csv.DictReader(stream) if row["case"] == case)
    if (observed["ECE"] != float(expected["ECE"])
            or observed["ece_support_sha256"] != expected["ece_support_sha256"]
            or observed["ece_target_sha256"] != expected["ece_target_sha256"]):
        raise RuntimeError("Original ACDC ECE or fixed support changed")
    del p, probability
    roi = config["data"]["roi_size"]
    starts = [(size - length) // 2 for size, length in zip(image.shape[2:], roi)]
    crop = tuple(slice(start, start + length) for start, length in zip(starts, roi))
    image = image[(slice(None), slice(None), *crop)].to(device)
    labels = labels[crop][None].to(device)
    for variant in VARIANTS[1:]:
        configure_heads(model, variant)
        model.train()
        changed = [False, False]
        handles = []
        for index, head in enumerate((model.head1, model.head2)):
            def check(_, inputs, output, index=index):
                changed[index] |= not torch.equal(inputs[0], output)
            handles.append(head.dropout.register_forward_hook(check))
        try:
            from ours.inference import autocast_context
            with autocast_context(device, args.amp):
                loss = sum(F.cross_entropy(logits.float(), labels) for logits in model.forward_logits(image))
            loss.backward()
            if not torch.isfinite(loss) or not all(changed):
                raise RuntimeError(f"Training smoke failed for {variant}")
            model.zero_grad(set_to_none=True)
            outputs = [p for head in (model.head1, model.head2) for p in head.out.parameters()]
            before = [p.detach().clone() for p in outputs]
            untouched = {name: p._version for name, p in model.named_parameters()
                         if not name.startswith(("head1.out.", "head2.out."))}
            changed[:] = [False, False]

            def inspect(model, *args, **kwargs):
                if not all(changed) or not any(not torch.equal(p, original)
                                              for p, original in zip(outputs, before)):
                    raise RuntimeError(f"TTA perturbation or output update missing for {variant}")
                if any(p._version != untouched[name] for name, p in model.named_parameters()
                       if name in untouched):
                    raise RuntimeError("TTA modified a frozen parameter")
                return infer_volume(model, *args, **kwargs)

            _, steps = predict_frame(model, image, config, True, inspect)
            if steps != 2 or any(not torch.equal(p, old) for p, old in zip(outputs, before)):
                raise RuntimeError(f"TTA reset failed for {variant}")
            first, _ = predict_frame(model, image, config, False, infer_volume)
            second, _ = predict_frame(model, image, config, False, infer_volume)
            torch.testing.assert_close(first.base_class_probability, second.base_class_probability,
                                       rtol=0, atol=0)
            print(f"Real ACDC smoke passed: {variant}", flush=True)
        finally:
            for handle in handles:
                handle.remove()


if __name__ == "__main__":
    main()
