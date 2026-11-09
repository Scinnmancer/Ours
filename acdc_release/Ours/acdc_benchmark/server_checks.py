"""Validate server inputs before any training or target evaluation."""
import argparse
import gzip
import json
from pathlib import Path

from .evaluate import acdc_config, load_predictor, split_sha256


def check_data(args):
    import nibabel as nib

    split_sha256(args.split_json)
    manifest = json.loads(args.split_json.read_text(encoding="utf-8"))
    data_root = args.data_root.resolve()
    for split, patients in (("train", 54), ("val", 13), ("test", 33)):
        records = manifest[split]
        if len(records) != patients * 2:
            raise ValueError(f"Unexpected frame count in {split}")
        for record in records:
            for key in ("image", "label"):
                path = (data_root / record[key]).resolve()
                if not path.is_relative_to(data_root):
                    raise ValueError(f"Input resolves outside the ACDC data directory: {path}")
                # Read to EOF to verify gzip size and CRC, without evaluating target labels.
                try:
                    with gzip.open(path, "rb") as volume:
                        while volume.read(1024 * 1024):
                            pass
                    nib.load(path)
                except (EOFError, OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
                    raise ValueError(f"Invalid ACDC volume: {path}") from exc
        print(f"Verified {split}: {patients} patients, {len(records)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acdc-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--method", choices=("baseline", "ml1_ace", "devis", "fgrm", "ours"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if bool(args.method) != bool(args.checkpoint):
        parser.error("--method and --checkpoint must be provided together")
    if args.checkpoint is None:
        check_data(args)
    else:
        import torch

        config = acdc_config(args)
        args.amp = False
        load_predictor(args, config, torch.device("cpu"))
        print(f"Validated {args.method} checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
