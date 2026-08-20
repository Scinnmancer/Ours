# Dual-head Swin UNETR with Zernike geometric reliability

This package implements a Zernike geometric-disagreement risk calibration path
without modifying the reference implementation in `baseline/`. Jensen-Shannon
probability disagreement is not computed or used by calibration, Z0 fitting, or
evaluation. Legacy fusion parameters remain in the state layout so existing
same-architecture warm-up checkpoints still load strictly.

## Probability convention

Each decoder follows the baseline convention and emits three independent logits
in `[TC, WT, ET]` order. Training applies the same sigmoid Dice loss as the
baseline directly to each head's logits; no BCE term is used. Independent
sigmoid probabilities are used for thresholding, Dice aggregation, and base
segmentation output. The uncertainty and label-transfer modules still require a mutually
exclusive distribution, so a nested copy is formed internally by propagating
inner-region evidence outwards before conversion to `[background, NCR/NET,
edema, enhancing tumor]`. This internal closure does not alter the external base
region probabilities.

Shared Swin, attention, and stochastic-depth dropout rates are zero. Each
decoder has its own output `Dropout3d` module, configured by
`model.head_dropout_rates` (default `[0.1, 0.1]`), so the two heads sample
independent masks during training without injecting dropout into the shared
encoder.

Training uses the configured baseline encoder checkpoint by default
(`model.encoder_warm_start=true`); both decoder branches remain independently
initialized.

## Data split

The checked-in split manifest is `ours/data/splits/brats2020_reliability.json`.
With seed 2026, TCIA is the source domain (134 training and 33 validation
cases). CBICA (129), 2013 (30), and TMC (9) are kept as three disjoint OOD test
domains. Source training data is used to fit Zernike descriptor statistics;
source validation is used for checkpoint selection and Z0 fitting and is not
reported as an independent test set.

## Environment

Use Python 3.10 or newer. The target CUDA runtime determines the appropriate
PyTorch wheel, so PyTorch is intentionally not pinned in `requirements.txt`.
Install a CUDA-compatible PyTorch build for the target server, then install the
remaining dependencies:

```bash
python -m pip install -r ours/requirements.txt
```

The default data root is `/root/autodl-tmp/archive`. Override it without editing
the checked-in configuration:

```bash
export BRATS_DATA_ROOT=/path/to/archive
```

## Prepare, train, and evaluate

Run from the repository root:

```bash
python -m ours.prepare --config ours/configs/brats2020.yaml
python -m ours.train --config ours/configs/brats2020.yaml --stage all
python -m ours.test \
  --config ours/configs/brats2020.yaml \
  --checkpoint ours/runs/dual_swin_zernike_brats2020/final.pt
```

Any YAML value can be overridden with repeated `--set key=value`, for example:

```bash
python -m ours.train --config ours/configs/brats2020.yaml --stage all \
  --set data.workers=4 --set training.warmup_epochs=2 --set data.debug_cases=2
```

The stages may also be run separately. `warmup` writes `best_seg.pt`, `stats`
fits the source-domain Zernike descriptor statistics and writes
`stats_fitted.pt`, and `calibration` writes `best_calibrated.pt` plus the final
checkpoint with validation-fitted `Z0`. An existing same-architecture warm-up
checkpoint can start calibration directly; missing Zernike statistics are fitted
automatically before the first calibration epoch:

```bash
python -m ours.train --config ours/configs/brats2020.yaml --stage calibration \
  --checkpoint ours/runs/dual_swin_zernike_brats2020/best_seg.pt
```

Calibration freezes the shared encoder and both decoders by default, keeps the
model in evaluation mode, feeds the same normalized patch to both heads, and
optimizes only the geometric-risk `xi` and `bias`. Set
`training.freeze_segmentation_during_calibration=false` to recover joint
fine-tuning. The active risk formula is
`sigmoid(bias + softplus(raw_xi) * zernike_disagreement)`; `raw_eta` is retained
only for checkpoint compatibility.

If the configured split JSON is absent, the training entry point generates the
deterministic center-based splits before constructing data loaders. Run
`ours.prepare` explicitly when full path, affine, spacing, and label validation
is desired before training.

## Outputs

Training stores the resolved configuration, environment metadata, baseline
encoder loading report, checkpoints, fitted Zernike statistics,
segmentation-weight hashes, and validation metrics under
`ours/runs/<experiment>/`. Evaluation writes case-level and domain-level
CSV/JSON for both the base and refined predictions and can optionally save NIfTI
segmentations by setting `evaluation.save_nifti=true`.

Warm-up runs for 150 epochs by default. Calibration validation saves voxelwise
uncertainty snapshots every 10 epochs under
`uncertainty_maps/calibration/epoch_XXXX/`. Each validation case
is stored as a compressed NPZ containing the fused uncertainty map, voxel error
mask, base and target atomic labels, affine, epoch, and—when
`monitoring.uncertainty_map_components=true`—the Zernike-disagreement map.

Every training launch also creates an immutable telemetry directory at
`ours/runs/<experiment>/telemetry/<run_id>/`. Its manifest records the resolved
configuration, command, source/split/checkpoint fingerprints, split overlap
counts, library/CUDA/GPU metadata, and determinism state. `events.jsonl` records
per-epoch timing, learning rate, memory, parameter/gradient health, per-head and
ensemble region Dice/volume fractions, head disagreement, AMP skipped steps,
Zernike statistic health, uncertainty calibration/discrimination, active fusion parameters,
checkpoint selection, frozen segmentation hashes, plateau advisories, and
terminal status.
No subject identifiers or voxel data are written to telemetry events.

Monitoring overhead is controlled by `monitoring.gradient_interval`; batch-level
events are disabled by default and can be enabled with
`monitoring.batch_interval=N`. Plateau detections are advisory and never stop
training automatically.

`ece` and `brier` evaluate the four-class segmentation probabilities.
`risk_ece` and `risk_brier` evaluate the learned error probability `u`; error
AUROC/AUPR likewise use `u` against errors in the base prediction. Calibration
checkpoints are selected by minimum source-validation `risk_ece`, subject to the
configured Dice tolerance.

Validation calibration statistics use a deterministic reservoir capped by
`evaluation.max_metric_voxels`, so their host-memory cost does not grow with the
number of cases. Standalone evaluation also drops unused label representations,
offloads the tensors needed by CPU metrics, and releases the CUDA allocator cache
after each case by default; set `evaluation.release_cuda_cache=false` only when
throughput is more important than a stable `nvidia-smi` footprint.

The `val` split is reported as `source_val`, not as an independent ID test. Only
the three held-out centers are treated as OOD tests.
