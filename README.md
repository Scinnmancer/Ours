# Dual-head Swin UNETR with Zernike geometric reliability

This package implements Zernike geometric-disagreement risk calibration without
modifying the reference implementation in `baseline/`. Jensen-Shannon
probability disagreement is not computed or used by calibration, Z0 fitting, or
evaluation. A monotonic fusion maps raw disagreement into `[0, 1]` and is fitted
with voxel-wise error targets during calibration.

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

Calibration defaults to `training.calibration_trainable_scope=heads`: the shared
encoder stays frozen while both decoder heads and the active fusion parameters
`raw_xi` and `bias` are optimized. The two heads receive independently augmented
intensity views and keep their independent `Dropout3d` modules active. Risk is
computed as `u = sigmoid(bias + softplus(raw_xi) * zernike_disagreement)`, and the
calibration term is the mean `(u - error)^2` over the union of predicted and true
tumor voxels. The total loss is the sum of both heads' Dice losses plus this Brier
term with constant `lambda_u=1.0`; it is not ramped. The encoder hash is checked
before and after calibration. Set the scope to `full` only when the shared
segmentation network should also be fine-tuned.

The output mapping remains
`sigmoid(bias + softplus(raw_xi) * zernike_disagreement)` so downstream label
transfer receives a bounded risk score. `raw_eta` is retained only for strict
checkpoint compatibility and stays frozen because JS disagreement is disabled.
Checkpoints are selected by minimum source-validation Risk ECE subject to the
configured Dice tolerance; Risk Brier and segmentation ECE remain logged as
secondary observations.

Standalone evaluation strengthens refinement only at test time with
`evaluation.refine_strength_scale` (default `2.0`). With the default
`label_transfer.alpha_max=0.35`, the effective update ceiling is `0.70`. The
base value, multiplier, and effective value are saved in
`checkpoint_metadata.json`; calibration, Z0 fitting, and checkpoint parameters
are unchanged.

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
Zernike statistic health, Brier risk alignment, uncertainty calibration,
trainable fusion parameters, checkpoint selection,
frozen encoder hashes, plateau advisories, and terminal status.
No subject identifiers or voxel data are written to telemetry events.

Monitoring overhead is controlled by `monitoring.gradient_interval`; batch-level
events are disabled by default and can be enabled with
`monitoring.batch_interval=N`. Plateau detections are advisory and never stop
training automatically.

`basic_ece` is the ordinary top-label multiclass ECE of the four-class
segmentation probabilities; `ece` is retained as an identical compatibility
alias. `risk_ece` and `risk_brier` instead describe the bounded uncertainty
score `u` and remain diagnostic fields in standalone evaluation. Evaluation
metadata identifies `basic_ece` as its calibration metric, while training-time
calibration checkpoints continue to be selected by minimum source-validation
`risk_ece`, subject to the configured Dice tolerance.

Validation calibration statistics use a deterministic reservoir capped by
`evaluation.max_metric_voxels`, so their host-memory cost does not grow with the
number of cases. Standalone evaluation also drops unused label representations,
offloads the tensors needed by CPU metrics, and releases the CUDA allocator cache
after each case by default; set `evaluation.release_cuda_cache=false` only when
throughput is more important than a stable `nvidia-smi` footprint.

The `val` split is reported as `source_val`, not as an independent ID test. Only
the three held-out centers are treated as OOD tests.
