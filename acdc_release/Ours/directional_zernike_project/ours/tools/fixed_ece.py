"""Exact four-class ECE on the original frozen-baseline foreground support."""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DEFINITION = 'atomic_top_label_fixed_baseline_union_brain_v1'
SUPPORT = 'brain & (GT foreground | frozen baseline atomic foreground)'


def support_sha256(mask):
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    digest = hashlib.sha256(str(mask.shape).encode())
    digest.update(memoryview(mask).cast('B'))
    return digest.hexdigest()


def target_sha256(labels):
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    digest = hashlib.sha256(str(labels.shape).encode())
    digest.update(memoryview(labels).cast('B'))
    return digest.hexdigest()


class FixedECEAccumulator:
    """All selected voxels contribute; retain only 15 bin sufficient statistics."""
    def __init__(self):
        self.count = np.zeros(15, np.int64)
        self.confidence = np.zeros(15, np.float64)
        self.correct = np.zeros(15, np.int64)
        self.nll = self.brier = 0.
        self.high_error = 0

    def add(self, atomic, labels, support):
        atomic, labels, support = map(np.asarray, (atomic, labels, support))
        if atomic.shape != (4, *labels.shape) or support.shape != labels.shape or support.dtype != np.bool_:
            raise ValueError('Invalid atomic probability/label/fixed support shape or dtype')
        if (not np.isfinite(atomic).all() or (atomic < 0).any() or (atomic > 1).any()
                or not np.allclose(atomic.sum(0), 1., atol=1e-4)
                or not np.isin(labels, (0, 1, 2, 3)).all()):
            raise ValueError('Invalid four-class probabilities or labels')
        p, y = atomic[:, support], labels[support].astype(np.int64)
        if not y.size:
            return
        confidence, correct = p.max(0), p.argmax(0) == y
        # Match the original tta_eval_metrics.metrics bin assignment exactly.
        bins = np.minimum((confidence * 15).astype(np.int64), 14)
        self.count += np.bincount(bins, minlength=15)
        self.confidence += np.bincount(bins, weights=confidence, minlength=15)
        self.correct += np.bincount(bins, weights=correct, minlength=15).astype(np.int64)
        self.nll += float(-np.log(np.maximum(p[y, np.arange(y.size)], 1e-7)).sum(dtype=np.float64))
        hot = np.eye(4, dtype=np.float32)[y].T
        self.brier += float(((p-hot)**2).sum(0).sum(dtype=np.float64))
        self.high_error += int(((confidence >= .95) & ~correct).sum())

    def result(self):
        count = int(self.count.sum())
        ece = float(np.abs(self.confidence-self.correct).sum()/count) if count else None
        return {'ECE': ece, 'ece_definition': DEFINITION, 'ece_bins': 15,
                'ece_support': SUPPORT, 'ece_confidence': 'max of four atomic class probabilities',
                'ece_correctness': 'atomic argmax class equals GT class',
                'ece_voxels': count, 'ece_sampled_voxels': count, 'ece_sampling': 'none; all support voxels',
                'ece_empty_support_policy': 'unavailable; no whole-image fallback',
                'fixed_foreground_ECE': ece, 'fixed_foreground_voxels': count,
                'fixed_foreground_status': 'valid' if count else 'empty_support',
                'fixed_foreground_NLL': self.nll/count if count else None,
                'fixed_foreground_Brier': self.brier/count if count else None,
                'high_confidence_error_fraction': self.high_error/count if count else None}


def find_baseline_directory(signature, frozen_hash, explicit=None):
    """Discover only a baseline with matching frozen weights and inference plan."""
    candidates = [Path(explicit)] if explicit is not None else [
        Path('/autodl-fs/data/soft_u_runs/no_context_seed2026/predictions'),
        Path('/root/autodl-fs/soft_u_runs/no_context_seed2026/predictions'),
        Path(__file__).resolve().parents[2]/'run/no_context_seed2026/predictions']
    for candidate in candidates:
        if (candidate/'predictions').is_dir():
            candidate = candidate/'predictions'
        path = candidate/'protocol.json'
        if not path.is_file():
            if explicit is not None:
                raise ValueError('Explicit baseline protocol is missing')
            continue
        protocol = json.loads(path.read_text())
        sig = protocol.get('signature', {})
        matches = (protocol.get('schema') == 'no_context_frozen_base_v1'
                   and bool(frozen_hash)
                   and sig.get('frozen_base_sha256') == frozen_hash
                   and all(k in sig and k in signature for k in ('roi', 'overlap', 'seed', 'plan'))
                   and all(sig.get(k) == signature.get(k) for k in ('roi', 'overlap', 'seed', 'plan')))
        if matches:
            return candidate.resolve()
        if explicit is not None:
            raise ValueError('Baseline frozen weights, seed, ROI, overlap or patient plan differ')
    return None


def load_baseline_foreground(directory, case, shape, affine, brain):
    """Read small baseline artifacts; never replace them with adapted foreground."""
    path = Path(directory)/'cases'/case
    marker = json.loads((path/'complete.json').read_text())
    if marker.get('status') != 'complete' or marker.get('method') != 'no_context':
        raise ValueError('Baseline case is incomplete or is not frozen no-context inference')
    original = np.load(path/'original_foreground.npy', mmap_mode='r')
    baseline_brain = np.load(path/'brain.npy', mmap_mode='r')
    image = nib.load(str(path/'geometry_step3_atomic.nii.gz'))
    if (original.shape != shape or original.dtype != np.bool_
            or baseline_brain.shape != shape or baseline_brain.dtype != np.bool_
            or not np.array_equal(baseline_brain, brain)
            or image.shape != shape or not np.allclose(image.affine, affine)):
        raise ValueError('Fixed baseline mask/brain coordinates or dtype differ')
    classes = np.asarray(image.dataobj)
    if not np.isin(classes, (0, 1, 2, 4)).all() or not np.array_equal(original, classes > 0):
        raise ValueError('Original baseline foreground differs from frozen atomic prediction')
    return np.array(original, copy=True)


def score_saved_case(directory, baseline, case, labels, affine, seed):
    """CPU snapshot scoring, reusing only online scores with the identical mask."""
    directory = Path(directory)
    brain = np.load(directory/'brain.npy', mmap_mode='r')
    original = load_baseline_foreground(baseline, case, labels.shape, affine, brain)
    fixed = brain & ((labels > 0) | original)
    digest = support_sha256(fixed)
    path = directory/'prediction_probabilities.npy'
    if path.is_file():
        p = np.load(path, mmap_mode='r')
        if p.shape != (7, *labels.shape) or p.dtype != np.float32:
            raise ValueError('Invalid saved probabilities')
        image = nib.load(str(directory/'prediction_atomic.nii.gz'))
        if image.shape != labels.shape or not np.allclose(image.affine, affine):
            raise ValueError('Prediction and fixed support coordinates differ')
        accumulator = FixedECEAccumulator()
        for start in range(0, labels.shape[0], 16):
            atomic = p[3:, start:start+16]
            classes = atomic.argmax(0).astype(np.uint8); classes[classes == 3] = 4
            if not np.array_equal(classes, np.asarray(image.dataobj[start:start+16])):
                raise ValueError('Probability and saved atomic segmentation differ')
            accumulator.add(atomic, labels[start:start+16], fixed[start:start+16])
        return {**accumulator.result(), 'ece_support_sha256': digest, 'ece_target_sha256': target_sha256(labels)}
    path = directory/'online_metrics.json'
    if not path.is_file():
        raise ValueError(f'{case}: probabilities unavailable; run ours/run_recheck_fixed_ece.sh')
    saved = json.loads(path.read_text())
    count = int(fixed.sum())
    if (saved.get('case') != case or saved.get('seed') != seed
            or saved.get('ece_definition') != DEFINITION or saved.get('ece_bins') != 15
            or saved.get('ece_sampling') != 'none; all support voxels'
            or saved.get('ece_support_sha256') != digest
            or saved.get('ece_target_sha256') != target_sha256(labels)
            or saved.get('ece_voxels') != count or saved.get('ece_sampled_voxels') != count
            or saved.get('fixed_foreground_voxels') != count):
        raise ValueError(f'{case}: old/incompatible online ECE; run ours/run_recheck_fixed_ece.sh')
    expected_status = 'valid' if count else 'empty_support'
    if saved.get('fixed_foreground_status') != expected_status or saved.get('ECE') != saved.get('fixed_foreground_ECE'):
        raise ValueError('Invalid cached fixed ECE status or alias')
    for key in ('ECE', 'fixed_foreground_NLL', 'fixed_foreground_Brier', 'high_confidence_error_fraction'):
        value = saved.get(key)
        if (count and (value is None or not np.isfinite(value) or value < 0)) or (not count and value is not None):
            raise ValueError('Invalid cached fixed calibration metric')
    return {k: v for k, v in saved.items() if k == 'ECE' or k.startswith(('ece_', 'fixed_foreground_'))
            or k == 'high_confidence_error_fraction'}
