from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .checkpoint import safe_torch_load
from .probability import (
    atomic_to_regions,
    independent_logits_to_atomic,
    independent_logits_to_regions,
)
from .transfer import UncertaintyGatedLabelTransfer
from .uncertainty import UncertaintyFusion, uncertainty_components
from .zernike import MultiScaleZernike, ZernikeStatistics


@dataclass
class DualHeadOutput:
    head_logits: tuple[torch.Tensor, torch.Tensor]
    head_region_probabilities: tuple[torch.Tensor, torch.Tensor]
    head_atomic_probabilities: tuple[torch.Tensor, torch.Tensor]
    base_atomic_probability: torch.Tensor
    probability_disagreement: torch.Tensor | None = None
    zernike_disagreement: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None
    refined_atomic_probability: torch.Tensor | None = None

    @property
    def base_region_probability(self) -> torch.Tensor:
        return 0.5 * (self.head_region_probabilities[0] + self.head_region_probabilities[1])

    @property
    def refined_region_probability(self) -> torch.Tensor | None:
        if self.refined_atomic_probability is None:
            return None
        return atomic_to_regions(self.refined_atomic_probability)


class DecoderBranch(nn.Module):
    def __init__(self, template: nn.Module, dropout_rate: float):
        super().__init__()
        self.decoder5 = template.decoder5
        self.decoder4 = template.decoder4
        self.decoder3 = template.decoder3
        self.decoder2 = template.decoder2
        self.decoder1 = template.decoder1
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.out = template.out

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        enc0, enc1, enc2, enc3, hidden3, bottleneck = features
        dec3 = self.decoder5(bottleneck, hidden3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        decoded = self.decoder1(dec0, enc0)
        return self.out(self.dropout(decoded))


def _make_swin_unetr(config: dict[str, Any]):
    from monai.networks.nets import SwinUNETR

    signature = inspect.signature(SwinUNETR)
    model_cfg = config["model"]
    kwargs: dict[str, Any] = {
        "in_channels": int(model_cfg.get("in_channels", 4)),
        "out_channels": int(model_cfg.get("region_channels", 3)),
        "feature_size": int(model_cfg.get("feature_size", 48)),
        "use_checkpoint": bool(model_cfg.get("use_checkpoint", True)),
    }
    if "img_size" in signature.parameters:
        kwargs["img_size"] = tuple(int(v) for v in config["data"]["roi_size"])
    optional = {
        "drop_rate": float(model_cfg.get("dropout_rate", 0.0)),
        "attn_drop_rate": float(model_cfg.get("attention_dropout_rate", 0.0)),
        "dropout_path_rate": float(model_cfg.get("dropout_path_rate", 0.0)),
        "spatial_dims": 3,
    }
    for key, value in optional.items():
        if key in signature.parameters:
            kwargs[key] = value
    return SwinUNETR(**kwargs)


class DualHeadSwinUNETR(nn.Module):
    ENCODER_PREFIXES = ("swinViT.", "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder10.")

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        output_mode = str(config["model"].get("output_mode", "independent_sigmoid"))
        if output_mode != "independent_sigmoid":
            raise ValueError("model.output_mode must be 'independent_sigmoid'")
        template1 = _make_swin_unetr(config)
        template2 = _make_swin_unetr(config)
        self.swinViT = template1.swinViT
        self.encoder1 = template1.encoder1
        self.encoder2 = template1.encoder2
        self.encoder3 = template1.encoder3
        self.encoder4 = template1.encoder4
        self.encoder10 = template1.encoder10
        self.normalize = getattr(template1, "normalize", True)
        head_dropout_rates = config["model"].get("head_dropout_rates", [0.2, 0.3])
        if not isinstance(head_dropout_rates, (list, tuple)) or len(head_dropout_rates) != 2:
            raise ValueError("model.head_dropout_rates must contain exactly two values")
        self.head1 = DecoderBranch(template1, float(head_dropout_rates[0]))
        self.head2 = DecoderBranch(template2, float(head_dropout_rates[1]))
        z_cfg = config["zernike"]
        orders = [tuple(pair) for pair in z_cfg["orders"]]
        self.zernike = MultiScaleZernike(z_cfg["windows"], orders, chunk_depth=int(z_cfg.get("chunk_depth", 0)))
        self.zernike_stats = ZernikeStatistics(len(z_cfg["windows"]), 4, len(orders))
        u_cfg = config["uncertainty"]
        self.fusion = UncertaintyFusion(
            eta=float(u_cfg.get("eta_init", 1.0)),
            xi=float(u_cfg.get("xi_init", 1.0)),
            bias=float(u_cfg.get("bias_init", -2.0)),
        )
        t_cfg = config["label_transfer"]
        self.label_transfer = UncertaintyGatedLabelTransfer(
            radius=int(t_cfg.get("radius", 2)),
            sigma=float(t_cfg.get("sigma", 1.0)),
            gamma=float(t_cfg.get("gamma", 2.0)),
            alpha_max=float(t_cfg.get("alpha_max", 0.35)),
            beta=float(t_cfg.get("beta", 2.0)),
            iterations=int(t_cfg.get("iterations", 3)),
            consensus_margin=float(t_cfg.get("consensus_margin", 0.25)),
            class_change_margin=float(t_cfg.get("class_change_margin", 0.30)),
        )

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.swinViT(image, self.normalize)
        enc0 = self.encoder1(image)
        enc1 = self.encoder2(hidden[0])
        enc2 = self.encoder3(hidden[1])
        enc3 = self.encoder4(hidden[2])
        bottleneck = self.encoder10(hidden[4])
        return enc0, enc1, enc2, enc3, hidden[3], bottleneck

    def forward_logits(self, image: torch.Tensor, second_view: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        features1 = self.encode(image)
        features2 = features1 if second_view is None else self.encode(second_view)
        return self.head1(features1), self.head2(features2)

    def stacked_logits(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.forward_logits(image), dim=1)

    def output_from_logits(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        compute_uncertainty: bool = True,
        refine: bool = False,
    ) -> DualHeadOutput:
        if refine and not compute_uncertainty:
            raise ValueError("refine=True requires compute_uncertainty=True")
        regions1 = independent_logits_to_regions(logits1)
        regions2 = independent_logits_to_regions(logits2)
        atomic1 = independent_logits_to_atomic(logits1)
        atomic2 = independent_logits_to_atomic(logits2)
        base = 0.5 * (atomic1 + atomic2)
        result = DualHeadOutput(
            head_logits=(logits1, logits2),
            head_region_probabilities=(regions1, regions2),
            head_atomic_probabilities=(atomic1, atomic2),
            base_atomic_probability=base,
        )
        if compute_uncertainty:
            z_disagreement = self.zernike.disagreement(atomic1, atomic2, self.zernike_stats)
            z_disagreement, uncertainty = uncertainty_components(
                z_disagreement,
                self.fusion,
            )
            result.probability_disagreement = None
            result.zernike_disagreement = z_disagreement
            result.uncertainty = uncertainty
            if refine:
                result.refined_atomic_probability = self.label_transfer(base, uncertainty)
        return result

    def forward(
        self,
        image: torch.Tensor,
        second_view: torch.Tensor | None = None,
        compute_uncertainty: bool = False,
        refine: bool = False,
    ) -> DualHeadOutput:
        logits1, logits2 = self.forward_logits(image, second_view)
        return self.output_from_logits(
            logits1,
            logits2,
            compute_uncertainty=compute_uncertainty,
            refine=refine,
        )

    def load_baseline_encoder(self, checkpoint_path: str) -> dict[str, Any]:
        checkpoint = safe_torch_load(checkpoint_path)
        source = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        source = {key.removeprefix("module."): value for key, value in source.items()}
        target = self.state_dict()
        encoder_targets = {key for key in target if key.startswith(self.ENCODER_PREFIXES)}
        compatible = {
            key: value
            for key, value in source.items()
            if key in encoder_targets and key in target and target[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError(f"No compatible encoder tensors found in {checkpoint_path}")
        self.load_state_dict(compatible, strict=False)
        missing = sorted(encoder_targets - compatible.keys())
        report = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "loaded_tensors": len(compatible),
            "expected_encoder_tensors": len(encoder_targets),
            "coverage": len(compatible) / max(len(encoder_targets), 1),
            "missing_encoder_keys": missing,
        }
        if report["coverage"] < 0.8:
            raise RuntimeError(f"Baseline encoder coverage is too low: {report['coverage']:.1%}")
        return report
