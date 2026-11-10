"""Chapter 3.3 online predict-then-adapt for ACDC's four atomic classes."""
from contextlib import contextmanager
from hashlib import sha256
from itertools import product

import torch

from directional_zernike_project.ours.early_adapter import MultiScale3DAdapter


def windows(shape, roi, overlap):
    axes = []
    for size, length in zip(shape, roi):
        if size < length:
            raise ValueError("ACDC preprocessing must pad the image to the ROI")
        end = size - length
        starts = list(range(0, end + 1, max(1, int(length * (1 - overlap)))))
        if starts[-1] != end:
            starts.append(end)
        axes.append([slice(start, start + length) for start in starts])
    return list(product(*axes))


@contextmanager
def seeded(device, case, phase, draw, aligned=False):
    namespace = f"u_min_v1:2026:{case}:{phase}:{draw}" if aligned else f"2026:{case}:{phase}:{draw}"
    seed = int.from_bytes(sha256(namespace.encode()).digest()[:8], "big")
    if aligned:
        seed %= 2**63 - 1
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


class ContinualMeanU:
    def __init__(self, model, config, device, learning_rate, frozen_inference):
        self.model = model.eval().requires_grad_(False)
        self.config = config
        self.device = device
        self.frozen_inference = frozen_inference
        self.roi = tuple(config["data"]["roi_size"])
        self.overlap = config["training"]["infer_overlap"]
        self.aligned = config.get("zernike", {}).get("descriptor") == "directional_coefficients"
        with seeded(torch.device("cpu"), "adapters", "init", 0, self.aligned):
            self.adapters = torch.nn.ModuleList(MultiScale3DAdapter() for _ in range(3)).to(device)
        self.enabled = False
        model.swinViT.patch_embed.register_forward_hook(
            lambda module, inputs, value: self.adapters[0](value) if self.enabled else value)
        for index, head in enumerate((model.head1, model.head2), 1):
            head.out.register_forward_pre_hook(
                lambda module, inputs, index=index:
                (self.adapters[index](inputs[0]),) if self.enabled else inputs)
        self.optimizer = torch.optim.Adam(self.adapters.parameters(), lr=learning_rate,
                                          betas=(.9, .99), weight_decay=0.)

    def activate(self, enabled):
        self.enabled = enabled
        for head in (self.model.head1, self.model.head2):
            head.dropout.train(enabled)

    @torch.no_grad()
    def __call__(self, image, case):
        self.activate(False)
        output = self.frozen_inference(self.model, image, self.config, compute_uncertainty=True)
        self.activate(True)
        grid = windows(image.shape[2:], self.roi, self.overlap)
        counts = torch.zeros(image.shape[2:], device=self.device, dtype=torch.int32)
        probabilities = torch.zeros_like(output.base_class_probability, dtype=torch.float32)
        for crop in grid:
            patch = image[(slice(None), slice(None), *crop)]
            for draw in range(2):
                with seeded(self.device, case, "evaluation", draw, self.aligned):
                    heads = self.model.forward_logits(patch)
                probabilities[(slice(None), slice(None), *crop)] += sum(
                    h.float().softmax(1) for h in heads) * .25
            counts[crop] += 1
        output.base_class_probability = probabilities / counts
        return output, 0

    @torch.enable_grad()
    def update(self, batch):
        self.activate(True)
        self.optimizer.zero_grad(set_to_none=True)
        measured = 0.
        for case, image in batch:
            grid = windows(image.shape[2:], self.roi, self.overlap)
            denominator = len(batch) * len(grid) * self.roi[0] * self.roi[1] * self.roi[2]
            for crop in grid:
                patch = image[(slice(None), slice(None), *crop)].to(self.device)
                with seeded(self.device, case, "adaptation", 0, self.aligned):
                    p = [head.float().softmax(1) for head in self.model.forward_logits(patch)]
                leaves = [value.detach().requires_grad_() for value in p]
                depth = patch.shape[2]
                halo = max(self.model.zernike.windows) // 2
                for start in range(0, depth, 16):
                    stop = min(depth, start + 16)
                    lo, hi = max(0, start - halo), min(depth, stop + halo)
                    a, b = [leaf[:, :, lo:hi] for leaf in leaves]
                    disagreement = self.model.zernike.disagreement(a, b, self.model.zernike_stats)
                    value = self.model.fusion(disagreement)[:, :, start-lo:stop-lo].sum() / denominator
                    if not torch.isfinite(value):
                        raise ValueError("Nonfinite mean U")
                    measured += float(value.detach())
                    value.backward()
                torch.autograd.backward(p, [leaf.grad for leaf in leaves])
        norm = float(torch.nn.utils.clip_grad_norm_(self.adapters.parameters(), 5., error_if_nonfinite=True))
        updated = int(norm > 1e-12)
        if updated:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return measured, norm, updated
