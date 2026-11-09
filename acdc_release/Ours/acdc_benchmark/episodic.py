"""Unlabeled, frame-local adaptation of the two output convolutions."""
import torch
import torch.nn.functional as F


def predict_frame(model, image, config, adapt, inference):
    model.eval()
    heads = (model.head1, model.head2)
    parameters = [p for head in heads for p in head.out.parameters()]
    saved = [p.detach().clone() for p in parameters] if adapt else []
    flags = [(p, p.requires_grad) for p in model.parameters()] if adapt else []
    steps = 0
    try:
        if adapt:
            roi = config["data"]["roi_size"]
            starts = [(size - length) // 2 for size, length in zip(image.shape[2:], roi)]
            patch = image[(slice(None), slice(None), *(
                slice(start, start + length) for start, length in zip(starts, roi)))]
            with torch.no_grad():
                features = model.encode(patch)
                teacher = torch.stack([head(features).float().softmax(1) for head in heads]).mean(0)
                confidence, predicted = teacher.max(1)
                selected = (predicted > 0) & (confidence >= 0.90)
            if selected.any():
                model.requires_grad_(False)
                for p in parameters:
                    p.requires_grad_(True)
                optimizer = torch.optim.Adam(parameters, lr=1e-4)
                for head in heads:
                    head.dropout.train()
                target = teacher.movedim(1, -1)[selected]
                with torch.enable_grad():
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        loss = sum(
                            -(target * F.log_softmax(head(features).float(), dim=1)
                              .movedim(1, -1)[selected]).sum(1).mean()
                            for head in heads
                        ) / len(heads)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite ACDC TTA loss")
                        loss.backward()
                        optimizer.step()
                        steps += 1
        model.eval()
        with torch.no_grad():
            return inference(model, image, config, compute_uncertainty=True), steps
    finally:
        with torch.no_grad():
            for p, weight in zip(parameters, saved):
                p.copy_(weight)
                p.grad = None
            for p, grad in flags:
                p.requires_grad_(grad)
        model.eval()
