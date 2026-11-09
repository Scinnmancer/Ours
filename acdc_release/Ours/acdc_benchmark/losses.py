"""Three ACDC baselines on the shared single-head Swin."""
import torch
from torch.nn import functional as F

from nacl.losses import NACLLoss


def edl_parameters(logits):
    alpha = F.relu(logits.float()) + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    return alpha, strength


def edl_probability(logits):
    alpha, strength = edl_parameters(logits)
    return alpha / strength


def standard_edl(logits, target, kl_weight):
    alpha, strength = edl_parameters(logits)
    target = F.one_hot(target.long(), num_classes=4).movedim(-1, 1).float()
    probability = alpha / strength
    mse = (target - probability).square() + probability * (1.0 - probability) / (strength + 1.0)

    # Remove evidence for the correct class from the KL regularizer.
    alpha = target + (1.0 - target) * alpha
    strength = alpha.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(strength) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
          - torch.lgamma(alpha.new_tensor(4.0))
          + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength)))
          .sum(dim=1, keepdim=True))
    return mse.sum(dim=1).mean() + kl_weight * kl.mean()


def margin_label_smoothing(logits, target, margin, weight):
    distance = logits.amax(dim=1, keepdim=True) - logits
    return F.cross_entropy(logits, target) + weight * F.relu(distance - margin).mean()


class TrainingAdapter:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        if args.method == "nacl":
            self.loss = NACLLoss(classes=4, spatial_dims=3, kernel_size=3, alpha=args.nacl_weight)

    def trainable_parameters(self):
        return self.model.parameters()

    def compute_loss(self, image, label, batch, epoch, batch_index):
        logits = self.model(image).float()
        target = label[:, 0]
        if self.args.method == "nacl":
            loss = self.loss(logits, target)
        elif self.args.method == "mbls":
            loss = margin_label_smoothing(logits, target, self.args.margin, self.args.margin_weight)
        else:
            loss = standard_edl(logits, target, min(1.0, epoch / self.args.edl_annealing_epochs))
        return loss, {}

    def after_backward(self, epoch):
        pass
