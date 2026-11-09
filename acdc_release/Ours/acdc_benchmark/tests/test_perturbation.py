import pytest
import torch
from torch import nn

from acdc_benchmark.episodic import predict_frame
from acdc_benchmark.perturbation import FeaturePerturbation


@pytest.mark.parametrize("variant,strength", [
    ("feature_noise", .1), ("cct_feature_dropout", 0), ("feature_cutout", .05)
])
def test_perturbation_training_eval_and_independence(variant, strength):
    torch.manual_seed(42)
    x = torch.linspace(0, 1, 4096).reshape(1, 1, 16, 16, 16).expand(3, 4, -1, -1, -1)
    op = FeaturePerturbation(variant, strength)
    first, second = op(x), op(x)
    assert first.shape == x.shape and first.dtype == x.dtype
    assert not torch.equal(first, x)
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    assert op.eval()(x) is x


def test_cct_guards_and_zero_noise():
    x = torch.ones(2, 4, 4, 4, 4)
    assert FeaturePerturbation("feature_noise", 0)(x) is x
    cct = FeaturePerturbation("cct_feature_dropout", 0)
    assert torch.equal(cct(x), x)
    assert torch.equal(cct(-x), -x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = FeaturePerturbation("feature_noise", .2)
        self.out = nn.Conv3d(1, 4, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.copy_(torch.tensor([0., 5., 0., 0.]))

    def forward(self, x):
        return self.out(self.dropout(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(1, 1, 1)
        self.head1, self.head2 = Head(), Head()

    def encode(self, image):
        return self.encoder(image)


def test_tta_updates_only_outputs_and_restores_each_frame():
    torch.manual_seed(42)
    model = Model()
    image = torch.ones(1, 1, 4, 4, 4)
    config = {"data": {"roi_size": [4, 4, 4]}}
    # A small difference between teachers makes the soft target update both heads.
    with torch.no_grad():
        model.head2.out.bias[1] -= .5
    initial = {key: value.clone() for key, value in model.state_dict().items()}

    def inference(model, image, config, compute_uncertainty):
        assert compute_uncertainty and not model.head1.dropout.training
        assert torch.equal(model.encoder.weight, initial["encoder.weight"])
        assert not torch.equal(model.head1.out.bias, initial["head1.out.bias"])
        return model.head1(model.encode(image))

    with torch.no_grad():
        _, steps = predict_frame(model, image, config, True, inference)
    assert steps == 2
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())

    with torch.no_grad():
        model.head1.out.bias.copy_(torch.tensor([5., 0., 0., 0.]))
        model.head2.out.bias.copy_(model.head1.out.bias)
    _, steps = predict_frame(model, image, config, True, lambda *a, **kw: None)
    assert steps == 0
