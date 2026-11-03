"""Disentangled multi-task CNN: one shared backbone, a crop branch and a disease branch, and adversarial "spies".

    image -> backbone -> f -> crop branch    -> z_crop -> crop head    -> P(crop)
                          -> disease branch -> z_dis  -> disease head -> P(disease)

Training adds two adversaries behind a gradient reversal layer (GRL, Ganin & Lempitsky 2015): one spy predicts the
crop from z_dis, the other the disease from z_crop. Each spy learns to guess; the reversed gradient makes the branch
remove what the spy uses, so z_dis keeps symptom information without crop cues and vice versa. An optional
decorrelation penalty pushes z_crop and z_dis apart directly. At test time only the two heads are used, and pair
scores are composed exactly as for the baselines (dpd.cnn.compose_pair_scores).

Used by train/train_disentangle.py (method "cnn_disentangle").
"""

import math

import torch
import torch.nn as nn

from dpd.cnn import HEAD_SIZES, build_backbone


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam: float):
    """Identity in the forward pass; multiplies the gradient by -lam in the backward pass."""
    return _GradReverse.apply(x, lam)


def adversarial_ramp(progress: float) -> float:
    """DANN schedule: 0 at the start of training, rising smoothly to 1, so the spies do not disturb early learning."""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def decorrelation_loss(z_a, z_b):
    """Mean squared cross-correlation between the dimensions of two feature batches (0 = uncorrelated)."""
    z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-5)
    z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-5)
    c = z_a.T @ z_b / len(z_a)
    return (c**2).mean()


def _branch(dim_in, dim_out, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out), nn.ReLU())


def _spy(dim_in, hidden, n_out):
    return nn.Sequential(nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class DisentangledNet(nn.Module):
    """forward(x) -> {"crop", "disease"} logits plus the branch features {"z_crop", "z_dis"}.

    spies(z_crop, z_dis, lam) -> {"spy_crop" (crop guessed from z_dis), "spy_disease" (disease guessed from z_crop)},
    both behind the GRL. Only the training loop calls spies(); prediction uses the heads alone.
    """

    def __init__(self, backbone: str, pretrained: bool, z_dim: int, spy_hidden: int, dropout: float, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained, drop_path_rate)
        self.crop_branch = _branch(dim, z_dim, dropout)
        self.disease_branch = _branch(dim, z_dim, dropout)
        self.crop_head = nn.Linear(z_dim, HEAD_SIZES["crop"])
        self.disease_head = nn.Linear(z_dim, HEAD_SIZES["disease"])
        self.spy_crop = _spy(z_dim, spy_hidden, HEAD_SIZES["crop"])
        self.spy_disease = _spy(z_dim, spy_hidden, HEAD_SIZES["disease"])

    def forward(self, x):
        f = self.backbone(x)
        z_crop, z_dis = self.crop_branch(f), self.disease_branch(f)
        return {"crop": self.crop_head(z_crop), "disease": self.disease_head(z_dis), "z_crop": z_crop, "z_dis": z_dis}

    def spies(self, z_crop, z_dis, lam: float):
        return {"spy_crop": self.spy_crop(grad_reverse(z_dis, lam)), "spy_disease": self.spy_disease(grad_reverse(z_crop, lam))}

    def head_parameters(self):
        """Everything except the backbone (branches, heads, spies): trained with the higher head learning rate."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]
