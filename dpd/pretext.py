"""Pretext task modules for multi-task CNNs.

Implements:
1. SimSiam (Chen & He, CVPR 2021): Projector + Predictor with stop-gradient negative cosine similarity.
2. Barlow Twins (Zbontar et al., ICML 2021): Projector with cross-correlation redundancy reduction loss.
3. PretextMultiTaskNet: Shared backbone with classification heads (crop, disease) and pretext head(s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.cnn import HEAD_SIZES, build_backbone


class SimSiamHead(nn.Module):
    """SimSiam projection and prediction heads (Chen & He, CVPR 2021).

    Structure:
      Backbone features (dim) -> Projector MLP -> z (proj_dim) -> Predictor MLP -> p (proj_dim)
    """

    def __init__(self, in_dim: int, proj_dim: int = 512, hidden_dim: int = 2048, pred_hidden: int = 512):
        super().__init__()
        # 3-layer projector MLP with BatchNorm and ReLU
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),  # output BN has no learnable affine parameters (per paper)
        )
        # 2-layer predictor bottleneck MLP
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, p): projected representation and prediction vector."""
        z = self.projector(f)
        p = self.predictor(z)
        return z, p


def simsiam_loss(p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Symmetrized negative cosine similarity with stop-gradient on target projection."""
    def _d(p, z):
        z_stop = z.detach()
        p_norm = F.normalize(p.float(), dim=-1)
        z_norm = F.normalize(z_stop.float(), dim=-1)
        return -(p_norm * z_norm).sum(dim=-1).mean()

    return 0.5 * _d(p1, z2) + 0.5 * _d(p2, z1)


class BarlowTwinsHead(nn.Module):
    """Barlow Twins projector MLP (Zbontar et al., ICML 2021)."""

    def __init__(self, in_dim: int, proj_dim: int = 2048, hidden_dim: int = 2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.projector(f)


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_offdiag: float = 0.005) -> torch.Tensor:
    """Barlow Twins cross-correlation loss.

    Invariance term pulls diagonal of cross-correlation to 1.
    Redundancy reduction pushes off-diagonal elements to 0.
    """
    n, d = z1.shape
    # Normalize along the batch dimension
    z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-4)
    z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-4)

    # Cross-correlation matrix C: (D, D)
    c = (z1_norm.T @ z2_norm) / n

    # Invariance on diagonal, redundancy reduction on off-diagonal
    diag = torch.diagonal(c)
    on_diag = ((1.0 - diag) ** 2).sum()
    off_diag = (c ** 2).sum() - (diag ** 2).sum()

    return on_diag + lambda_offdiag * off_diag


class PretextMultiTaskNet(nn.Module):
    """Shared backbone with classification heads (crop, disease) and pretext head."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        heads: dict[str, int] | None = None,
        dropout: float = 0.3,
        pretext_type: str = "simsiam",
        pretext_proj_dim: int = 512,
        pretext_hidden_dim: int = 2048,
        pretext_pred_hidden: int = 512,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone, self.dim = build_backbone(backbone, pretrained, drop_path_rate)
        heads = heads or HEAD_SIZES
        self.heads = nn.ModuleDict({
            k: nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, n))
            for k, n in heads.items()
        })
        self.pretext_type = pretext_type.lower()
        if self.pretext_type == "simsiam":
            self.pretext_head = SimSiamHead(
                self.dim, proj_dim=pretext_proj_dim, hidden_dim=pretext_hidden_dim, pred_hidden=pretext_pred_hidden
            )
        elif self.pretext_type == "barlow_twins":
            self.pretext_head = BarlowTwinsHead(
                self.dim, proj_dim=pretext_proj_dim, hidden_dim=pretext_hidden_dim
            )
        else:
            raise ValueError(f"Unknown pretext_type: {pretext_type!r}")

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_heads(self, f: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: h(f) for k, h in self.heads.items()}

    def forward_pretext(self, f: torch.Tensor):
        return self.pretext_head(f)

    # Aliases for backwards compatibility
    forward_ssl = forward_pretext

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Standard classification forward pass."""
        f = self.backbone(x)
        return self.forward_heads(f)


# Alias
SSLMultiTaskNet = PretextMultiTaskNet
