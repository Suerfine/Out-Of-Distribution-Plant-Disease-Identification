"""Alternative training terms that replace the SupCon term in train/train_supcon.py (everything else in that trainer stays).

Each is opt-in through the loss config and acts on the backbone features f next to the cross-entropy heads, exactly where
the SupCon heads sit. Scoring stays on the cross-entropy heads, as in the SupCon runs, so the terms only shape the backbone.

  loss.taxonomy.mode "euclid"      linear heads that predict the crop's botanical family and the disease's pathogen group
  loss.taxonomy.mode "hyperbolic"  heads that embed the image in a Poincare ball and classify it, and its family / group,
                                   by hyperbolic distance to learned prototypes, with a pull of each class prototype
                                   towards its group prototype (the hierarchy)
  loss.smooth.mode  "structured"   label smoothing whose mass goes to related classes (diseases that occur on the same
                                   crops, crops that share diseases, taken from the fit images' pairs) instead of uniformly
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.cnn import HEAD_SIZES
from dpd.taxonomy import taxonomy_ids

HEADS = ("crop", "disease")


class TaxonomyEuclid(nn.Module):
    """Auxiliary linear heads on the backbone features (same dropout as the cross-entropy heads)."""

    def __init__(self, dim: int, groups: dict, dropout: float, weight: float):
        super().__init__()
        self.weight = weight
        self.register_buffer("crop_group", torch.as_tensor(groups["crop"][0]))
        self.register_buffer("disease_group", torch.as_tensor(groups["disease"][0]))
        self.heads = nn.ModuleDict({h: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, len(groups[h][1]))) for h in HEADS})

    def param_groups(self, lr_head: float):
        return [{"params": list(self.parameters()), "lr": lr_head}]

    def forward(self, f, labels):
        loss = 0.0
        for h in HEADS:
            g = (self.crop_group if h == "crop" else self.disease_group)[labels[h]]
            loss = loss + F.cross_entropy(self.heads[h](f), g)
        return self.weight * loss


def expmap0(v, eps=1e-6):
    """Exponential map at the origin of the unit Poincare ball (curvature 1)."""
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.tanh(n) * v / n


def poincare_dist(x, y):
    """(B, K) hyperbolic distances between the rows of x (B, d) and y (K, d), both inside the unit ball."""
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1).unsqueeze(0)
    sq = (x2 + y2 - 2 * x @ y.T).clamp_min(0.0)
    arg = 1 + 2 * sq / ((1 - x2).clamp_min(1e-5) * (1 - y2).clamp_min(1e-5))
    return torch.acosh(arg.clamp_min(1 + 1e-6))


class HyperbolicTerm(nn.Module):
    """Per head: MLP (same shape as the SupCon projection head) -> norm clipping -> expmap -> classify by -scale * distance to
    class prototypes (loss 1) and to group prototypes (loss 2); loss 3 pulls each class prototype towards its group's."""

    def __init__(self, dim: int, proj_dim: int, groups: dict, clip_r: float, scale: float, w_group: float, w_pull: float,
                 proto_lr_mult: float):
        super().__init__()
        self.clip_r, self.w_group, self.w_pull, self.proto_lr_mult = clip_r, w_group, w_pull, proto_lr_mult
        self.proj = nn.ModuleDict({h: nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj_dim)) for h in HEADS})
        self.cls_tan, self.grp_tan = nn.ParameterDict(), nn.ParameterDict()
        for h in HEADS:
            self.register_buffer(f"group_of_{h}", torch.as_tensor(groups[h][0]))
            for store, n in ((self.cls_tan, HEAD_SIZES[h]), (self.grp_tan, len(groups[h][1]))):
                v = torch.randn(n, proj_dim)
                store[h] = nn.Parameter(v / v.norm(dim=-1, keepdim=True))  # tangent norm 1 -> ball radius tanh(1)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(scale))))

    def param_groups(self, lr_head: float):
        protos = list(self.cls_tan.parameters()) + list(self.grp_tan.parameters())
        ids = {id(p) for p in protos}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": rest, "lr": lr_head}, {"params": protos, "lr": lr_head * self.proto_lr_mult, "weight_decay": 0.0}]

    def forward(self, f, labels):
        loss, s = 0.0, self.log_scale.exp()
        for h in HEADS:
            z = self.proj[h](f)
            z = z * (self.clip_r / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)  # clip the tangent vector
            x = expmap0(z)
            P, G = expmap0(self.cls_tan[h]), expmap0(self.grp_tan[h])
            group_of = getattr(self, f"group_of_{h}")
            y = labels[h]
            loss = loss + F.cross_entropy(-s * poincare_dist(x, P), y) + self.w_group * F.cross_entropy(-s * poincare_dist(x, G), group_of[y])
            D = poincare_dist(P, G)
            loss = loss + self.w_pull * D[torch.arange(len(group_of), device=D.device), group_of].mean()
        return loss


def build_extra(cfg_tax: dict, dim: int, proj_dim: int, dropout: float, pairs):
    groups = taxonomy_ids(pairs)
    if cfg_tax["mode"] == "euclid":
        return TaxonomyEuclid(dim, groups, dropout, cfg_tax.get("weight", 1.0))
    if cfg_tax["mode"] == "hyperbolic":
        return HyperbolicTerm(dim, proj_dim, groups, cfg_tax.get("clip_r", 2.0), cfg_tax.get("scale", 5.0), cfg_tax.get("w_group", 1.0),
                              cfg_tax.get("w_pull", 0.1), cfg_tax.get("proto_lr_mult", 10.0))
    raise ValueError(cfg_tax["mode"])


def relation_matrices(train_df, device) -> dict:
    """Row-normalised relation of each head's classes, from the fit images' (crop, disease) pairs only. Diseases are related
    through the crops they share, crops through the diseases they share; a class with no relation falls back to uniform."""
    M = np.zeros((HEAD_SIZES["crop"], HEAD_SIZES["disease"]), np.float32)
    u = train_df.drop_duplicates("pair_id")
    M[u["crop_id"].to_numpy(), u["disease_id"].to_numpy()] = 1.0
    out = {}
    for h, R in (("crop", M @ M.T), ("disease", M.T @ M)):
        R = R.copy()
        np.fill_diagonal(R, 0.0)
        n = R.shape[0]
        empty = R.sum(1) == 0
        R[empty] = 1.0 / max(n - 1, 1)
        R[empty, np.arange(n)[empty]] = 0.0
        out[h] = torch.as_tensor(R / R.sum(1, keepdims=True)).to(device)
    return out


def structured_smoothing_ce(logits, y, relation, eps):
    """Cross-entropy against (1 - eps) * one-hot + eps * (relation row of the true class)."""
    target = (1 - eps) * F.one_hot(y, logits.shape[1]).float() + eps * relation[y]
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
