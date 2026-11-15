"""Primitive visual branches (v5.11+, model.primitive_branches), a light version of Troika (Huang et al., CVPR 2024).

Plain CSP-style scoring compares one image embedding (the CLS) with the pair, crop and disease prompts alike, so the
crop and disease losses pull the same vector in two directions. Troika gives each primitive its own visual
features. Here each of crop and disease gets a residual linear map on the CLS embedding,
    v_crop    = normalise(v + W_crop v)
    v_disease = normalise(v + W_disease v)
with W initialised to zero (so training starts exactly at plain scoring). The crop loss uses v_crop, the disease
loss uses v_disease and the pair loss keeps the plain v. At prediction time the branch scores are added to each
pair's score through primitive fusion (dpd/fusion.py), with weights chosen on validation (evaluation.posthoc).
Troika's cross-modal traction module (prompts shifted toward the image) is not included.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BRANCHES = ("crop", "disease")


def build_branches(dim: int) -> nn.ModuleDict:
    branches = nn.ModuleDict({b: nn.Linear(dim, dim, bias=False) for b in BRANCHES})
    for m in branches.values():
        nn.init.zeros_(m.weight)
    return branches


def branch_features(branches, name: str, v):
    """The branch embedding of L2-normalised image embeddings v (torch tensor or numpy array; same type back).
    branches None = plain v."""
    if branches is None:
        return v
    if isinstance(v, np.ndarray):
        w = branches[name].weight
        with torch.no_grad():
            t = torch.as_tensor(v, device=w.device, dtype=w.dtype)
            return F.normalize(t + branches[name](t), dim=-1).cpu().numpy()
    return F.normalize(v + branches[name](v.to(branches[name].weight.dtype)), dim=-1)
