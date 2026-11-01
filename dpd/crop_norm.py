"""Crop-normalised disease prototypes ("disease normalisation"): a disease looks different on each crop, like one shirt
worn by different people, so the disease network removes the crop's effect before classifying the disease.

    image -> backbone -> f -> Linear -> z
    normalise for crop c:  n_c = T_c^-1(z)          ("take the shirt off the wearer")
    disease logits:        s * cos(n_c, p_d)        p_d = one base prototype per disease, shared by every crop
    reverse:               T_c(p_d) ~ z             ("put the shirt back on": the reverse check)

T_c is one of (crop_norm.transform):
    "none":     identity (control: a cosine-prototype disease classifier, otherwise the independent baseline)
    "film":     n = (z - shift_c) * exp(-log_gain_c), one learned gain and shift vector per crop, initialised to the
                identity and held near it by an L2 penalty (crops with little data or one disease stay unchanged)
    "centroid": n = z - global mean - shrink_c * (crop mean_c - global mean), running means of z; shrink_c =
                (D_c - 1) / D_c for a crop with D_c training diseases, because with one disease the crop mean is that
                disease's mean and removing it would remove the disease too

Pair score (dpd.cnn.compose_pair_scores's scale): 2 * (w * log P(crop c) + (1 - w) * log P(disease d | n_c)), where the
disease softmax runs over all 175 diseases under each candidate crop's normalisation. An unseen pair (c, d) is scored
with T_c learned from c's other diseases and p_d learned from d's other crops; this is compositional zero-shot
learning with attributes as operators (Nagarajan & Grauman 2018; coupling/decoupling with a symmetry check: SymNet,
Li et al. 2020).

Used by train/train_crop_norm.py (method "cnn_crop_norm") and dpd.inference.CropNormRunModel.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpd.cnn import HEAD_SIZES, adjust_logits, build_backbone

N_CROPS, N_DISEASES = HEAD_SIZES["crop"], HEAD_SIZES["disease"]


def crop_diversity(df) -> torch.Tensor:
    """Number of distinct diseases per crop in `df` (0 for a crop that does not occur)."""
    counts = df.groupby("crop_id")["disease_id"].nunique().reindex(range(N_CROPS), fill_value=0)
    return torch.tensor(counts.to_numpy(), dtype=torch.float32)


class CropNormNet(nn.Module):
    """Disease network with crop normalisation; with crop_norm.shared_backbone it also has a crop head on f."""

    def __init__(self, backbone: str, pretrained: bool, cn: dict, dropout: float, diversity: torch.Tensor | None = None):
        super().__init__()
        self.backbone, fdim = build_backbone(backbone, pretrained)
        d = cn["embed_dim"]
        self.transform = cn["transform"]
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, d))
        self.prototypes = nn.Parameter(torch.randn(N_DISEASES, d) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(cn["init_scale"])), requires_grad=cn["learn_scale"])
        if self.transform == "film":
            self.log_gain = nn.Parameter(torch.zeros(N_CROPS, d))
            self.shift = nn.Parameter(torch.zeros(N_CROPS, d))
        elif self.transform == "centroid":
            self.momentum = cn["centroid_momentum"]
            div = diversity if diversity is not None else torch.zeros(N_CROPS)
            self.register_buffer("crop_mean", torch.zeros(N_CROPS, d))
            self.register_buffer("global_mean", torch.zeros(d))
            self.register_buffer("crop_seen", torch.zeros(N_CROPS, dtype=torch.bool))
            self.register_buffer("shrink", (div - 1).clamp(min=0) / div.clamp(min=1))
        elif self.transform != "none":
            raise ValueError(f"unknown crop_norm.transform {self.transform!r}")
        self.crop_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fdim, N_CROPS)) if cn["shared_backbone"] else None

    def no_decay_parameters(self) -> list[nn.Parameter]:
        """Prototypes, scale and the crop transform: no weight decay (the transform has its own identity penalty)."""
        params = [self.prototypes, self.log_scale]
        if self.transform == "film":
            params += [self.log_gain, self.shift]
        return params

    def _gain_shift(self):
        """(log gain or None, shift), each (crops, d): n = (z - shift) * exp(-log gain)."""
        if self.transform == "film":
            return self.log_gain, self.shift
        return None, self.global_mean + self.shrink[:, None] * (self.crop_mean - self.global_mean)

    def normalise(self, z: torch.Tensor, crop: torch.Tensor | None = None) -> torch.Tensor:
        """z (B, d) -> n for each row's own crop (B, d), or for every crop (B, crops, d) when `crop` is None."""
        if self.transform == "none":
            return z if crop is not None else z[:, None, :].expand(-1, N_CROPS, -1)
        g, b = self._gain_shift()
        if crop is not None:
            n = z - b[crop]
            return n * torch.exp(-g[crop]) if g is not None else n
        n = z[:, None, :] - b[None]
        return n * torch.exp(-g)[None] if g is not None else n

    def denormalise(self, n: torch.Tensor, crop: torch.Tensor) -> torch.Tensor:
        """Inverse of normalise for each row's crop: T_c(n)."""
        if self.transform == "none":
            return n
        g, b = self._gain_shift()
        return (n * torch.exp(g[crop]) if g is not None else n) + b[crop]

    def disease_logits(self, n: torch.Tensor) -> torch.Tensor:
        """Scaled cosine similarity of n (..., d) to every disease prototype -> (..., diseases)."""
        return self.log_scale.exp() * F.normalize(n, dim=-1) @ F.normalize(self.prototypes, dim=-1).T

    @torch.no_grad()
    def update_centroids(self, z: torch.Tensor, crop: torch.Tensor):
        """Running means of z per crop and overall (centroid transform only); a crop's first batch sets its mean."""
        if self.transform != "centroid":
            return
        m = self.momentum
        batch_mean = z.mean(0)
        self.global_mean.copy_(m * self.global_mean + (1 - m) * batch_mean if self.crop_seen.any() else batch_mean)
        for c in crop.unique():
            mean = z[crop == c].mean(0)
            self.crop_mean[c] = m * self.crop_mean[c] + (1 - m) * mean if self.crop_seen[c] else mean
            self.crop_seen[c] = True


def cross_crop_supcon(n: torch.Tensor, disease: torch.Tensor, crop: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss on normalised features whose positives are the same disease on a DIFFERENT crop.

    This is what defines the base disease: after normalisation, rust on sugarcane and rust on corn should coincide.
    The same disease on the same crop is neither a positive nor a negative.
    """
    n = F.normalize(n, dim=-1)
    sim = n @ n.T / temperature
    same_d, same_c = disease[:, None] == disease[None], crop[:, None] == crop[None]
    pos = same_d & ~same_c
    sim = sim.masked_fill(same_d & same_c, -1e9)  # a large finite value (not -inf) keeps all-masked rows NaN-free
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return n.sum() * 0.0
    return (-(logprob.masked_fill(~pos, 0.0).sum(1)[has] / pos.sum(1)[has])).mean()


def losses(model: CropNormNet, f: torch.Tensor, z: torch.Tensor, crop: torch.Tensor, disease: torch.Tensor,
           cn: dict, lcfg: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Weighted loss terms, disease logits under the true crop, and crop logits (shared backbone only).

    z must be float32; the true crop is used for normalisation during training.
    """
    n = model.normalise(z, crop)
    logits = model.disease_logits(n)
    ls = lcfg["label_smoothing"]
    out = {"disease": lcfg["weights"]["disease"] * F.cross_entropy(logits, disease, label_smoothing=ls)}
    if cn["lambda_reverse"]:
        # Put the base disease back on this crop: T_c(|n| * p_d) should reproduce the image's own feature z.
        p = F.normalize(model.prototypes[disease], dim=-1) * n.norm(dim=-1, keepdim=True)
        out["reverse"] = cn["lambda_reverse"] * (1 - F.cosine_similarity(model.denormalise(p, crop), z, dim=-1)).mean()
    if cn["lambda_consistency"]:
        out["consistency"] = cn["lambda_consistency"] * cross_crop_supcon(n, disease, crop, cn["consistency_temperature"])
    if model.transform == "film" and cn["lambda_identity"]:
        out["identity"] = cn["lambda_identity"] * (model.log_gain.pow(2).mean() + model.shift.pow(2).mean())
    crop_logits = None
    if model.crop_head is not None:
        crop_logits = model.crop_head(f.float())
        out["crop"] = lcfg["weights"]["crop"] * F.cross_entropy(crop_logits, crop, label_smoothing=ls)
    return out, logits, crop_logits


class CrossCropBatchSampler:
    """Batches in which `n_pairs` slots are two images of one disease on two different crops (the consistency loss's
    positives, rare under uniform sampling: only 42 of 175 diseases occur on more than one crop); the rest is uniform.
    Same number of batches per epoch as a uniform loader with drop_last. Seeded from torch, so seed_everything applies.
    """

    def __init__(self, crop: np.ndarray, disease: np.ndarray, batch_size: int, n_pairs: int):
        self.n, self.batch_size, self.n_pairs = len(crop), batch_size, n_pairs
        groups: dict[int, dict[int, list[int]]] = {}
        for i, (c, d) in enumerate(zip(crop, disease)):
            groups.setdefault(int(d), {}).setdefault(int(c), []).append(i)
        self.groups = {d: [np.array(v) for v in cs.values()] for d, cs in groups.items() if len(cs) >= 2}
        self.diseases = np.array(sorted(self.groups))
        if not len(self.diseases):
            raise ValueError("no disease occurs on two crops; the cross-crop consistency loss has no positives")

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(2**31 - 1, (1,)).item()))
        for _ in range(len(self)):
            batch = []
            for d in rng.choice(self.diseases, self.n_pairs):
                a, b = rng.choice(len(self.groups[d]), 2, replace=False)
                batch += [int(rng.choice(self.groups[d][a])), int(rng.choice(self.groups[d][b]))]
            batch += rng.choice(self.n, self.batch_size - len(batch), replace=False).tolist()
            yield batch


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def compose_pair_scores(logp_crop: np.ndarray, disease_logits: np.ndarray, pairs, crop_weight: float = 0.5,
                        tau: float = 0.0, log_prior: dict | None = None) -> np.ndarray:
    """(images x 333) pair scores from crop log-probs (images x crops) and disease logits under every crop's
    normalisation (images x crops x diseases); columns follow pair_table order.

    score = 2 * (w * log P(crop c) + (1 - w) * log P(disease d | image normalised for c)), on the same scale as
    dpd.cnn.compose_pair_scores; tau is post-hoc logit adjustment by the training log prior of each head.
    """
    if tau:
        logp_crop = adjust_logits(logp_crop, tau, log_prior["crop"])
        disease_logits = disease_logits - tau * log_prior["disease"][None, None, :]
    c, d = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    logp_disease = disease_logits[:, c, d] - _logsumexp(disease_logits, axis=2)[:, c]
    return 2 * crop_weight * logp_crop[:, c] + 2 * (1 - crop_weight) * logp_disease
