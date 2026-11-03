"""Supervised contrastive learning (SupCon) for crop and disease, scored with class prototypes.

An ImageNet-pretrained backbone feeds two projection heads, one for the crop and one for the disease. Each head is
trained with the SupCon loss on its own label, so images of the same crop (or disease) are pulled together in that
head's embedding space and the rest are pushed apart. Classification then needs no trained classifier:
    P(crop = c)    = softmax over crops    of cos(z_crop, prototype_c) / temperature
    P(disease = d) = softmax over diseases of cos(z_disease, prototype_d) / temperature
where a prototype is the mean embedding of that class's training images. The pair score is
log P(crop) + log P(disease), the same composition as the CNN baselines (dpd.cnn.compose_pair_scores), so all 333
pairs, including the 10 unseen ones, get a score.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from dpd.cnn import HEAD_SIZES, build_backbone
from dpd.resize_cache import cached_name

HEADS = tuple(HEAD_SIZES)  # ("crop", "disease")


class ContrastiveNet(nn.Module):
    """Backbone + one projection MLP per head; the output embeddings are L2-normalised."""

    def __init__(self, backbone: str, pretrained: bool, proj_dim: int, proj_hidden: int | None = None,
                 ce_dropout: float | None = None, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained, drop_path_rate)
        hidden = proj_hidden or dim
        self.proj = nn.ModuleDict(
            {h: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, proj_dim)) for h in HEADS}
        )
        # Optional linear classifier heads on the backbone features (as in the CNN baselines), trained with
        # cross-entropy next to the SupCon loss. None = pure SupCon.
        self.cls = None
        if ce_dropout is not None:
            self.cls = nn.ModuleDict(
                {h: nn.Sequential(nn.Dropout(ce_dropout), nn.Linear(dim, HEAD_SIZES[h])) for h in HEADS}
            )

    def classify(self, f) -> dict[str, torch.Tensor]:
        return {h: c(f) for h, c in self.cls.items()}

    def encode(self, x) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Backbone features and the normalised embedding of each head."""
        f = self.backbone(x)
        return f, {h: F.normalize(p(f).float(), dim=-1) for h, p in self.proj.items()}

    def forward(self, x) -> dict[str, torch.Tensor]:
        return self.encode(x)[1]


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon (Khosla et al. 2020). `z` is (N, d), L2-normalised, holding every view of every image in the batch;
    `labels` is (N,). Positives of an anchor = every other embedding with the same label (including its other view)."""
    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    keep = n_pos > 0  # an anchor with no positive (only possible with a single view) is skipped
    mean_log_prob_pos = log_prob.masked_fill(~pos, 0.0).sum(1)[keep] / n_pos[keep]
    return -mean_log_prob_pos.mean()


class TwoViewImageLabels(Dataset):
    """Images from the resized cache, each returned as two independently augmented views plus both labels."""

    def __init__(self, df, image_dir, transform):
        self.files = [image_dir / cached_name(f) for f in df["file"]]
        self.crop = df["crop_id"].to_numpy()
        self.disease = df["disease_id"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.transform(img), int(self.crop[i]), int(self.disease[i])


@torch.no_grad()
def embed(model: ContrastiveNet, loader, device: str, amp: bool) -> dict[str, np.ndarray]:
    """Per head the normalised embeddings, plus "feat" (the backbone features) for every image in the loader
    (an ImageLabels loader, test transform)."""
    model.eval()
    logp_keys = [f"logp_{h}" for h in HEADS] if model.cls is not None else []  # the CE heads' log-probabilities
    out = {k: [] for k in (*HEADS, "feat", *logp_keys)}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            f, z = model.encode(x.to(device, non_blocking=True))
            logits = model.classify(f) if model.cls is not None else {}
        out["feat"].append(f.float().cpu())
        for h in HEADS:
            out[h].append(z[h].cpu())
            if logits:
                out[f"logp_{h}"].append(F.log_softmax(logits[h].float(), dim=-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def class_prototypes(emb: np.ndarray, labels: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_classes x d) unit-length mean embeddings and a mask of the classes that have at least one image."""
    protos = np.zeros((n_classes, emb.shape[1]), np.float32)
    np.add.at(protos, labels, emb)
    valid = np.bincount(labels, minlength=n_classes) > 0
    protos[valid] /= np.maximum(np.linalg.norm(protos[valid], axis=1, keepdims=True), 1e-12)
    return protos, valid


def fit_linear_heads(feat: np.ndarray, labels: dict[str, np.ndarray], ccfg: dict, device: str) -> dict:
    """Scoring alternative to the prototypes: one linear softmax classifier per head, trained with cross-entropy on the
    frozen backbone features (standardised with the fit set's mean and std). Full-batch AdamW for `steps` steps.
    Classes with no training image are masked out of both the training softmax and the scores."""
    x = torch.from_numpy(feat).float().to(device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    heads = {}
    for h in HEADS:
        y = torch.from_numpy(labels[h]).long().to(device)
        n = HEAD_SIZES[h]
        valid = torch.bincount(y, minlength=n) > 0
        lin = nn.Linear(x.shape[1], n).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=ccfg["lr"], weight_decay=ccfg["weight_decay"])
        for _ in range(ccfg["steps"]):
            loss = F.cross_entropy(lin(x).masked_fill(~valid, -1e4), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        heads[h] = (lin.eval(), valid)
    return {"mean": mean, "std": std, "heads": heads, "device": device}


@torch.no_grad()
def linear_log_probs(feat: np.ndarray, clf: dict, head: str) -> np.ndarray:
    """Log-probabilities of one head's classifier for the given backbone features."""
    x = (torch.from_numpy(feat).float().to(clf["device"]) - clf["mean"]) / clf["std"]
    lin, valid = clf["heads"][head]
    return F.log_softmax(lin(x).masked_fill(~valid, -1e4), dim=-1).cpu().numpy()


def log_probs(emb: np.ndarray, protos: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    """Log-softmax over classes of cosine similarity / temperature; classes without a prototype get ~zero probability."""
    logits = emb @ protos.T / temperature
    logits[:, ~valid] = -1e4
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
