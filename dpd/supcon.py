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


class CosineLinear(nn.Module):
    """Cosine classifier: scale * cos(feature, class weight). Opt-in replacement for the linear head."""

    def __init__(self, dim, n_out, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.scale = scale

    def forward(self, x):
        return self.scale * F.linear(F.normalize(x.float(), dim=-1), F.normalize(self.weight.float(), dim=-1))


class ContrastiveNet(nn.Module):
    """Backbone + one projection MLP per head; the output embeddings are L2-normalised."""

    def __init__(self, backbone: str, pretrained: bool, proj_dim: int, proj_hidden: int | None = None,
                 ce_dropout: float | None = None, drop_path_rate: float = 0.0, cosine_scale: float | None = None,
                 head_hidden: int | None = None):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained, drop_path_rate)
        self.feat_dim = dim
        hidden = proj_hidden or dim
        self.proj = nn.ModuleDict(
            {h: nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, proj_dim)) for h in HEADS}
        )
        # Optional linear classifier heads on the backbone features (as in the CNN baselines), trained with
        # cross-entropy next to the SupCon loss. None = pure SupCon.
        self.cls = None
        self.logit_tau = None  # LogitNorm: when set, classify() returns logits / (tau * ||logits||)
        if ce_dropout is not None:
            def make_head(h):  # linear (default), cosine (opt-in) or two-layer MLP (opt-in)
                if cosine_scale:
                    return nn.Sequential(nn.Dropout(ce_dropout), CosineLinear(dim, HEAD_SIZES[h], cosine_scale))
                if head_hidden:
                    return nn.Sequential(nn.Dropout(ce_dropout), nn.Linear(dim, head_hidden), nn.GELU(), nn.Dropout(ce_dropout),
                                         nn.Linear(head_hidden, HEAD_SIZES[h]))
                return nn.Sequential(nn.Dropout(ce_dropout), nn.Linear(dim, HEAD_SIZES[h]))

            self.cls = nn.ModuleDict({h: make_head(h) for h in HEADS})
        self.tta = False  # opt-in: average the head outputs over the image and its flips when embedding

    def classify(self, f) -> dict[str, torch.Tensor]:
        out = {h: c(f) for h, c in self.cls.items()}
        if self.logit_tau:
            out = {h: l.float() / (l.float().norm(dim=-1, keepdim=True) * self.logit_tau + 1e-7) for h, l in out.items()}
        return out

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
    aux = getattr(model, "aux", None)  # opt-in terms (dpd/new_terms.py)
    has_comp = aux is not None and hasattr(aux, "comp")
    out = {k: [] for k in (*HEADS, "feat", *logp_keys, *(["logp_comp"] if has_comp else []))}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            f, z = model.encode(x.to(device, non_blocking=True))
            logits = model.classify(f) if model.cls is not None else {}
            if model.tta and logits:  # flip test-time augmentation (horizontal and vertical, as in training)
                xs = x.to(device, non_blocking=True)
                for flip in ((-1,), (-2,)):
                    lf = model.classify(model.backbone(xs.flip(flip)))
                    logits = {h: logits[h].float() + lf[h].float() for h in HEADS}
                logits = {h: l / 3 for h, l in logits.items()}
        out["feat"].append(f.float().cpu())
        if has_comp:
            out["logp_comp"].append(aux.comp_logp(f.float()).cpu())
        for h in HEADS:
            out[h].append(z[h].cpu())
            if logits:
                out[f"logp_{h}"].append(F.log_softmax(logits[h].float(), dim=-1).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    if has_comp:
        res["fuse_w"] = np.array([float(aux.fusion_weight())])
    return res


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


def cnc_loss(z: torch.Tensor, disease: torch.Tensor, crop: torch.Tensor, temperature: float) -> torch.Tensor:
    """Correct-n-Contrast-style loss for the disease head. `z` is (2N, d) with the two views of image i at rows i and
    i + N; `disease` and `crop` are (2N,).
    Positives of an anchor: its other view, and every image with the SAME disease on a DIFFERENT crop (so the crop cannot
    help to match them). Negatives: every image with a different disease (the batch sampler adds other diseases on the
    same crop, so the crop cannot help to separate them either). Images with the same disease on the same crop are
    left out of both the positives and the denominator: matching them is exactly the crop shortcut."""
    n2 = z.shape[0]
    n = n2 // 2
    idx = torch.arange(n2, device=z.device)
    self_mask = idx[:, None] == idx[None, :]
    own = (idx[:, None] - idx[None, :]).abs() == n
    same_d = disease[:, None] == disease[None, :]
    same_c = crop[:, None] == crop[None, :]
    pos = (same_d & ~same_c) | own
    neutral = same_d & same_c & ~own & ~self_mask
    logits = (z @ z.T / temperature).masked_fill(self_mask | neutral, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    mean_log_prob_pos = log_prob.masked_fill(~pos, 0.0).sum(1) / pos.sum(1)  # every anchor has its own other view
    return -mean_log_prob_pos.mean()


class CnCBatchSampler:
    """Batches for cnc_loss. Each batch = `groups` groups plus random images. A group: one disease that occurs on at least
    two crops in the fit set, `per_crop` images of it on each of two crops (cross-crop positives), and `hard_neg` images
    of other diseases on the first of those crops (same-crop hard negatives). The rest of the batch is uniform random,
    so the class mix stays close to the natural one. Built from the fit images only."""

    def __init__(self, crop, disease, batch_size: int, groups: int, per_crop: int, hard_neg: int, seed: int):
        self.crop, self.disease = np.asarray(crop), np.asarray(disease)
        self.batch_size, self.groups, self.per_crop, self.hard_neg = batch_size, groups, per_crop, hard_neg
        self.by_pair, self.by_crop, crops_of = {}, {}, {}
        for i, (c, d) in enumerate(zip(self.crop, self.disease)):
            self.by_pair.setdefault((c, d), []).append(i)
            self.by_crop.setdefault(c, []).append(i)
            crops_of.setdefault(d, set()).add(c)
        self.by_pair = {k: np.array(v) for k, v in self.by_pair.items()}
        self.by_crop = {k: np.array(v) for k, v in self.by_crop.items()}
        self.multi = {d: sorted(cs) for d, cs in crops_of.items() if len(cs) >= 2}
        self.multi_ids = np.array(sorted(self.multi))
        self.n_batches = len(self.crop) // batch_size
        self.rng = np.random.default_rng(seed)
        assert groups * (2 * per_crop + hard_neg) <= batch_size, "groups do not fit in the batch"

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        r = self.rng
        for _ in range(self.n_batches):
            batch = []
            if len(self.multi_ids):
                for d in r.choice(self.multi_ids, self.groups, replace=len(self.multi_ids) < self.groups):
                    c1, c2 = r.choice(self.multi[d], 2, replace=False)
                    for c in (c1, c2):
                        pool = self.by_pair[(c, d)]
                        batch += r.choice(pool, self.per_crop, replace=len(pool) < self.per_crop).tolist()
                    pool = self.by_crop[c1]
                    pool = pool[self.disease[pool] != d]
                    if len(pool):
                        batch += r.choice(pool, self.hard_neg, replace=len(pool) < self.hard_neg).tolist()
            batch += r.integers(0, len(self.crop), self.batch_size - len(batch)).tolist()
            yield batch
