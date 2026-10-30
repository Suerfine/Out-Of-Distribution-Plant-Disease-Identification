"""CNN baselines: an ImageNet-pretrained torchvision backbone with one linear head per target (crop, disease).

Pair scores for the shared evaluation are composed from the two heads:
    score(crop c, disease d) = log P(crop = c) + log P(disease = d)
which treats crop and disease as independent, so every one of the 333 pairs (including the 10 unseen ones)
gets a score even though only crop and disease were trained.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from dpd.resize_cache import cached_name

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HEAD_SIZES = {"crop": 55, "disease": 175}
# Networks per method: name -> heads. Each network is saved as <name>.pt in the run folder.
NETWORKS = {
    "cnn_independent": {"crop_net": ["crop"], "disease_net": ["disease"]},
    "cnn_multitask": {"multitask_net": ["crop", "disease"]},
}


def build_backbone(name: str, pretrained: bool):
    """Feature extractor (classifier removed) and its output width."""
    if name == "resnet50":
        net = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        net = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name == "convnext_tiny":
        net = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"unknown backbone {name!r}")
    return net, dim


class MultiHeadNet(nn.Module):
    """Shared backbone, one linear head per target. One head = a single-task CNN; two heads = multi-task."""

    def __init__(self, backbone: str, pretrained: bool, heads: dict[str, int], dropout: float):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained)
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n)) for k, n in heads.items()})

    def forward(self, x):
        f = self.backbone(x)
        return {k: h(f) for k, h in self.heads.items()}


class ImageLabels(Dataset):
    """Images from the resized cache with their crop and disease labels."""

    def __init__(self, df, image_dir, transform):
        self.files = [image_dir / cached_name(f) for f in df["file"]]
        self.crop = df["crop_id"].to_numpy()
        self.disease = df["disease_id"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as img:
            x = self.transform(img.convert("RGB"))
        return x, int(self.crop[i]), int(self.disease[i])


def build_transforms(input_cfg: dict, aug_cfg: dict):
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    size = input_cfg["crop_size"]
    train = [T.RandomResizedCrop(size, scale=tuple(aug_cfg["random_resized_crop_scale"]), antialias=True)]
    if aug_cfg["horizontal_flip_p"]:
        train.append(T.RandomHorizontalFlip(aug_cfg["horizontal_flip_p"]))
    if aug_cfg["vertical_flip_p"]:
        train.append(T.RandomVerticalFlip(aug_cfg["vertical_flip_p"]))
    cj = aug_cfg["color_jitter"]
    if any(cj.values()):
        train.append(T.ColorJitter(**cj))
    test = [T.Resize(input_cfg["resize"], antialias=True), T.CenterCrop(size)]
    return T.Compose(train + norm), T.Compose(test + norm)


def log_priors(df) -> dict[str, np.ndarray]:
    """Log class frequency per head in `df` (add-one smoothed, so a class missing from `df` still has a finite prior)."""
    return {h: np.log((np.bincount(df[f"{h}_id"], minlength=n) + 1) / (len(df) + n)) for h, n in HEAD_SIZES.items()}


def adjust_logits(logp: np.ndarray, tau: float, log_prior: np.ndarray) -> np.ndarray:
    """Post-hoc logit adjustment (Menon et al., 2021): log-softmax of logp - tau * log prior.

    Removes (part of) the training-frequency bias from a head, so rare crops / diseases are not drowned out by
    common ones. tau 0 returns logp unchanged; tau 1 is the full Bayes correction towards a uniform prior.
    """
    if not tau:
        return logp
    z = logp - tau * log_prior[None, :]
    m = z.max(axis=1, keepdims=True)
    return z - (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))


def compose_pair_scores(logp_crop: np.ndarray, logp_disease: np.ndarray, pairs, crop_weight: float = 0.5,
                        tau: float = 0.0, log_prior: dict | None = None) -> np.ndarray:
    """(images x 333) pair scores from per-head log-probabilities; columns follow pair_table order.

    score = 2 * (w * log P(crop) + (1 - w) * log P(disease)), each head optionally logit-adjusted by `tau`.
    The defaults (w 0.5, tau 0) give exactly log P(crop) + log P(disease), the baselines' score; the factor 2 keeps
    that scale for every w, so the gamma grid means the same thing.
    """
    if tau:
        logp_crop = adjust_logits(logp_crop, tau, log_prior["crop"])
        logp_disease = adjust_logits(logp_disease, tau, log_prior["disease"])
    return (2 * crop_weight * logp_crop[:, pairs["crop_id"].to_numpy()]
            + 2 * (1 - crop_weight) * logp_disease[:, pairs["disease_id"].to_numpy()])
