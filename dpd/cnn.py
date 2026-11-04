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
    "multi_task_cnn": {"multitask_net": ["crop", "disease"]},
}


def build_backbone(name: str, pretrained: bool, drop_path_rate: float = 0.0):
    """Feature extractor (classifier removed) and its output width.

    "timm:<model>" loads any timm model with its pooled features (e.g. "timm:convnext_base.fb_in22k_ft_in1k",
    ImageNet-22k pretraining); drop_path_rate (stochastic depth) applies to timm models only.
    """
    if name.startswith("timm:"):
        import timm

        net = timm.create_model(name[len("timm:"):], pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        return net, net.num_features
    if name in ("resnet18", "resnet34"):
        fn = getattr(torchvision.models, name)
        net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name in ("resnet50", "resnet101"):
        fn = getattr(torchvision.models, name)
        net = fn(weights="IMAGENET1K_V2" if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name in ("efficientnet_b0", "efficientnet_b2", "efficientnet_b3"):
        fn = getattr(torchvision.models, name)
        net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name == "efficientnet_b1":
        net = torchvision.models.efficientnet_b1(weights="IMAGENET1K_V2" if pretrained else None)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name in ("convnext_tiny", "convnext_small"):
        fn = getattr(torchvision.models, name)
        net = fn(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
    elif name == "vit_b_16":
        net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        dim = net.heads.head.in_features
        net.heads.head = nn.Identity()
    else:
        try:
            import timm
            net = timm.create_model(name, pretrained=pretrained, num_classes=0)
            dim = net.num_features
        except Exception:
            raise ValueError(f"unknown backbone {name!r}")
    return net, dim


class MultiHeadNet(nn.Module):
    """Shared backbone, one linear head per target. One head = a single-task CNN; two heads = multi-task."""

    def __init__(self, backbone: str, pretrained: bool, heads: dict[str, int], dropout: float, drop_path_rate: float = 0.0):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained, drop_path_rate)
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
    # Resampling for crops and resizes; bilinear is torchvision's default (ConvNeXt was pretrained with bicubic).
    interp = T.InterpolationMode[input_cfg.get("interpolation", "bilinear").upper()]
    train = [T.RandomResizedCrop(size, scale=tuple(aug_cfg["random_resized_crop_scale"]), interpolation=interp, antialias=True)]
    if aug_cfg["horizontal_flip_p"]:
        train.append(T.RandomHorizontalFlip(aug_cfg["horizontal_flip_p"]))
    if aug_cfg["vertical_flip_p"]:
        train.append(T.RandomVerticalFlip(aug_cfg["vertical_flip_p"]))
    cj = aug_cfg["color_jitter"]
    if any(cj.values()):
        train.append(T.ColorJitter(**cj))
    # Optional stronger augmentation (all off when the keys are absent, so existing configs are unchanged).
    if aug_cfg.get("rotation_degrees"):
        train.append(T.RandomRotation(aug_cfg["rotation_degrees"]))
    if aug_cfg.get("random_grayscale_p"):
        train.append(T.RandomGrayscale(aug_cfg["random_grayscale_p"]))
    if aug_cfg.get("gaussian_blur_p"):
        train.append(T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=aug_cfg["gaussian_blur_p"]))
    post = []
    if aug_cfg.get("random_erasing_p"):
        post.append(T.RandomErasing(p=aug_cfg["random_erasing_p"], scale=(0.02, 0.2), value=0))
    test = [T.Resize(input_cfg["resize"], interpolation=interp, antialias=True), T.CenterCrop(size)]
    return T.Compose(train + norm + post), T.Compose(test + norm)


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
