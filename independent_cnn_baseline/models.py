"""Owned timm model/transform adapter; existing group CNN recipes stay unchanged."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T

PROJECT = Path(__file__).resolve().parents[1]
TARGETS = {"crop": 55, "disease": 175}


class IndependentNet(nn.Module):
    def __init__(self, config, target, pretrained=False):
        super().__init__()
        import timm

        options = {"num_classes": 0, "drop_rate": 0.0,
                   "drop_path_rate": config["model"]["drop_path_rate"]}
        if pretrained:
            path = Path(config["model"]["pretrained_file"])
            if not path.is_absolute():
                path = PROJECT / path
            if not path.is_file():
                raise FileNotFoundError(f"Pinned pretrained weights missing: {path}")
            options["pretrained_cfg_overlay"] = {"file": str(path), "custom_load": False}
        self.backbone = timm.create_model(config["model"]["backbone"], pretrained=pretrained, **options)
        self.head = nn.Sequential(nn.Dropout(config["model"]["head_dropout"]),
                                  nn.Linear(self.backbone.num_features, TARGETS[target]))
        self.target = target

    def forward(self, x):
        return self.head(self.backbone(x))


def freeze_batchnorm_statistics(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # Affine weight/bias remain trainable.


def transforms(config):
    inp, aug = config["input"], config["augmentation"]
    interpolation = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}[inp["interpolation"]]
    normalize = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(inp["mean"], inp["std"])]
    train = T.Compose([
        T.RandomResizedCrop(inp["crop_size"], scale=tuple(aug["random_resized_crop_scale"]),
                            interpolation=interpolation, antialias=True),
        T.RandomHorizontalFlip(aug["horizontal_flip_p"]),
        T.RandomVerticalFlip(aug["vertical_flip_p"]),
        T.ColorJitter(**aug["color_jitter"]), *normalize])
    test = T.Compose([T.Resize(inp["resize"], interpolation=interpolation, antialias=True),
                      T.CenterCrop(inp["crop_size"]), *normalize])
    return train, test


class CachedImages(Dataset):
    def __init__(self, frame, cache_dir, transform):
        self.files = [Path(cache_dir) / (Path(f).stem + ".jpg") for f in frame["file"]]
        self.crop = frame["crop_id"].to_numpy()
        self.disease = frame["disease_id"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(self.crop[index]), int(self.disease[index])


def optimizer(model, config):
    """Separate backbone/head LRs; no decay on biases or normalization scalars."""
    ocfg = config["optimizer"]
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith("head.")
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups.setdefault((head, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": params, "lr": ocfg["lr_head" if head else "lr_backbone"],
         "initial_lr": ocfg["lr_head" if head else "lr_backbone"],
         "weight_decay": ocfg["weight_decay"] if decay else 0.0}
        for (head, decay), params in groups.items()], betas=tuple(ocfg["betas"]), eps=ocfg["eps"])
