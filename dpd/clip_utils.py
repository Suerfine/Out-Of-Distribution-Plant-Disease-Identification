"""CLIP (open_clip) loading, prompt building and embedding, driven by the config's model/names/prompts sections."""

import hashlib
import json
import re

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm

from dpd.data import PROJECT_ROOT, load_image

CACHE_DIR = PROJECT_ROOT / "runs" / "cache"


class ImageDataset(Dataset):
    def __init__(self, files, preprocess):
        self.files, self.preprocess = list(files), preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.preprocess(load_image(self.files[i]))


def load_clip(model_cfg: dict, device: str):
    """arch is an open_clip model name (with `pretrained` naming the weights) or an 'hf-hub:org/name' id (pretrained null).
    Optional `image_size` (e.g. 336) runs the image encoder at that resolution; open_clip interpolates the position
    embeddings of the pretrained weights to the larger patch grid."""
    extra = {"force_image_size": model_cfg["image_size"]} if model_cfg.get("image_size") else {}
    model, _, preprocess = open_clip.create_model_and_transforms(model_cfg["arch"], pretrained=model_cfg["pretrained"], **extra)
    tokenizer = open_clip.get_tokenizer(model_cfg["arch"])
    return model.to(device).eval(), preprocess, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, files, model_cfg: dict, runtime: dict, cache_name: str | None = None) -> np.ndarray:
    """L2-normalised image embeddings (n_images x dim). Cached on disk: the encoder is frozen, so they never change."""
    files = list(files)
    cache = None
    if cache_name:
        key = hashlib.sha1("\n".join(files).encode()).hexdigest()[:10]
        model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{model_cfg['arch']}_{model_cfg['pretrained']}")  # hf-hub ids contain ':' and '/'
        cache = CACHE_DIR / f"{model_tag}_{cache_name}_{key}.npy"
        if cache.exists():
            return np.load(cache)
    loader = DataLoader(
        ImageDataset(files, preprocess),
        batch_size=runtime["batch_size"],
        num_workers=runtime["num_workers"],
        pin_memory=runtime["device"] == "cuda",
    )
    feats = []
    for x in tqdm(loader, desc=f"encode {cache_name or 'images'}"):
        with torch.autocast(runtime["device"], enabled=runtime["amp"]):
            f = model.encode_image(x.to(runtime["device"], non_blocking=True))
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    feats = torch.cat(feats).numpy()
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, feats)
    return feats


def display_name(raw: str, overrides: dict, underscore_to_space: bool) -> str:
    if raw in overrides:
        return overrides[raw]
    return raw.replace("_", " ") if underscore_to_space else raw


def load_descriptions(prompts_cfg: dict) -> dict:
    """Raw disease name -> description, from prompts.descriptions_file (relative to the project root), or {}."""
    path = prompts_cfg.get("descriptions_file")
    if not path:
        return {}
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def build_prompts(pairs: pd.DataFrame, names_cfg: dict, prompts_cfg: dict) -> list[list[str]]:
    """One list of prompt strings per pair id (one per template); the class embedding is their mean.

    Templates may use {crop}, {disease} and {description}; the last needs prompts.descriptions_file.
    """
    u2s = names_cfg["underscore_to_space"]
    descriptions = load_descriptions(prompts_cfg)
    out = []
    for r in pairs.itertuples(index=False):
        crop = display_name(r.crop, names_cfg["crop_overrides"], u2s)
        disease = display_name(r.disease, names_cfg["disease_overrides"], u2s)
        templates = prompts_cfg["healthy_templates"] if r.disease == "healthy" else prompts_cfg["templates"]
        if any("{description}" in t for t in templates) and r.disease not in descriptions:
            raise KeyError(f"No description for disease '{r.disease}' in {prompts_cfg.get('descriptions_file')}")
        out.append([t.format(crop=crop, disease=disease, description=descriptions.get(r.disease, "")) for t in templates])
    return out


def truncated_prompts(tokenizer, prompts: list[list[str]]) -> list[str]:
    """Prompts longer than CLIP's 77-token context (the tokenizer cuts them silently)."""
    flat = [p for ps in prompts for p in ps]
    tokens = tokenizer(flat)
    return [p for p, t in zip(flat, tokens) if t[-1] != 0]


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts: list[list[str]], device: str) -> np.ndarray:
    """L2-normalised class embeddings (n_classes x dim): mean of each class's normalised template embeddings."""
    embs = []
    for class_prompts in prompts:
        t = model.encode_text(tokenizer(class_prompts).to(device)).float()
        t = torch.nn.functional.normalize(t, dim=-1).mean(dim=0)
        embs.append(torch.nn.functional.normalize(t, dim=-1).cpu())
    return torch.stack(embs).numpy()


CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def clip_transforms(input_cfg: dict, aug_cfg: dict):
    """Train: augmented crops; test: CLIP's own resize + centre crop. Both on the resize-cache images."""
    size = input_cfg["crop_size"]
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(CLIP_MEAN, CLIP_STD)]
    train = [T.RandomResizedCrop(size, scale=tuple(aug_cfg["random_resized_crop_scale"]),
                                 interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    if aug_cfg["horizontal_flip_p"]:
        train.append(T.RandomHorizontalFlip(aug_cfg["horizontal_flip_p"]))
    if aug_cfg["vertical_flip_p"]:
        train.append(T.RandomVerticalFlip(aug_cfg["vertical_flip_p"]))
    if any(aug_cfg["color_jitter"].values()):
        train.append(T.ColorJitter(**aug_cfg["color_jitter"]))
    test = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(size)]
    return T.Compose(train + norm), T.Compose(test + norm)


# Test-time augmentation views (on a batch of image tensors); embeddings of the views are averaged.
TTA_VIEWS = {
    "identity": lambda x: x,
    "hflip": lambda x: x.flip(-1),
    "vflip": lambda x: x.flip(-2),
    "hvflip": lambda x: x.flip(-1).flip(-2),
}
