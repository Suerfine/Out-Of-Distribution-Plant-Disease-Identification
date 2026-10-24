"""Load a finished run from runs/vN and score new images with it. No Streamlit imports (the app wraps these in caches)."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dpd.clip_utils import build_prompts, encode_images, encode_prompts, load_clip
from dpd.data import PROJECT_ROOT, pair_table

RUNS_DIR = PROJECT_ROOT / "runs"
SUPPORTED_METHODS = ("clip_zeroshot", "clip_prompt_learning")


def list_runs() -> list[dict]:
    """Finished runs (config.json + metrics.json present) of a supported method, newest version first."""
    runs = []
    for d in RUNS_DIR.glob("v*"):
        if not ((d / "config.json").exists() and (d / "metrics.json").exists()):
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("method") not in SUPPORTED_METHODS:
            continue
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        runs.append({"dir": d, "version": cfg["version"], "method": cfg["method"], "cfg": cfg, "metrics": metrics})
    return sorted(runs, key=lambda r: int(r["version"].lstrip("v")), reverse=True)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def pair_text_embeddings(run_dir: Path, clip_model, tokenizer, pairs) -> np.ndarray:
    """(333 x dim) L2-normalised text embedding per pair, as used to produce the run's test results."""
    saved = run_dir / "pair_text_embeddings.npy"
    if saved.exists():
        return np.load(saved)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg["method"] == "clip_zeroshot":
        return encode_prompts(clip_model, tokenizer, build_prompts(pairs, cfg["names"], cfg["prompts"]), device())
    # Older prompt-learning runs only saved the vectors: rebuild the prompts and re-encode.
    from dpd.prompt_learner import build_model

    model = build_model(cfg, clip_model, tokenizer, pairs).to(device())
    state = torch.load(run_dir / "prompt_vectors.pt", map_location=device())
    model.crop_vecs.data.copy_(state["crop_vecs"])
    model.disease_vecs.data.copy_(state["disease_vecs"])
    return model.encode("pairs").float().cpu().numpy()


@torch.no_grad()
def embed_image(clip_model, preprocess, image: Image.Image) -> np.ndarray:
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device())
    f = clip_model.encode_image(x).float()
    return torch.nn.functional.normalize(f, dim=-1)[0].cpu().numpy()


def train_embeddings(clip_model, preprocess, model_cfg: dict, train_files) -> np.ndarray:
    """Image embeddings of every training image (from the shared cache when it exists), for retrieval."""
    runtime = {"device": device(), "amp": device() == "cuda", "batch_size": 64, "num_workers": 0}
    return encode_images(clip_model, preprocess, train_files, model_cfg, runtime, cache_name="train")


def rank_pairs(image_emb: np.ndarray, pair_emb: np.ndarray, unseen_mask: np.ndarray, gamma: float, logit_scale: float = 100.0):
    """Calibrated scores and softmax confidences over all pairs, highest first."""
    scores = pair_emb @ image_emb - gamma * (~unseen_mask)
    probs = np.exp(logit_scale * (scores - scores.max()))
    probs /= probs.sum()
    order = np.argsort(-scores)
    return order, scores, probs


def nearest(image_emb: np.ndarray, bank: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = bank @ image_emb
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def load_run_model(run: dict, df):
    """Everything the app needs for one run: CLIP, its pair embeddings and gamma."""
    clip_model, preprocess, tokenizer = load_clip(run["cfg"]["model"], device())
    pairs = pair_table(df)
    return {
        "clip": clip_model,
        "preprocess": preprocess,
        "pairs": pairs,
        "pair_emb": pair_text_embeddings(run["dir"], clip_model, tokenizer, pairs),
        "gamma": float(run["metrics"].get("gamma", 0.0)),
    }
