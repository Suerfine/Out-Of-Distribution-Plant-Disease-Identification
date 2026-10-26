"""Load a finished run from runs/<family>/<name> and score new images with it. No Streamlit imports (the app wraps these in caches)."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dpd.clip_utils import build_prompts, encode_images, encode_prompts, load_clip
from dpd.data import PROJECT_ROOT, pair_table

RUNS_DIR = PROJECT_ROOT / "runs"
SUPPORTED_METHODS = ("clip_zeroshot", "clip_prompt_learning", "cnn_independent", "cnn_multitask")


def list_runs(methods=SUPPORTED_METHODS) -> list[dict]:
    """Finished runs (config.json + metrics.json present) under runs/<family>/<name>/ whose method is in `methods`.

    Families are runs/vlm (versions vN, newest first) and runs/baseline (named runs, after the VLM runs).
    """
    runs = []
    for d in RUNS_DIR.glob("*/*"):
        if d.parent.name == "cache" or not ((d / "config.json").exists() and (d / "metrics.json").exists()):
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("method") not in methods:
            continue
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        runs.append({"dir": d, "family": d.parent.name, "version": cfg["version"], "method": cfg["method"], "cfg": cfg, "metrics": metrics})

    def order(r):
        m = re.fullmatch(r"v(\d+)", r["version"])
        return (0, -int(m.group(1))) if m else (1, r["version"])

    return sorted(runs, key=order)


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


def rank_scores(scores: np.ndarray, unseen_mask: np.ndarray, gamma: float, logit_scale: float):
    """Calibrated scores and softmax confidences over all pairs, highest first."""
    scores = scores - gamma * (~unseen_mask)
    probs = np.exp(logit_scale * (scores - scores.max()))
    probs /= probs.sum()
    return np.argsort(-scores), scores, probs


def nearest(image_emb: np.ndarray, bank: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = bank @ image_emb
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


class ClipRunModel:
    """A zero-shot or prompt-learning run: score = cosine(image, pair text embedding)."""

    logit_scale = 100.0  # CLIP's temperature, for turning scores into confidences
    score_help = "Cosine similarity between the image and the pair's text embedding, after calibration"

    def __init__(self, run: dict, pairs):
        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        self.clip, self.preprocess, tokenizer = load_clip(run["cfg"]["model"], device())
        self.pair_emb = pair_text_embeddings(run["dir"], self.clip, tokenizer, pairs)

    def predict(self, src):
        """(retrieval embedding, uncalibrated score per pair) for an image path or file-like object."""
        with Image.open(src) as img:
            emb = embed_image(self.clip, self.preprocess, img)
        return emb, self.pair_emb @ emb

    def train_bank(self, train) -> np.ndarray:
        return train_embeddings(self.clip, self.preprocess, self.run["cfg"]["model"], train["file"])


class CnnRunModel:
    """A CNN baseline run: score = log P(crop) + log P(disease) from the saved network(s)."""

    logit_scale = 1.0  # scores are already log-probabilities
    score_help = "log P(crop) + log P(disease), after calibration"

    def __init__(self, run: dict, pairs):
        from dpd.cnn import HEAD_SIZES, NETWORKS, MultiHeadNet, build_transforms

        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        cfg = run["cfg"]
        self.transform = build_transforms(cfg["input"], cfg["augmentation"])[1]
        self.nets = {}
        for name, heads in NETWORKS[cfg["method"]].items():
            net = MultiHeadNet(cfg["model"]["backbone"], False, {h: HEAD_SIZES[h] for h in heads}, cfg["model"]["head_dropout"])
            net.load_state_dict(torch.load(run["dir"] / f"{name}.pt", map_location="cpu"))
            self.nets[name] = net.to(device()).eval()

    @torch.no_grad()
    def _forward(self, x):
        """Concatenated L2-normalised backbone features of every network, and log-probs per head."""
        feats, logp = [], {}
        with torch.autocast(device(), enabled=device() == "cuda"):
            for net in self.nets.values():
                f = net.backbone(x)
                feats.append(torch.nn.functional.normalize(f.float(), dim=-1))
                for h, head in net.heads.items():
                    logp[h] = torch.log_softmax(head(f).float(), dim=-1)
        emb = torch.nn.functional.normalize(torch.cat(feats, dim=-1), dim=-1)
        return emb.cpu().numpy(), {h: v.cpu().numpy() for h, v in logp.items()}

    def predict(self, src):
        """(retrieval embedding, uncalibrated score per pair) for an image path or file-like object."""
        from dpd.cnn import compose_pair_scores
        from dpd.resize_cache import standardise

        # Same downscale + JPEG round trip as the training/evaluation cache, so scores match the evaluated model.
        x = self.transform(standardise(src, self.run["cfg"]["input"]["resize_cache"])).unsqueeze(0).to(device())
        emb, logp = self._forward(x)
        return emb[0], compose_pair_scores(logp["crop"], logp["disease"], self.pairs)[0]

    def train_bank(self, train) -> np.ndarray:
        """Backbone features of every training image, computed once and saved in the run folder."""
        from torch.utils.data import DataLoader

        from dpd.cnn import ImageLabels
        from dpd.resize_cache import ensure_resized

        path = self.run["dir"] / "train_features.npy"
        if path.exists():
            bank = np.load(path)
            if len(bank) == len(train):
                return bank
        image_dir = ensure_resized(train["file"], self.run["cfg"]["input"]["resize_cache"])
        loader = DataLoader(ImageLabels(train, image_dir, self.transform), batch_size=64, num_workers=0)
        bank = np.concatenate([self._forward(x.to(device()))[0] for x, _, _ in loader])
        np.save(path, bank)
        return bank


def load_run_model(run: dict, df):
    """A model object with .predict(path or file-like), .train_bank(train_df), .pairs, .gamma, .logit_scale, .score_help."""
    pairs = pair_table(df)
    if run["method"] in ("clip_zeroshot", "clip_prompt_learning"):
        return ClipRunModel(run, pairs)
    if run["method"] in ("cnn_independent", "cnn_multitask"):
        return CnnRunModel(run, pairs)
    raise ValueError(f"unsupported method {run['method']!r}")
