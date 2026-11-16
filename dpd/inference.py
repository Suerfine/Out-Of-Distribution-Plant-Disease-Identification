"""Load a finished run from runs/<family>/<name> and score new images with it. No Streamlit imports (the app wraps these in caches)."""

import io
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dpd.clip_utils import build_prompts, encode_images, encode_prompts, load_clip
from dpd.data import PROJECT_ROOT, pair_table
from dpd.evaluate import ranked_indices

RUNS_DIR = PROJECT_ROOT / "runs"
SUPPORTED_METHODS = ("clip_zeroshot", "clip_prompt_learning", "clip_prompt_lora", "clip_feature_gen", "cnn_independent", "cnn_multitask", "multi_task_cnn", "cnn_crop_norm", "ensemble")


def list_runs(methods=SUPPORTED_METHODS) -> list[dict]:
    """Finished runs (config.json + metrics.json present) under runs/<family>/<name>/ whose method is in `methods`.

    Families are runs/vlm (versions vN, newest first), runs/baseline, and runs/multitask.
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
        m = re.fullmatch(r"v(\d+)(?:\.(\d+))?", r["version"])  # v5, v3.1 (a variant of v3), ...
        return (0, -int(m.group(1)), -int(m.group(2) or 0)) if m else (1, r["version"])

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
        renormalize = cfg["prompts"].get("class_score", "normalized_mean") != "mean_of_scores"
        return encode_prompts(clip_model, tokenizer, build_prompts(pairs, cfg["names"], cfg["prompts"]), device(), renormalize)
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
    return ranked_indices(scores), scores, probs


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


class ClipLoraRunModel(ClipRunModel):
    """A prompt-learning run whose image encoder carries trained LoRA adapters (train/train_lora.py).

    Images go through the same path as in training: the resize cache's downscale + JPEG round trip, then CLIP's
    resize/centre crop. Retrieval features come from the adapted encoder, so they are computed per run.
    """

    def __init__(self, run: dict, pairs):
        from dpd.clip_utils import clip_transforms
        from dpd.lora import load_lora, set_lora_scale

        super().__init__(run, pairs)
        cfg = run["cfg"]
        lcfg = cfg["image_encoder"]["lora"]
        state = torch.load(run["dir"] / "lora_image_encoder.pt", map_location="cpu")
        load_lora(self.clip.visual, state, lcfg["last_n_blocks"], lcfg["rank"], lcfg["alpha"], lcfg["targets"])
        self.clip.to(device()).eval()
        self.transform = clip_transforms(cfg["input"], cfg["augmentation"])[1]
        # Patch-level disease scoring (v7+): needs the run's disease text embeddings, saved by train_lora.py.
        self.patch = cfg["model"].get("patch_disease")
        if self.patch:
            self.disease_emb = torch.as_tensor(np.load(run["dir"] / "disease_text_embeddings.npy"), device=device())
            self.pair_disease = pairs["disease_id"].to_numpy()
        # Post-hoc knobs (v9+, evaluation.posthoc): LoRA scale and crop / disease fusion weights chosen on validation.
        m = run["metrics"]
        set_lora_scale(self.clip.visual, m.get("lora_scale", 1.0))
        self.fusion = (m.get("fusion_crop_weight", 0.0), m.get("fusion_disease_weight", 0.0))
        # Troika-style primitive branches (model.primitive_branches): fusion scores use the branch features.
        self.branches = None
        if cfg["model"].get("primitive_branches"):
            from dpd.primitive_branches import build_branches

            self.branches = build_branches(self.clip.visual.output_dim)
            self.branches.load_state_dict(torch.load(run["dir"] / "primitive_branches.pt", map_location="cpu"))
            self.branches.to(device()).eval()
        if any(self.fusion):
            self.crop_text = np.load(run["dir"] / "crop_text_embeddings.npy") if self.fusion[0] else None
            self.disease_text = np.load(run["dir"] / "disease_text_embeddings.npy") if self.fusion[1] else None

    @torch.no_grad()
    def _encode(self, x, with_patch=False):
        """Same as evaluation: mean of the normalised embeddings over the run's TTA views (default: none); with
        `with_patch`, also the mean per-disease patch scores (None for runs without patch scoring)."""
        from dpd.clip_utils import TTA_VIEWS
        from dpd.patch_disease import image_features

        x = x.to(device())
        dt = self.disease_emb if (with_patch and self.patch) else None
        views = []
        for v in self.run["cfg"]["evaluation"].get("tta", ["identity"]):
            with torch.autocast(device(), enabled=device() == "cuda"):
                views.append(image_features(self.clip, TTA_VIEWS[v](x), dt, self.patch and self.patch["top_k"]))
        emb = torch.nn.functional.normalize(torch.stack([c for c, _ in views]).mean(0), dim=-1).cpu().numpy()
        if not with_patch:
            return emb
        return emb, (torch.stack([p for _, p in views]).mean(0).cpu().numpy() if dt is not None else None)

    def predict(self, src):
        from dpd.patch_disease import combine
        from dpd.resize_cache import standardise

        emb, patch = self._encode(self.transform(standardise(src, self.run["cfg"]["input"]["resize_cache"])).unsqueeze(0), with_patch=True)
        lam = self.patch["lambda"] if self.patch else 0.0
        scores = combine((emb @ self.pair_emb.T), patch, getattr(self, "pair_disease", None), lam)
        if any(self.fusion):
            from dpd.fusion import fuse

            a, b = self.fusion
            from dpd.primitive_branches import branch_features

            crop = branch_features(self.branches, "crop", emb) @ self.crop_text.T if a else None
            dis = branch_features(self.branches, "disease", emb) @ self.disease_text.T if b else None
            scores = fuse(scores, crop, dis, self.pairs["crop_id"].to_numpy(), self.pairs["disease_id"].to_numpy(), a, b)
        return emb[0], scores[0]

    def train_bank(self, train) -> np.ndarray:
        """Adapted-encoder embeddings of every training image, computed once and saved in the run folder."""
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
        bank = np.concatenate([self._encode(x) for x, _, _ in loader])
        np.save(path, bank)
        return bank


class FeatureGenLoraRunModel(ClipLoraRunModel):
    """A feature-generation run (train/train_feature_gen.py) on a LoRA run's features: that run's adapted encoder
    embeds the image (and the training bank, shared with it), this run's classifier weights score it."""

    def __init__(self, run: dict, pairs):
        src = PROJECT_ROOT / run["cfg"]["features"]["run"]
        super().__init__({"dir": src, "cfg": json.loads((src / "config.json").read_text(encoding="utf-8")),
                          "metrics": json.loads((src / "metrics.json").read_text(encoding="utf-8"))}, pairs)
        self.pair_emb = np.load(run["dir"] / "pair_text_embeddings.npy")
        self.gamma = float(run["metrics"].get("gamma", 0.0))


class CnnRunModel:
    """A CNN run: score = log P(crop) + log P(disease) from the saved network(s), optionally weighted and logit-adjusted."""

    logit_scale = 1.0  # scores are already log-probabilities
    score_help = "log P(crop) + log P(disease) (weighted and logit-adjusted if the run chose so), after calibration"

    def __init__(self, run: dict, pairs):
        from dpd.cnn import HEAD_SIZES, NETWORKS, MultiHeadNet

        self._load_scoring(run, pairs)
        cfg = run["cfg"]
        self.nets = {}
        for name, heads in NETWORKS[cfg["method"]].items():
            net = MultiHeadNet(cfg["model"]["backbone"], False, {h: HEAD_SIZES[h] for h in heads}, cfg["model"]["head_dropout"])
            net.load_state_dict(torch.load(run["dir"] / f"{name}.pt", map_location="cpu"))
            self.nets[name] = net.to(device()).eval()

    def _load_scoring(self, run: dict, pairs):
        """Gamma, crop weight, tau (and the training log prior it needs) from metrics.json, and the test transform."""
        from dpd.cnn import build_transforms

        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        # Scoring chosen on validation (train_cnn "composition"); runs without it used the plain sum (w 0.5, tau 0).
        self.crop_weight = float(run["metrics"].get("crop_weight", 0.5))
        self.tau = float(run["metrics"].get("tau", 0.0))
        if self.tau:
            from dpd.cnn import log_priors
            from dpd.data import read_metadata

            self.log_prior = log_priors(read_metadata().query("split == 'train'"))  # the final fit trains on all train
        else:
            self.log_prior = None
        cfg = run["cfg"]
        self.transform = build_transforms(cfg["input"], cfg["augmentation"])[1]

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
        scores = compose_pair_scores(logp["crop"], logp["disease"], self.pairs, self.crop_weight, self.tau, self.log_prior)
        return emb[0], scores[0]

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


class CropNormRunModel(CnnRunModel):
    """A crop-normalisation run (train/train_crop_norm.py): log P(crop) + log P(disease | image normalised for that crop),
    from the crop CNN (or the shared network's crop head) and the crop-normalised disease network."""

    score_help = "log P(crop) + log P(disease | image normalised for that crop) (weighted if the run chose so), after calibration"

    def __init__(self, run: dict, pairs):
        from dpd.cnn import HEAD_SIZES, MultiHeadNet
        from dpd.crop_norm import CropNormNet

        self._load_scoring(run, pairs)
        cfg = run["cfg"]
        mcfg = cfg["model"]
        self.disease_net = CropNormNet(mcfg["backbone"], False, cfg["crop_norm"], mcfg["head_dropout"])
        self.disease_net.load_state_dict(torch.load(run["dir"] / "disease_net.pt", map_location="cpu"))
        self.disease_net = self.disease_net.to(device()).eval()
        self.crop_net = None
        if not cfg["crop_norm"]["shared_backbone"]:
            self.crop_net = MultiHeadNet(mcfg["backbone"], False, {"crop": HEAD_SIZES["crop"]}, mcfg["head_dropout"])
            self.crop_net.load_state_dict(torch.load(run["dir"] / "crop_net.pt", map_location="cpu"))
            self.crop_net = self.crop_net.to(device()).eval()

    @torch.no_grad()
    def _forward(self, x):
        """Concatenated L2-normalised backbone features, and (crop log-probs, disease logits under every crop)."""
        dn = self.disease_net
        with torch.autocast(device(), enabled=device() == "cuda"):
            f = dn.backbone(x)
            z = dn.proj(f).float()
            feats = [f.float()]
            if self.crop_net is not None:
                fc = self.crop_net.backbone(x)
                feats.insert(0, fc.float())
                crop_logits = self.crop_net.heads["crop"](fc)
            else:
                crop_logits = dn.crop_head(f)
        with torch.autocast(device(), enabled=False):
            dis_logits = dn.disease_logits(dn.normalise(z))
        emb = torch.nn.functional.normalize(torch.cat([torch.nn.functional.normalize(v, dim=-1) for v in feats], dim=-1), dim=-1)
        logp_crop = torch.log_softmax(crop_logits.float(), dim=-1)
        return emb.cpu().numpy(), (logp_crop.cpu().numpy(), dis_logits.cpu().numpy())

    def predict(self, src):
        """(retrieval embedding, uncalibrated score per pair) for an image path or file-like object."""
        from dpd.crop_norm import compose_pair_scores
        from dpd.resize_cache import standardise

        x = self.transform(standardise(src, self.run["cfg"]["input"]["resize_cache"])).unsqueeze(0).to(device())
        emb, (logp_crop, dis_logits) = self._forward(x)
        scores = compose_pair_scores(logp_crop, dis_logits, self.pairs, self.crop_weight, self.tau, self.log_prior)
        return emb[0], scores[0]


def member_run(path: str) -> dict:
    """A run dict (as list_runs builds) for an ensemble member given its folder relative to the project root."""
    d = PROJECT_ROOT / path
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return {"dir": d, "family": d.parent.name, "version": cfg["version"], "method": cfg["method"], "cfg": cfg,
            "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8"))}


class EnsembleRunModel:
    """An ensemble run (train/ensemble.py): w * logprob(member 1) + (1 - w) * logprob(member 2), with the weight and
    gamma chosen on validation. Loads both members; retrieval uses the first member's image features."""

    logit_scale = 1.0  # scores are log-probabilities
    score_help = "Weighted sum of the members' log-probabilities, after calibration"

    def __init__(self, run: dict, pairs):
        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        self.weight = float(run["metrics"]["weight"])
        self.members = [member_run(m["run"]) for m in run["cfg"]["members"]]
        self.models = [build_run_model(m, pairs) for m in self.members]

    def predict(self, src):
        from dpd.ensemble import to_logprob

        data = Path(src).read_bytes() if isinstance(src, (str, Path)) else src.read()
        (emb, a), (_, b) = (m.predict(io.BytesIO(data)) for m in self.models)
        mixed = (self.weight * to_logprob(a, self.members[0]["method"])
                 + (1 - self.weight) * to_logprob(b, self.members[1]["method"]))
        return emb, mixed

    def train_bank(self, train) -> np.ndarray:
        return self.models[0].train_bank(train)


def model_class(method: str):
    """ClipRunModel or CnnRunModel; class attributes (logit_scale, score_help) are usable without loading weights."""
    if method in ("clip_zeroshot", "clip_prompt_learning", "clip_feature_gen"):
        return ClipRunModel
    if method == "clip_prompt_lora":
        return ClipLoraRunModel
    if method in ("cnn_independent", "cnn_multitask", "multi_task_cnn"):
        return CnnRunModel
    if method == "cnn_crop_norm":
        return CropNormRunModel
    if method == "ensemble":
        return EnsembleRunModel
    raise ValueError(f"unsupported method {method!r}")


def build_run_model(run: dict, pairs):
    if run["cfg"].get("implementation") == "independent_cnn_baseline_tiny_refit":
        from independent_cnn_baseline.tiny_refit import TinyRefitRunModel

        return TinyRefitRunModel(run, pairs)
    if run["cfg"].get("implementation") == "independent_cnn_baseline":
        from independent_cnn_baseline.inference import IndependentCnnRunModel

        return IndependentCnnRunModel(run, pairs)
    if run["method"] == "clip_feature_gen" and run["cfg"].get("features", {}).get("source") == "lora_run":
        return FeatureGenLoraRunModel(run, pairs)
    return model_class(run["method"])(run, pairs)


def load_run_model(run: dict, df):
    """A model object with .predict(path or file-like), .train_bank(train_df), .pairs, .gamma, .logit_scale, .score_help."""
    return build_run_model(run, pair_table(df))
