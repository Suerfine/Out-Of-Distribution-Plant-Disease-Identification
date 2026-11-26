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
SUPPORTED_METHODS = ("clip_zeroshot", "clip_prompt_learning", "clip_prompt_lora", "clip_feature_gen", "cnn_independent", "cnn_multitask", "multi_task_cnn", "pretext_multitask_cnn", "ssl_multitask_cnn", "cnn_crop_norm", "ensemble", "cnn_supcon", "supcon_mix", "router")


def list_runs(methods=SUPPORTED_METHODS) -> list[dict]:
    """Finished runs (config.json + metrics.json present) under runs/<family>/<name>/ whose method is in `methods`.

    Families are runs/vlm (versions vN, newest first), runs/baseline, runs/multitask and runs/contrastive.
    """
    runs = []
    for d in RUNS_DIR.glob("*/*"):
        if d.parent.name == "cache" or not ((d / "config.json").exists() and (d / "metrics.json").exists()):
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("method") not in methods:
            continue
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        if cfg["method"] == "cnn_supcon" and not supcon_loadable(d, cfg, metrics):
            continue
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
    for k, v in state.get("context", {}).items():  # CoOp context (prompts.learned_context)
        model.ctx[k].data.copy_(v)
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
    supports_gradcam = False
    gradcam_reason = (
        "Grad-CAM is only available for convolutional CNN models. VLM / CLIP zero-shot models "
        "compute multimodal text-image alignment and do not have convolutional classification heads."
    )

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
        path = self.run["dir"] / "train_features.npy"
        if path.exists():
            bank = np.load(path)
            if len(bank) == len(train):
                return bank
        return train_embeddings(self.clip, self.preprocess, self.run["cfg"]["model"], train["file"])

    def explain(self, src, pred_crop_id: int, pred_disease_id: int):
        return {"supported": False, "reason": self.gradcam_reason, "cams": {}}


class ClipLoraRunModel(ClipRunModel):
    """A prompt-learning run whose image encoder carries trained LoRA adapters (train/train_lora.py).

    Images go through the same path as in training: the resize cache's downscale + JPEG round trip, then CLIP's
    resize/centre crop. Retrieval features come from the adapted encoder, so they are computed per run.
    """

    def __init__(self, run: dict, pairs):
        from dpd.clip_utils import clip_transforms, image_norm_stats
        from dpd.lora import load_lora, set_lora_scale

        super().__init__(run, pairs)
        cfg = run["cfg"]
        lcfg = cfg["image_encoder"]["lora"]
        state = torch.load(run["dir"] / "lora_image_encoder.pt", map_location="cpu")
        load_lora(self.clip.visual, state, lcfg["last_n_blocks"], lcfg["rank"], lcfg["alpha"], lcfg["targets"])
        self.clip.to(device()).eval()
        self.transform = clip_transforms(cfg["input"], cfg["augmentation"], image_norm_stats(cfg["model"], self.preprocess))[1]
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

    @property
    def supports_gradcam(self) -> bool:
        bb = self.run["cfg"].get("model", {}).get("backbone", "")
        if "vit" in bb.lower():
            return False
        return True

    @property
    def gradcam_reason(self) -> str | None:
        if not self.supports_gradcam:
            return "Vision Transformer (ViT) does not use 2D convolutional feature maps. Grad-CAM requires a convolutional backbone."
        return None

    def explain(self, src, pred_crop_id: int, pred_disease_id: int):
        if not self.supports_gradcam:
            return {"supported": False, "reason": self.gradcam_reason, "cams": {}}
        from dpd.gradcam import explain_cnn

        method = self.run["cfg"]["method"]
        if "multitask_net" in self.nets:
            net = self.nets["multitask_net"]
            return explain_cnn(
                net,
                self.transform,
                self.run["cfg"]["input"]["resize_cache"],
                src,
                pred_crop_id,
                pred_disease_id,
                is_multi_task=True,
            )
        elif "crop_net" in self.nets and "disease_net" in self.nets:
            return explain_cnn(
                self.nets["crop_net"],
                self.transform,
                self.run["cfg"]["input"]["resize_cache"],
                src,
                pred_crop_id,
                pred_disease_id,
                is_multi_task=False,
                crop_net=self.nets["crop_net"],
                disease_net=self.nets["disease_net"],
            )
        return {
            "supported": False,
            "reason": f"Grad-CAM is not implemented for method {method!r}.",
            "cams": {},
        }


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


def supcon_loadable(d: Path, cfg: dict, metrics: dict) -> bool:
    """Contrastive runs (train/train_supcon.py) the app can score: full runs that saved their network and test scores and
    whose pair score is the plain sum of the two cross-entropy heads' log-probabilities (scoring.mode "heads", no learned
    seen/unseen detector, no extra scoring terms). Screening runs and the other scoring variants are left out of the list."""
    from dpd.hf_sync import is_hf_configured

    has_weights = (d / "supcon_net.pt").exists() or is_hf_configured()
    return (has_weights and (d / "scores.npz").exists() and cfg.get("scoring", {}).get("mode") == "heads"
            and cfg["model"].get("ce_dropout") is not None and not cfg.get("loss", {}).get("new")
            and (metrics.get("val") or {}).get("scale") is None)


class SupconRunModel:
    """A contrastive-family run (train/train_supcon.py, method cnn_supcon) in "heads" scoring: one ConvNeXt backbone with a
    crop head and a disease head; score = log P(crop) + log P(disease). Retrieval uses the L2-normalised backbone features."""

    logit_scale = 1.0  # scores are already log-probabilities
    score_help = "log P(crop) + log P(disease) from the two cross-entropy heads, after calibration"

    def __init__(self, run: dict, pairs):
        from dpd.cnn import build_transforms
        from dpd.supcon import ContrastiveNet

        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        cfg, m = run["cfg"], run["cfg"]["model"]
        net = ContrastiveNet(m["backbone"], False, m["proj_dim"], m.get("proj_hidden"), m.get("ce_dropout"),
                             m.get("drop_path_rate", 0.0), m.get("cosine_scale"), m.get("head_hidden"))
        net.load_state_dict(torch.load(run["dir"] / "supcon_net.pt", map_location="cpu"))
        net.logit_tau = (cfg["loss"].get("logitnorm") or {}).get("tau")
        self.flips = bool(cfg["scoring"].get("tta"))  # the run's own flip averaging, as at evaluation
        self.net = net.to(device()).eval()
        self.transform = build_transforms(cfg["input"], cfg["augmentation"])[1]
        self.crop_id, self.disease_id = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()

    @torch.no_grad()
    def _forward(self, x):
        """L2-normalised backbone features and the log-probabilities of the crop and disease heads (as dpd.supcon.embed)."""
        x = x.to(device())
        with torch.autocast(device(), enabled=device() == "cuda"):
            f = self.net.backbone(x)
            logits = {h: v.float() for h, v in self.net.classify(f).items()}
            if self.flips:
                for flip in ((-1,), (-2,)):
                    lf = self.net.classify(self.net.backbone(x.flip(flip)))
                    logits = {h: logits[h] + lf[h].float() for h in logits}
                logits = {h: v / 3 for h, v in logits.items()}
        emb = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
        return emb, {h: torch.log_softmax(v.float(), dim=-1).cpu().numpy() for h, v in logits.items()}

    def head_logprobs(self, src):
        """(retrieval embedding, {"crop": log-probs, "disease": log-probs}) for one image path or file-like object."""
        from dpd.resize_cache import standardise

        # Same downscale + JPEG round trip as the training/evaluation cache, so scores match the evaluated model.
        emb, logp = self._forward(self.transform(standardise(src, self.run["cfg"]["input"]["resize_cache"])).unsqueeze(0))
        return emb[0], {h: v[0] for h, v in logp.items()}

    def predict(self, src):
        """(retrieval embedding, uncalibrated score per pair) for an image path or file-like object."""
        emb, logp = self.head_logprobs(src)
        return emb, logp["crop"][self.crop_id] + logp["disease"][self.disease_id]

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
        bank = np.concatenate([self._forward(x)[0] for x, _, _ in loader])
        np.save(path, bank)
        return bank


class SupconMixRunModel:
    """A mix of contrastive-family ConvNeXt runs with one or more CLIP + LoRA runs (method supcon_mix; built by
    train/supcon_eval/mix_v103.py or mix_v103_two_seeds.py, and mix_v103_report.py). Several CLIP runs are averaged in log-probabilities. The ConvNeXt members' head log-probabilities are averaged and
    composed with a crop weight; both sides are put on log-probabilities over the 333 pairs (dpd/ensemble.py) and mixed as
    clip_weight * CLIP + (1 - clip_weight) * ConvNeXt. Weights and gamma were chosen on validation. Retrieval uses the first
    ConvNeXt member's backbone features."""

    logit_scale = 1.0  # scores are log-probabilities
    score_help = "Weighted sum of the CLIP member's and the averaged ConvNeXt members' log-probabilities, after calibration"

    def __init__(self, run: dict, pairs):
        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        mem, ens = run["cfg"]["members"], run["cfg"]["ensemble"]
        self.clip_weight, self.crop_weight = float(ens["clip_weight"]), float(ens["crop_weight"])
        self.cnns = [SupconRunModel(member_run(p), pairs) for p in mem["convnext"]]
        clip_paths = [mem["clip"]] if isinstance(mem["clip"], str) else list(mem["clip"])  # one CLIP run or several (averaged)
        self.clip_runs = [member_run(p) for p in clip_paths]
        self.clips = [build_run_model(r, pairs) for r in self.clip_runs]
        self.crop_id, self.disease_id = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()

    def predict(self, src):
        from dpd.ensemble import to_logprob

        data = Path(src).read_bytes() if isinstance(src, (str, Path)) else src.read()
        outs = [m.head_logprobs(io.BytesIO(data)) for m in self.cnns]
        crop = np.mean([o[1]["crop"] for o in outs], axis=0)
        disease = np.mean([o[1]["disease"] for o in outs], axis=0)
        cnn = to_logprob(self.crop_weight * crop[self.crop_id] + disease[self.disease_id], "cnn_independent")
        clip = np.mean([to_logprob(m.predict(io.BytesIO(data))[1], r["method"]) for m, r in zip(self.clips, self.clip_runs)], axis=0)
        return outs[0][0], self.clip_weight * clip + (1 - self.clip_weight) * cnn

    def train_bank(self, train) -> np.ndarray:
        return self.cnns[0].train_bank(train)


def member_run(path: str) -> dict:
    """A run dict (as list_runs builds) for an ensemble member given its folder relative to the project root."""
    d = PROJECT_ROOT / path
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return {"dir": d, "family": d.parent.name, "version": cfg["version"], "method": cfg["method"], "cfg": cfg,
            "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8"))}


class EnsembleRunModel:
    """An ensemble run (train/ensemble.py): sum_k w_k * logprob(member k), with the weights and gamma chosen on
    validation (two members: w and 1 - w). Loads every member; retrieval uses the first member's image features."""

    logit_scale = 1.0  # scores are log-probabilities
    score_help = "Weighted sum of the members' log-probabilities, after calibration"
    supports_gradcam = False
    gradcam_reason = (
        "Grad-CAM is not supported for multi-model ensembles. Select an individual CNN model "
        "(such as Multi-Task CNN or Independent CNN) to view Grad-CAM heatmaps."
    )

    def __init__(self, run: dict, pairs):
        self.run, self.pairs = run, pairs
        self.gamma = float(run["metrics"].get("gamma", 0.0))
        self.weight = float(run["metrics"]["weight"])
        # Runs from before three-member ensembles store only the first member's weight.
        self.weights = [float(w) for w in run["metrics"].get("weights", [self.weight, 1 - self.weight])]
        self.members = [member_run(m["run"]) for m in run["cfg"]["members"]]
        assert len(self.weights) == len(self.members), "metrics.json weights do not match the config's members"
        self.models = [build_run_model(m, pairs) for m in self.members]

    def predict(self, src):
        from dpd.ensemble import to_logprob

        data = Path(src).read_bytes() if isinstance(src, (str, Path)) else src.read()
        outs = [m.predict(io.BytesIO(data)) for m in self.models]
        mixed = sum(w * to_logprob(s, m["method"]) for w, (_, s), m in zip(self.weights, outs, self.members))
        return outs[0][0], mixed

    def train_bank(self, train) -> np.ndarray:
        return self.models[0].train_bank(train)

    def explain(self, src, pred_crop_id: int, pred_disease_id: int):
        return {"supported": False, "reason": self.gradcam_reason, "cams": {}}


class RouterRunModel:
    """A gated-router run (train/route.py, v15): both members score the image, the saved gate (gate.joblib: scaler +
    logistic regression on the members' seen/unseen margins) plus the validation bias decides seen vs unseen, and the
    chosen branch's expert (its own mix of the members) ranks the pairs. Scores are the routed expert's
    log-probabilities with the other branch pushed down by 1e6 (hard routing), so they are ranked with gamma 0.
    Retrieval uses the first member's image features."""

    logit_scale = 1.0  # scores are log-probabilities
    score_help = "Log-probability from the expert the seen/unseen gate routed the image to"
    supports_gradcam = False
    gradcam_reason = "Grad-CAM is not supported for the gated router."

    def __init__(self, run: dict, pairs):
        import joblib

        gate_file = run["dir"] / "gate.joblib"
        if not gate_file.exists():
            raise FileNotFoundError(f"{gate_file} is missing: re-run train/route.py to save the fitted gate")
        self.run, self.pairs, self.gamma = run, pairs, 0.0
        self.gate = joblib.load(gate_file)
        self.val = run["metrics"]["val"]
        self.members = [member_run(m["run"]) for m in run["cfg"]["members"]]
        assert len(self.members) == 2, "the router routes between exactly two members"
        self.models = [build_run_model(m, pairs) for m in self.members]
        self.unseen = pairs["is_unseen"].to_numpy()
        self._gate_logits = {}  # image bytes -> gate logit (bias added), for route_info()

    def _route(self, data: bytes):
        from dpd.ensemble import to_logprob
        from train.route import branch_logp, gate_features, log_softmax, routed_scores

        outs = [m.predict(io.BytesIO(data)) for m in self.models]
        a, b = (to_logprob(s, m["method"])[None] for (_, s), m in zip(outs, self.members))
        mix = lambda w: w * a + (1 - w) * b
        v, seen, unseen = self.val, ~self.unseen, self.unseen
        x = gate_features((a, b, log_softmax(mix(v["ensemble_weight"]))), seen, unseen, v["gate_features"])[0]
        z = self.gate.decision_function(x) + v["bias"]
        scores = routed_scores(z, branch_logp(mix(v["seen_expert_weight"]), seen),
                               branch_logp(mix(v["unseen_expert_weight"]), unseen), seen, v["mode"])[0]
        return outs[0][0], scores, float(z[0])

    def predict(self, src):
        data = Path(src).read_bytes() if isinstance(src, (str, Path)) else src.read()
        emb, scores, z = self._route(data)
        self._gate_logits[hash(data)] = z
        return emb, scores

    def route_info(self, data: bytes) -> dict:
        """The gate's decision for an image: logit (bias added), P(unseen) and the branch it was sent to."""
        z = self._gate_logits.get(hash(data))
        if z is None:
            z = self._route(data)[2]
            self._gate_logits[hash(data)] = z
        return {"logit": z, "p_unseen": float(1 / (1 + np.exp(-z))), "branch": "unseen" if z > 0 else "seen"}

    def train_bank(self, train) -> np.ndarray:
        return self.models[0].train_bank(train)

    def explain(self, src, pred_crop_id: int, pred_disease_id: int):
        return {"supported": False, "reason": self.gradcam_reason, "cams": {}}


def model_class(method: str):
    """ClipRunModel or CnnRunModel; class attributes (logit_scale, score_help) are usable without loading weights."""
    if method in ("clip_zeroshot", "clip_prompt_learning", "clip_feature_gen"):
        return ClipRunModel
    if method == "clip_prompt_lora":
        return ClipLoraRunModel
    if method in ("cnn_independent", "cnn_multitask", "multi_task_cnn", "pretext_multitask_cnn", "ssl_multitask_cnn"):
        return CnnRunModel
    if method == "cnn_crop_norm":
        return CropNormRunModel
    if method == "ensemble":
        return EnsembleRunModel
    if method == "cnn_supcon":
        return SupconRunModel
    if method == "supcon_mix":
        return SupconMixRunModel
    if method == "router":
        return RouterRunModel
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
