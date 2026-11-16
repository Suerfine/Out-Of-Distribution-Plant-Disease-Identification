"""Streamlit-only helpers: cached data access and the thumbnail grid. Keep Streamlit imports out of dpd/data.py."""

import io

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from dpd.data import IMAGE_DIR, pair_table, read_metadata
from dpd.evaluate import calibrate, ranked_indices
from dpd.inference import list_runs, load_run_model, model_class

SPLIT_LABELS = {"train": "Train", "test_seen": "Test seen", "test_unseen": "Test unseen"}

# How a top-1 prediction relates to the truth. A pair is one (crop, disease) combination, so a wrong pair always has
# a wrong crop, a wrong disease, or both.
OUTCOMES = ["Correct", "Wrong crop", "Wrong disease", "Both wrong"]

KIND = {
    "clip_zeroshot": "zero-shot CLIP",
    "clip_prompt_learning": "trained prompts",
    "clip_prompt_lora": "trained prompts + LoRA",
    "clip_feature_gen": "generated unseen features",
    "cnn_independent": "baseline: independent CNNs",
    "cnn_multitask": "baseline: multi-task CNN",
    "multi_task_cnn": "multi-task CNN",
    "ssl_multitask_cnn": "SSL multi-task CNN",
    "cnn_disentangle": "disentangled CNN",
    "cnn_crop_norm": "crop-normalised CNN",
    "cnn_compositional": "compositional CNN",
    "ensemble": "ensemble",
    "cnn_supcon": "contrastive-family CNN (crop + disease heads)",
    "supcon_mix": "mix: ConvNeXt ensemble + CLIP",
}


@st.cache_data
def load_metadata() -> pd.DataFrame:
    return read_metadata()


@st.cache_data(max_entries=3000)
def thumbnail(file: str, size: int = 256) -> bytes:
    """Square, centre-cropped JPEG bytes for the gallery grid; full images are too large to send by the page-full."""
    with Image.open(IMAGE_DIR / file) as img:
        img = ImageOps.fit(img.convert("RGB"), (size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@st.cache_data(max_entries=500)
def thumbnail_bytes(data: bytes, size: int = 256) -> bytes:
    """Like thumbnail(), for an uploaded image's bytes."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.fit(img.convert("RGB"), (size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def outcome(crop_ok, disease_ok) -> np.ndarray:
    """OUTCOMES label per prediction, from boolean arrays of crop / disease correctness."""
    crop_ok, disease_ok = np.asarray(crop_ok), np.asarray(disease_ok)
    return np.select(
        [crop_ok & disease_ok, ~crop_ok & disease_ok, crop_ok & ~disease_ok], OUTCOMES[:3], default=OUTCOMES[3]
    )


@st.cache_data(max_entries=16, show_spinner="Reading saved test predictions…")
def test_predictions(run_dir: str, method: str, gamma: float, k: int = 5) -> pd.DataFrame:
    """One row per test image (both splits) from the run's saved scores.npz, ranked with the run's calibration.

    Columns: file, split, true / pred pair ids, top-k pair ids (`topk`, a list), top-1 confidence, crop_ok,
    disease_ok, outcome. Matches metrics.json exactly, and needs no model or GPU.
    """
    d = Path(run_dir)
    z = np.load(d / "scores.npz")
    pairs = pair_table(load_metadata())
    unseen_mask = pairs["is_unseen"].to_numpy()
    crop_ids, disease_ids = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    logit_scale = model_class(method).logit_scale
    frames = []
    for split in ("test_seen", "test_unseen"):
        s = calibrate(z[f"{split}_scores"], unseen_mask, gamma)
        labels = z[f"{split}_labels"]
        order = ranked_indices(s, k)
        pred = order[:, 0]
        e = np.exp(logit_scale * (s - s.max(axis=1, keepdims=True)))
        conf = e[np.arange(len(s)), pred] / e.sum(axis=1)
        crop_ok, disease_ok = crop_ids[pred] == crop_ids[labels], disease_ids[pred] == disease_ids[labels]
        files = pd.read_csv(d / f"predictions_{split}.csv", usecols=["file"])["file"]
        frames.append(pd.DataFrame({
            "file": files, "split": split, "true_id": labels, "pred_id": pred, "topk": list(order),
            "confidence": conf, "crop_ok": crop_ok, "disease_ok": disease_ok, "outcome": outcome(crop_ok, disease_ok),
        }))
    return pd.concat(frames, ignore_index=True)


def backbone(run: dict) -> str:
    if run["method"] == "ensemble":
        from dpd.inference import member_run

        return " + ".join(f"{m['version']} ({backbone(m)})" for m in map(member_run, (x["run"] for x in run["cfg"]["members"])))
    if run["method"] == "supcon_mix":
        from dpd.inference import member_run

        mem = run["cfg"]["members"]
        clips = [member_run(p) for p in ([mem["clip"]] if isinstance(mem["clip"], str) else mem["clip"])]
        return (f"{len(mem['convnext'])} × {backbone(member_run(mem['convnext'][0]))} + "
                f"{' + '.join(c['version'] for c in clips)} ({backbone(clips[0])})")
    if run["method"] == "cnn_supcon":
        name = {"timm:convnext_base.fb_in22k_ft_in1k": "ConvNeXt-Base"}.get(run["cfg"]["model"]["backbone"], run["cfg"]["model"]["backbone"])
        return f"{name} ({run['cfg']['input']['crop_size']} px)"
    mcfg = run["cfg"].get("model", {})
    if "backbone" in mcfg:
        bb = mcfg["backbone"]
        n = "2 × " if run["method"] == "cnn_independent" else ""
        return n + {"resnet50": "ResNet-50"}.get(bb, bb)
    if "arch" in mcfg:
        return "BioCLIP" if "bioclip" in mcfg["arch"] else "OpenAI CLIP"
    return "Unknown"


def run_label(run: dict) -> str:
    name = f"{run['version']} {run['cfg']['title']}" if run["cfg"].get("title") else run["version"]
    method_name = KIND.get(run["method"], run["method"].replace("_", " "))
    return f"{name} · {method_name} · test HM {run['metrics']['harmonic_mean_top1']:.1%}"


def best_run_index(runs: list[dict]) -> int:
    """The best proposed (VLM) model by test HM Top-1; the default model selection on every page."""
    candidates = [i for i, r in enumerate(runs) if r["family"] == "vlm"] or list(range(len(runs)))
    return max(candidates, key=lambda i: runs[i]["metrics"]["harmonic_mean_top1"])


@st.cache_resource(max_entries=2, show_spinner="Loading model…")
def get_model(run_dir: str):
    """A ClipRunModel or CnnRunModel (dpd/inference.py). st.cache_resource keys on this function's source, so edit
    this docstring if the returned object's interface changes, or a running server keeps serving stale objects."""
    run = next(r for r in list_runs() if str(r["dir"]) == run_dir)
    return load_run_model(run, load_metadata())


@st.cache_resource(max_entries=2, show_spinner="Loading training-image embeddings…")
def get_train_bank(run_dir: str):
    train = load_metadata().query("split == 'train'").reset_index(drop=True)
    return train, get_model(run_dir).train_bank(train)


@st.cache_data(persist="disk", show_spinner="Reading image headers…")
def image_sizes(files: tuple[str, ...]) -> pd.DataFrame:
    """Width/height/mode per image. Only reads headers, but still touches every file, so it is cached to disk."""
    rows = []
    for f in files:
        with Image.open(IMAGE_DIR / f) as img:
            rows.append((f, img.width, img.height, img.mode))
    return pd.DataFrame(rows, columns=["file", "width", "height", "mode"])


def image_grid(df: pd.DataFrame, n_cols: int = 6, caption_fields: tuple[str, ...] = ("pair", "split", "source")) -> None:
    """Render rows of `df` as a fixed-width grid of thumbnails with short captions."""
    if df.empty:
        st.caption("No images.")
        return
    rows = list(df.itertuples(index=False))
    for start in range(0, len(rows), n_cols):
        cols = st.columns(n_cols)
        for col, r in zip(cols, rows[start : start + n_cols]):
            parts = []
            for f in caption_fields:
                v = getattr(r, f)
                parts.append(SPLIT_LABELS.get(v, v) if f == "split" else str(v))
            with col:
                st.image(
                    thumbnail(r.file),
                    caption=" · ".join(parts),
                    width="stretch",
                    alt=f"{r.crop} leaf, {r.disease.replace('_', ' ')}",
                )
