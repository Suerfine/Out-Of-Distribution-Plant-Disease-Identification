"""Auto-pull dataset/ and runs/ from a Hugging Face Hub repository for Streamlit deployment.

Expected layout inside the Hugging Face repo (Model or Dataset repo):
    dataset/
        DPD_Small_train.csv
        DPD_Small_test_seen.csv
        DPD_Small_test_unseen.csv
        resized_256.zip              # any .zip of images (or a loose image subfolder)
    runs/
        <family>/<run_name>/
            config.json
            metrics.json
            scores.npz
            predictions_test_seen.csv
            predictions_test_unseen.csv
            confusion_test_seen.csv
            <model weights .pt / .npy>
            train_features.npy       # optional precomputed retrieval bank

Configuration (read from Streamlit secrets first, then environment variables):
    HF_REPO_ID    Required to enable auto-pull (e.g. "username/dpd-demo"). Unset = no-op (local dev).
    HF_REPO_TYPE  Optional: "model" (default) or "dataset".
    HF_TOKEN      Optional: Hugging Face token for private repos.
    HF_REVISION   Optional: branch, tag, or commit hash.

Can also be run directly from the CLI to test or pre-populate locally:
    python -m dpd.hf_sync --repo-id <user/repo> [--repo-type model] [--all-weights]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

from dpd.data import IMAGE_DIR, PROJECT_ROOT, ROOT, SPLIT_FILES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lightweight files pulled on startup so the Dashboard, Overview, and Test-set browsing work immediately.
RUN_METADATA_PATTERNS = [
    "runs/**/config.json",
    "runs/**/metrics.json",
    "runs/**/scores.npz",
    "runs/**/predictions_*.csv",
    "runs/**/confusion_*.csv",
    "runs/**/*_text_embeddings.npy",
]


def _get_setting(key: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit secrets (when running inside Streamlit), falling back to env vars."""
    try:
        import streamlit as st

        if key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    val = os.environ.get(key, "").strip()
    return val or default


def hf_config() -> dict[str, str | None]:
    return {
        "repo_id": _get_setting("HF_REPO_ID"),
        "repo_type": _get_setting("HF_REPO_TYPE", "model"),
        "token": _get_setting("HF_TOKEN"),
        "revision": _get_setting("HF_REVISION"),
    }


def is_hf_configured() -> bool:
    return bool(hf_config()["repo_id"])


def _dataset_ready() -> bool:
    csvs_ok = all((ROOT / name).exists() for name in SPLIT_FILES.values())
    images_ok = IMAGE_DIR.exists() and any(IMAGE_DIR.iterdir())
    return csvs_ok and images_ok


def _extract_dataset_zip(zip_path: Path, root_dir: Path, image_dir: Path) -> int:
    """Extract split CSVs into `root_dir` and image files flat into `image_dir`, ignoring nested folder prefixes."""
    root_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_names = set(SPLIT_FILES.values())
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or name.startswith(".") or "__MACOSX" in info.filename:
                continue
            if name in csv_names:
                target = root_dir / name
            elif Path(name).suffix.lower() in IMAGE_EXTS:
                target = image_dir / name
            else:
                continue
            if not target.exists():
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            count += 1
    return count


def sync_dataset(cfg: dict[str, str | None] | None = None) -> bool:
    """Pull `dataset/` from Hugging Face into `dpd.data.ROOT` if not already present. Returns True if downloaded."""
    cfg = cfg or hf_config()
    if not cfg["repo_id"] or _dataset_ready():
        return False

    from huggingface_hub import snapshot_download

    stage_dir = PROJECT_ROOT / ".hf_dataset_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type=cfg["repo_type"],
            token=cfg["token"],
            revision=cfg["revision"],
            allow_patterns=["dataset/**"],
            local_dir=str(stage_dir),
        )
        src_root = stage_dir / "dataset"
        if not src_root.exists():
            raise FileNotFoundError(
                f"Expected a `dataset/` folder in Hugging Face repo `{cfg['repo_id']}`, but none was found."
            )

        ROOT.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Copy metadata CSVs (and any other top-level metadata files like parquet) into ROOT.
        for csv_name in SPLIT_FILES.values():
            src_csv = src_root / csv_name
            if src_csv.exists() and not (ROOT / csv_name).exists():
                shutil.move(str(src_csv), str(ROOT / csv_name))

        # 2. Extract any dataset zip archive(s) (split CSVs -> ROOT, images -> IMAGE_DIR).
        for zip_file in sorted(src_root.rglob("*.zip")):
            _extract_dataset_zip(zip_file, ROOT, IMAGE_DIR)
            zip_file.unlink(missing_ok=True)

        # 3. Also support loose images uploaded under dataset/DPD_Small_dataset/ or dataset/images/ or similar.
        for img_path in src_root.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                target = IMAGE_DIR / img_path.name
                if not target.exists():
                    shutil.move(str(img_path), str(target))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return True


def sync_runs_metadata(cfg: dict[str, str | None] | None = None, include_weights: bool = False) -> bool:
    """Pull `runs/` metadata (or all files when `include_weights=True`) from Hugging Face into `PROJECT_ROOT/runs/`."""
    cfg = cfg or hf_config()
    if not cfg["repo_id"]:
        return False

    from huggingface_hub import snapshot_download

    patterns = ["runs/**"] if include_weights else RUN_METADATA_PATTERNS
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type=cfg["repo_type"],
        token=cfg["token"],
        revision=cfg["revision"],
        allow_patterns=patterns,
        local_dir=str(PROJECT_ROOT),
    )
    return True


def _referenced_member_runs(run_path: Path) -> list[str]:
    """Return any member run paths referenced by an ensemble, supcon_mix, or lora_run feature-gen config."""
    cfg_file = run_path / "config.json"
    if not cfg_file.exists():
        return []
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    method = cfg.get("method", "")
    refs: list[str] = []
    if method == "ensemble":
        refs.extend(m["run"] for m in cfg.get("members", []) if isinstance(m, dict) and "run" in m)
    elif method == "supcon_mix":
        mem = cfg.get("members", {})
        refs.extend(mem.get("convnext", []))
        clip_ref = mem.get("clip")
        if isinstance(clip_ref, str):
            refs.append(clip_ref)
        elif isinstance(clip_ref, list):
            refs.extend(clip_ref)
    elif method == "clip_feature_gen" and cfg.get("features", {}).get("source") == "lora_run":
        if "run" in cfg.get("features", {}):
            refs.append(cfg["features"]["run"])
    return refs


def ensure_run_weights(run_dir: str | Path, cfg: dict[str, str | None] | None = None) -> None:
    """Lazily download all remaining files (weights, train_features.npy, etc.) for a single run when selected."""
    cfg = cfg or hf_config()
    if not cfg["repo_id"]:
        return

    from huggingface_hub import snapshot_download

    run_path = Path(run_dir).resolve()
    try:
        rel = run_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return

    patterns = [f"{rel}/**"]

    # Also pull any member runs referenced by ensembles / mixes / feature-gen.
    for ref in _referenced_member_runs(run_path):
        ref_rel = Path(ref).as_posix().lstrip("./")
        patterns.append(f"{ref_rel}/**")

    # Frozen CLIP runs may reuse cached training embeddings under runs/cache/*.npy.
    cfg_file = run_path / "config.json"
    if cfg_file.exists():
        run_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if run_cfg.get("method") in ("clip_zeroshot", "clip_prompt_learning", "clip_feature_gen"):
            patterns.append("runs/cache/*.npy")

    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type=cfg["repo_type"],
        token=cfg["token"],
        revision=cfg["revision"],
        allow_patterns=patterns,
        local_dir=str(PROJECT_ROOT),
    )


def sync_all(include_weights: bool = False) -> bool:
    """Entry point called by Streamlit on startup: pulls `dataset/` and `runs/` metadata from Hugging Face."""
    cfg = hf_config()
    if not cfg["repo_id"]:
        return False
    sync_dataset(cfg)
    sync_runs_metadata(cfg, include_weights=include_weights)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull dataset/ and runs/ from a Hugging Face Hub repository.")
    parser.add_argument("--repo-id", default=None, help="Hugging Face repo ID (overrides HF_REPO_ID)")
    parser.add_argument("--repo-type", default=None, choices=["model", "dataset"], help="HF repo type (default: model)")
    parser.add_argument("--token", default=None, help="HF access token (overrides HF_TOKEN)")
    parser.add_argument("--revision", default=None, help="HF revision/branch (overrides HF_REVISION)")
    parser.add_argument("--all-weights", action="store_true", help="Also download all model weights immediately")
    args = parser.parse_args()

    cfg = hf_config()
    if args.repo_id:
        cfg["repo_id"] = args.repo_id
    if args.repo_type:
        cfg["repo_type"] = args.repo_type
    if args.token:
        cfg["token"] = args.token
    if args.revision:
        cfg["revision"] = args.revision

    if not cfg["repo_id"]:
        raise SystemExit("No HF_REPO_ID configured. Pass --repo-id <user/repo> or set HF_REPO_ID.")

    print(f"Syncing dataset/ from {cfg['repo_id']} ({cfg['repo_type']})...")
    sync_dataset(cfg)
    print(f"Syncing runs/ ({'all files' if args.all_weights else 'metadata only'}) from {cfg['repo_id']}...")
    sync_runs_metadata(cfg, include_weights=args.all_weights)
    print("Done.")


if __name__ == "__main__":
    main()
