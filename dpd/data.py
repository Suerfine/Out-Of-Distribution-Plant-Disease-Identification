"""DPD-Small metadata and image loading, shared by the Streamlit pages (and later the models)."""

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(os.environ.get("DPD_ROOT", Path(__file__).resolve().parent.parent / "DPD_Small"))
IMAGE_DIR = ROOT / "DPD_Small_dataset"
SPLIT_FILES = {
    "train": "DPD_Small_train.csv",
    "test_seen": "DPD_Small_test_seen.csv",
    "test_unseen": "DPD_Small_test_unseen.csv",
}
SPLITS = list(SPLIT_FILES)
# The CSVs have no header row.
COLUMNS = ["file", "crop_id", "disease_id", "pair_id", "crop", "disease", "pair"]


@st.cache_data
def load_metadata() -> pd.DataFrame:
    """All three splits in one frame, with `split`, `source` (file-name prefix) and `is_unseen` columns."""
    frames = []
    for split, name in SPLIT_FILES.items():
        df = pd.read_csv(ROOT / name, header=None, names=COLUMNS)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["source"] = df["file"].str.split("_").str[0]
    unseen_pairs = set(df.loc[df["split"] == "test_unseen", "pair_id"])
    df["is_unseen"] = df["pair_id"].isin(unseen_pairs)
    return df


@st.cache_data(max_entries=3000)
def thumbnail(file: str, size: int = 256) -> bytes:
    """Square, centre-cropped JPEG bytes for the gallery grid; full images are too large to send by the page-full."""
    with Image.open(IMAGE_DIR / file) as img:
        img = ImageOps.fit(img.convert("RGB"), (size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def load_image(file: str) -> Image.Image:
    with Image.open(IMAGE_DIR / file) as img:
        return img.convert("RGB")


@st.cache_data(persist="disk", show_spinner="Reading image headers…")
def image_sizes(files: tuple[str, ...]) -> pd.DataFrame:
    """Width/height/mode per image. Only reads headers, but still touches every file, so it is cached to disk."""
    rows = []
    for f in files:
        with Image.open(IMAGE_DIR / f) as img:
            rows.append((f, img.width, img.height, img.mode))
    return pd.DataFrame(rows, columns=["file", "width", "height", "mode"])
