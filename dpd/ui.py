"""Streamlit-only helpers: cached data access and the thumbnail grid. Keep Streamlit imports out of dpd/data.py."""

import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from dpd.data import IMAGE_DIR, read_metadata

SPLIT_LABELS = {"train": "Train", "test_seen": "Test seen", "test_unseen": "Test unseen"}


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
