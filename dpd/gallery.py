"""Reusable image-grid rendering for the Streamlit pages."""

import pandas as pd
import streamlit as st

from dpd.data import thumbnail

SPLIT_LABELS = {"train": "Train", "test_seen": "Test seen", "test_unseen": "Test unseen"}


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
