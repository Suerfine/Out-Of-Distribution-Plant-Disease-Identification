"""DPD-Small metadata and image loading. Plain pandas/PIL, so models, scripts and the Streamlit app can all use it."""

import os
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("DPD_ROOT", PROJECT_ROOT / "DPD_Small"))
IMAGE_DIR = ROOT / "DPD_Small_dataset"
SPLIT_FILES = {
    "train": "DPD_Small_train.csv",
    "test_seen": "DPD_Small_test_seen.csv",
    "test_unseen": "DPD_Small_test_unseen.csv",
}
SPLITS = list(SPLIT_FILES)
# The CSVs have no header row.
COLUMNS = ["file", "crop_id", "disease_id", "pair_id", "crop", "disease", "pair"]


def read_metadata() -> pd.DataFrame:
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


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pair id (0..332), sorted by id: crop/disease ids and names plus `is_unseen`.

    The unseen ids are interleaved with the seen ones, so always look them up here rather than assuming an id range.
    """
    cols = ["pair_id", "crop_id", "disease_id", "crop", "disease", "pair", "is_unseen"]
    table = df[cols].drop_duplicates("pair_id").sort_values("pair_id").reset_index(drop=True)
    assert (table["pair_id"] == range(len(table))).all(), "pair ids are expected to be contiguous from 0"
    return table


def load_image(file: str) -> Image.Image:
    with Image.open(IMAGE_DIR / file) as img:
        return img.convert("RGB")
