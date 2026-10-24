"""Write a run's test results in the same format for every method."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dpd.evaluate import predictions_table, seen_confusion_matrix


def save_test_outputs(out: Path, result: dict, scores: dict, labels: dict, files: dict, pairs: pd.DataFrame, gamma: float):
    """metrics.json, confusion_test_seen.csv, predictions_<split>.csv and raw scores.npz."""
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    seen_confusion_matrix(scores["test_seen"], labels["test_seen"], pairs, gamma).to_csv(out / "confusion_test_seen.csv")
    for split in scores:
        predictions_table(files[split], scores[split], labels[split], pairs, gamma).to_csv(out / f"predictions_{split}.csv", index=False)
    np.savez_compressed(out / "scores.npz", **{f"{s}_scores": scores[s] for s in scores}, **{f"{s}_labels": labels[s] for s in labels})
