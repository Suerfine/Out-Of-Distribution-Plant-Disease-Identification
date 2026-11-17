"""Validation scores of the retrained v10.3 (CLIP with LoRA) for the mixing weight (GPU, ~15 min).
For each of the three validation splits (seeds 42, 43, 44, the splits v10.3 itself used) retrain v10.3 on the split's fit part for
its selected epoch and score the validation images, exactly as train/ensemble.py does for its members (train.ensemble.member_val_scores).
Checks that each regenerated split equals the one saved by the resolution-336 control run (same validation images and fit pairs),
so the two sides of the mix can be compared image by image. Saves queue/v103_val/val_scores_split{k}.npy (cosine scores, n_val x 333)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402
from train.ensemble import member_val_scores  # noqa: E402

def main():
    OUT = REPO / "runs" / "vlm" / "v10.3"
    OUT.mkdir(exist_ok=True)
    run_dir = REPO / "runs" / "vlm" / "v10.3"
    mcfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    metrics = json.load(open(run_dir / "metrics.json"))
    epochs = metrics["selected_epoch"]
    df = read_metadata()
    pairs = pair_table(df)
    train = df[df["split"] == "train"]
    v = mcfg["validation"]
    for k in range(3):
        cache = OUT / f"val_scores_split{k + 1}.npy"
        fit_df, val_seen, val_unseen, _ = make_validation_split(
            train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"], mcfg["seed"] + k,
            tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
        ref = np.load(REPO / "runs" / "contrastive" / "full_res336" / f"val_parts_split{k + 1}.npz")
        assert (val_seen["pair_id"].to_numpy() == ref["y_val_seen"]).all() and (val_unseen["pair_id"].to_numpy() == ref["y_val_unseen"]).all()
        assert (np.sort(fit_df["pair_id"].unique()) == ref["fit_pairs"]).all()
        print(f"split {k + 1}: same validation images and fit pairs as the 336 control run", flush=True)
        if cache.exists():
            continue
        val = member_val_scores(run_dir, mcfg, metrics, fit_df, pd.concat([val_seen, val_unseen]), epochs, pairs)
        np.save(cache, val)
        print(f"split {k + 1}: saved {val.shape}", flush=True)


if __name__ == "__main__":
    main()
