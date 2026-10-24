"""Validation split carved out of train, mimicking the seen/unseen test setup.

- pseudo-unseen: whole seen pairs held out of training (the model never sees them), chosen so their crop and
  disease still occur in other training pairs, exactly like the real unseen pairs.
- val-seen: a random fraction of images from the remaining pairs (each pair keeps at least one training image).
Used to pick the epoch count and the calibration gamma without touching either test set.
"""

import numpy as np
import pandas as pd


def make_validation_split(train: pd.DataFrame, n_holdout_pairs: int, min_holdout_images: int, val_fraction: float, seed: int,
                          exclude_diseases: tuple[str, ...] = (), max_images_per_holdout_pair: int | None = None):
    rng = np.random.default_rng(seed)
    counts = train.loc[~train["disease"].isin(exclude_diseases), "pair_id"].value_counts()
    candidates = rng.permutation(counts[counts >= min_holdout_images].index.to_numpy())

    holdout = []
    for p in candidates:
        if len(holdout) == n_holdout_pairs:
            break
        keep = train[~train["pair_id"].isin(holdout + [p])]
        row = train[train["pair_id"] == p].iloc[0]
        # The crop and the disease must both survive in training, otherwise the pair is not compositional.
        if row["crop_id"] in set(keep["crop_id"]) and row["disease_id"] in set(keep["disease_id"]):
            holdout.append(int(p))
    if len(holdout) < n_holdout_pairs:
        raise ValueError(f"Only found {len(holdout)} eligible pseudo-unseen pairs")

    val_unseen = train[train["pair_id"].isin(holdout)]
    if max_images_per_holdout_pair:
        # Cap so one big pair (e.g. cassava_mosaic, 544 images) does not dominate; the dropped images are not trained on.
        keep_idx = [
            i
            for _, g in val_unseen.groupby("pair_id")
            for i in rng.choice(g.index.to_numpy(), size=min(len(g), max_images_per_holdout_pair), replace=False)
        ]
        val_unseen = val_unseen.loc[sorted(keep_idx)]
    rest = train[~train["pair_id"].isin(holdout)]
    val_idx = []
    for _, g in rest.groupby("pair_id"):
        n = int(np.floor(len(g) * val_fraction))
        n = min(n, len(g) - 1)  # keep at least one training image per pair
        if n > 0:
            val_idx.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))
    val_seen = rest.loc[val_idx]
    fit = rest.drop(index=val_idx)
    return fit, val_seen, val_unseen, holdout
