"""Gamma-zero validation objective for the corrected ConvNeXt-Base search."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dpd.cnn import compose_pair_scores
from dpd.evaluate import evaluate

OBJECTIVE_NAME = "validation_composition_top1_harmonic_mean_gamma_zero"


def validation_uncalibrated(crop, disease, frames, pairs):
    """Rank raw composed scores; there is no gamma sweep or tuned score offset."""
    seen = frames["validation_seen"]
    pseudo = frames["validation_unseen"]
    if not len(seen) or not len(pseudo):
        raise ValueError("Both validation partitions are required")
    if pairs.pair_id.tolist() != list(range(len(pairs))):
        raise ValueError("Composition IDs must match score columns")
    labels = pd.concat([seen, pseudo], ignore_index=True).pair_id.to_numpy()
    real_unseen = set(pairs.loc[pairs.is_unseen, "pair_id"])
    if real_unseen.intersection(labels):
        raise ValueError("Official unseen compositions cannot occur in validation")
    pseudo_ids = set(pseudo.pair_id)
    if pseudo_ids.intersection(seen.pair_id):
        raise ValueError("Seen and pseudo-unseen validation compositions overlap")
    arrays = [torch.as_tensor(value) for value in (crop, disease)]
    if any(value.ndim != 2 or len(value) != len(labels) or not bool(torch.isfinite(value).all())
           for value in arrays):
        raise ValueError("Invalid validation factor logits")
    scores = compose_pair_scores(*(torch.log_softmax(value, 1).numpy() for value in arrays), pairs)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation composition scores")
    scores[:, pairs.is_unseen.to_numpy()] = -np.inf
    validation_pairs = pairs.assign(is_unseen=pairs.is_unseen | pairs.pair_id.isin(pseudo_ids))
    metrics = evaluate(scores[:len(seen)], labels[:len(seen)], scores[len(seen):],
                       labels[len(seen):], validation_pairs, gamma=0.0)
    return {"objective_name": OBJECTIVE_NAME, "gamma": 0.0,
            "harmonic_mean_top1": metrics["harmonic_mean_top1"],
            "gamma0_harmonic_mean_top1": metrics["harmonic_mean_top1"],
            "seen_top1": metrics["test_seen"]["top1"],
            "unseen_top1": metrics["test_unseen"]["top1"],
            "seen_top5": metrics["test_seen"]["top5"],
            "unseen_top5": metrics["test_unseen"]["top5"],
            "harmonic_mean_top5": metrics["harmonic_mean_top5"]}
