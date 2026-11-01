"""Primitive fusion (v9+): add the crop and disease prompt scores to each pair's score at prediction time.

The crop and disease prompts are trained every step (the crop and disease losses) but plain scoring uses only the
pair prompt. Here pair p (crop c, disease d) scores
    (pair(p) + a * crop(c) + b * disease(d)) / (1 + a + b)
All three are cosines, so the division keeps the scores on the cosine scale and the gamma grid still fits.
a = b = 0 is exactly the plain pair score.
"""

import numpy as np


def fuse(pair, crop, disease, pair_crop, pair_disease, a: float, b: float):
    """pair: (N, 333), crop: (N, n_crops), disease: (N, n_diseases) cosines; pair_crop / pair_disease: id per pair.
    A term whose weight is 0 is skipped, so its scores may be None."""
    if not a and not b:
        return pair
    out = pair.copy()
    if a:
        out += a * crop[:, pair_crop]
    if b:
        out += b * disease[:, pair_disease]
    return out / (1 + a + b)


def bootstrap_hm_diff(seen_new, unseen_new, seen_base, unseen_base, n=2000, seed=0):
    """Paired bootstrap of HM(new) - HM(base) over validation images (seen and unseen resampled separately).
    Inputs are per-image top-1 correctness (bool) on the same images. Returns (mean diff, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(seen_new), (n, len(seen_new)))
    j = rng.integers(0, len(unseen_new), (n, len(unseen_new)))

    def hm(s, u):
        a, b = s[i].mean(1), u[j].mean(1)
        return np.where(a + b > 0, 2 * a * b / np.maximum(a + b, 1e-12), 0.0)

    d = hm(seen_new, unseen_new) - hm(seen_base, unseen_base)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
