"""Shared evaluation protocol. Every model hands in a score matrix over all 333 pairs, so results stay comparable.

Protocol (generalised zero-shot): each test image is ranked against ALL pairs (seen + unseen), for both test splits.
  - Top-1 / Top-5 accuracy per split (per image), plus the per-class mean as a balance check.
  - Harmonic mean of seen and unseen accuracy, for Top-1 and Top-5.
  - Confusion matrix for test_seen: rows = the 323 seen pairs (true), columns = all 333 pairs (predicted),
    so seen images misclassified as an unseen pair stay visible.
`gamma` (calibrated stacking) is subtracted from every seen pair's score before ranking; it must be chosen
without looking at the test sets.
Exact score ties prefer the lower pair id, consistently across metrics, predictions and confusion matrices.
"""

import numpy as np
import pandas as pd


TIE_BREAK_POLICY = "score_descending_then_pair_id_ascending"


def parse_gamma_grid(grid_cfg) -> np.ndarray:
    """Parse gamma grid configuration into a numpy array of gammas.

    If grid_cfg is 0, 0.0, [0.0], [0.0, 0.0], or None/False, disables gamma tuning and returns np.array([0.0]).
    """
    if grid_cfg is None or grid_cfg is False or grid_cfg == 0 or grid_cfg == 0.0:
        return np.array([0.0], dtype=np.float64)
    if isinstance(grid_cfg, (int, float)):
        return np.array([float(grid_cfg)], dtype=np.float64)
    if isinstance(grid_cfg, (list, tuple)):
        if len(grid_cfg) == 0:
            return np.array([0.0], dtype=np.float64)
        if len(grid_cfg) == 1:
            return np.array([float(grid_cfg[0])], dtype=np.float64)
        start = float(grid_cfg[0])
        stop = float(grid_cfg[1])
        step = float(grid_cfg[2]) if len(grid_cfg) >= 3 else 1.0
        if step <= 0.0 or start >= stop:
            return np.array([start], dtype=np.float64)
        arr = np.round(np.arange(start, stop, step), 6)
        if len(arr) == 0:
            return np.array([start], dtype=np.float64)
        return arr
    return np.array([0.0], dtype=np.float64)


def gamma0_index(gammas) -> int:
    """Position of gamma = 0 in a gamma grid, for reporting the uncalibrated result from a sweep.

    Grids may start below 0 (e.g. the feature-generation runs sweep -0.1..0.1), so index 0 is not gamma 0 in general.
    """
    hits = np.flatnonzero(np.isclose(np.asarray(gammas, dtype=np.float64), 0.0, rtol=0.0, atol=1e-9))
    if len(hits) == 0:
        raise ValueError("gamma grid does not contain 0; add it so the uncalibrated (gamma=0) result can be reported")
    return int(hits[0])


def calibrate(scores: np.ndarray, unseen_mask: np.ndarray, gamma: float) -> np.ndarray:
    """Subtract `gamma` from the seen-pair columns (counteracts the bias towards seen classes)."""
    return scores - gamma * (~unseen_mask)[None, :]


def ranked_indices(scores: np.ndarray, k: int | None = None) -> np.ndarray:
    """Rank the last axis by descending score, then ascending pair id on exact ties.

    Columns must follow increasing pair id, as pair_table() and the unseen-only
    subset do. Stable sorting preserves that order without changing score values.
    Accepts a single image or a batch; k larger than the candidate count returns all.
    """
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    order = np.argsort(-np.asarray(scores), axis=-1, kind="stable")
    return order if k is None else order[..., :k]


def topk_hits(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Boolean per image: is the true label among the k highest scores?"""
    topk = ranked_indices(scores, k)
    return (topk == labels[:, None]).any(axis=1)


def harmonic_mean(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def evaluate_split(scores: np.ndarray, labels: np.ndarray, pairs: pd.DataFrame, top_k=(1, 5)) -> dict:
    """Accuracy for one split. `pairs` is dpd.data.pair_table() (row i = pair id i)."""
    order = ranked_indices(scores)
    pred = order[:, 0]
    out = {"n_images": int(len(labels)), "n_classes": int(len(np.unique(labels)))}
    for k in top_k:
        hits = (order[:, :k] == labels[:, None]).any(axis=1)
        out[f"top{k}"] = float(hits.mean())
        # Mean over classes: guards against one big class (e.g. tomato_early_blight is 1/3 of test_unseen).
        out[f"top{k}_class_mean"] = float(pd.Series(hits).groupby(labels).mean().mean())
    # Diagnostics for the compositional view: was the crop / disease right even when the pair was wrong?
    crop_ids, disease_ids = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    out["crop_top1"] = float((crop_ids[pred] == crop_ids[labels]).mean())
    out["disease_top1"] = float((disease_ids[pred] == disease_ids[labels]).mean())
    out["predicted_as_unseen"] = float(pairs["is_unseen"].to_numpy()[pred].mean())
    return out


def evaluate(
    scores_seen: np.ndarray,
    labels_seen: np.ndarray,
    scores_unseen: np.ndarray,
    labels_unseen: np.ndarray,
    pairs: pd.DataFrame,
    gamma: float = 0.0,
    top_k=(1, 5),
) -> dict:
    unseen_mask = pairs["is_unseen"].to_numpy()
    s_seen = calibrate(scores_seen, unseen_mask, gamma)
    s_unseen = calibrate(scores_unseen, unseen_mask, gamma)
    seen = evaluate_split(s_seen, labels_seen, pairs, top_k)
    unseen = evaluate_split(s_unseen, labels_unseen, pairs, top_k)
    result = {"gamma": gamma, "test_seen": seen, "test_unseen": unseen}
    for k in top_k:
        result[f"harmonic_mean_top{k}"] = harmonic_mean(seen[f"top{k}"], unseen[f"top{k}"])

    # Diagnostic only (not the protocol): unseen accuracy when choosing among the 10 unseen pairs alone.
    # The gap to test_unseen.top1 shows how much the seen pairs are stealing unseen images.
    unseen_ids = np.flatnonzero(unseen_mask)
    remap = {p: i for i, p in enumerate(unseen_ids)}
    closed = scores_unseen[:, unseen_ids]
    closed_labels = np.array([remap[p] for p in labels_unseen])
    result["diagnostic_unseen_only_top1"] = float(topk_hits(closed, closed_labels, 1).mean())
    return result


def top1_over_gammas(scores: np.ndarray, labels: np.ndarray, unseen_mask: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Top-1 accuracy for every gamma at once; same ranking as calibrate() + argmax (up to exact ties).

    Calibration only shifts seen against unseen columns, so the prediction is either the best seen pair or the best
    unseen pair: the unseen one wins when its score beats the best seen score minus gamma.
    """
    seen_ids, unseen_ids = np.flatnonzero(~unseen_mask), np.flatnonzero(unseen_mask)
    s, u = scores[:, seen_ids], scores[:, unseen_ids]
    ok_seen = seen_ids[s.argmax(axis=1)] == labels
    ok_unseen = unseen_ids[u.argmax(axis=1)] == labels
    pick_unseen = u.max(axis=1)[None, :] > s.max(axis=1)[None, :] - np.asarray(gammas)[:, None]
    return np.where(pick_unseen, ok_unseen[None, :], ok_seen[None, :]).mean(axis=1)


def hm_top1_over_gammas(scores_seen, labels_seen, scores_unseen, labels_unseen, pairs: pd.DataFrame, gammas) -> np.ndarray:
    """Top-1 harmonic mean for every gamma; a fast equivalent of evaluate(...)["harmonic_mean_top1"] for sweeps."""
    unseen_mask = pairs["is_unseen"].to_numpy()
    a = top1_over_gammas(scores_seen, labels_seen, unseen_mask, gammas)
    b = top1_over_gammas(scores_unseen, labels_unseen, unseen_mask, gammas)
    return np.where(a + b > 0, 2 * a * b / np.maximum(a + b, 1e-12), 0.0)


def seen_unseen_curve(scores_seen, labels_seen, scores_unseen, labels_unseen, pairs: pd.DataFrame) -> pd.DataFrame:
    """Seen and unseen Top-1 at every calibration gamma where a prediction changes (Chao et al., 2016).

    An image switches from its best seen pair to its best unseen pair once gamma exceeds (best seen - best unseen),
    so those per-image gaps are the only gammas where accuracy can change; one extra point on each side of them
    gives the ends. Rows run from all-seen (unseen Top-1 0) to all-unseen (seen Top-1 0).
    """
    unseen_mask = pairs["is_unseen"].to_numpy()

    def gaps(s):
        return s[:, ~unseen_mask].max(axis=1) - s[:, unseen_mask].max(axis=1)

    g = np.unique(np.concatenate([gaps(scores_seen), gaps(scores_unseen)]))
    g = np.concatenate([[g[0] - 1.0], g, [g[-1] + 1.0]])
    return pd.DataFrame({
        "gamma": g,
        "seen": top1_over_gammas(scores_seen, labels_seen, unseen_mask, g),
        "unseen": top1_over_gammas(scores_unseen, labels_unseen, unseen_mask, g),
    })


def ausuc(curve: pd.DataFrame) -> float:
    """Area under the seen-unseen curve (x seen Top-1, y unseen Top-1, trapezoid rule): 1 for a model that is perfect
    on both at once. Independent of the chosen gamma, unlike the harmonic mean, which is one point on the curve."""
    c = curve.sort_values(["seen", "unseen"])
    return float(np.trapezoid(c["unseen"], c["seen"]))


def seen_confusion_matrix(scores_seen: np.ndarray, labels_seen: np.ndarray, pairs: pd.DataFrame, gamma: float = 0.0) -> pd.DataFrame:
    """Counts with rows = true seen pair, columns = predicted pair (all 333), labelled by pair name."""
    pred = ranked_indices(calibrate(scores_seen, pairs["is_unseen"].to_numpy(), gamma), 1)[:, 0]
    seen_ids = pairs.loc[~pairs["is_unseen"], "pair_id"].to_numpy()
    cm = np.zeros((len(seen_ids), len(pairs)), dtype=np.int32)
    row_of = {p: i for i, p in enumerate(seen_ids)}
    np.add.at(cm, ([row_of[t] for t in labels_seen], pred), 1)
    names = pairs["pair"].to_numpy()
    return pd.DataFrame(cm, index=pd.Index(names[seen_ids], name="true"), columns=pd.Index(names, name="predicted"))


def predictions_table(files, scores: np.ndarray, labels: np.ndarray, pairs: pd.DataFrame, gamma: float = 0.0, k: int = 5) -> pd.DataFrame:
    """Per-image top-k predictions, for error analysis and the UI."""
    s = calibrate(scores, pairs["is_unseen"].to_numpy(), gamma)
    order = ranked_indices(s, k)
    names = pairs["pair"].to_numpy()
    df = pd.DataFrame({"file": list(files), "true": names[labels], "correct_top1": order[:, 0] == labels})
    for i in range(order.shape[1]):
        df[f"pred{i + 1}"] = names[order[:, i]]
        df[f"score{i + 1}"] = np.take_along_axis(s, order[:, i : i + 1], axis=1)[:, 0]
    return df


def format_summary(result: dict) -> str:
    s, u = result["test_seen"], result["test_unseen"]
    lines = [
        f"gamma = {result['gamma']:g}",
        f"{'':22s}{'top1':>8s}{'top5':>8s}{'top1 cls':>10s}{'crop':>8s}{'disease':>9s}{'->unseen':>10s}",
    ]
    for name, r in (("test_seen", s), ("test_unseen", u)):
        lines.append(
            f"{name:22s}{r['top1']:8.2%}{r['top5']:8.2%}{r['top1_class_mean']:10.2%}"
            f"{r['crop_top1']:8.2%}{r['disease_top1']:9.2%}{r['predicted_as_unseen']:10.2%}"
        )
    lines.append(f"{'harmonic mean':22s}{result['harmonic_mean_top1']:8.2%}{result['harmonic_mean_top5']:8.2%}")
    lines.append(f"(diagnostic) unseen top1 among the 10 unseen pairs only: {result['diagnostic_unseen_only_top1']:.2%}")
    return "\n".join(lines)
