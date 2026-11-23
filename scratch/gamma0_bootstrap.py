"""Paired bootstrap of each iteration step on the test sets: is the change bigger than resampling noise?

For every parent -> child step (PARENT in scratch/gamma0_audit.py, plus a few extra comparisons) the seen-test and
unseen-test images are resampled with replacement (independently per split, the same draw for both runs), and the
difference child - parent is recorded for:
  AUSUC, closed-set seen Top-1, closed-set unseen Top-1   (gamma-free)
  HM at gamma 0                                           (uncalibrated)
  HM at each run's own validation gamma                   (calibrated, the protocol number)
Output: docs/vlm/gamma0_audit_bootstrap.csv (mean delta and 95% interval per metric).
Test-only evidence: it says which changes are real on this test set, not which model to choose.
Run from the project root:  PYTHONPATH=. .venv/Scripts/python scratch/gamma0_bootstrap.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma0_audit import PARENT, RUNS  # noqa: E402

from dpd.data import pair_table, read_metadata
from dpd.evaluate import ausuc, seen_unseen_curve

OUT = Path(__file__).resolve().parents[1] / "docs" / "vlm" / "gamma0_audit_bootstrap.csv"
EXTRA = [("v11.2", "v11"), ("v10.3", "v11.3"), ("v5.8", "v5.14"), ("v4.2", "v5.5"), ("v5.5", "v5.13"), ("v5.5", "v5.14"),
         ("v10.1", "v10.3"), ("v5.5", "v5.8"), ("v5.13", "v11.3"), ("v5.13", "v6.5"), ("v5.13", "v11.5"), ("v10.3", "v11.5")]
B = 1000


def per_image(run: Path, unseen_mask):
    """For each test image: gap = best seen - best unseen score, and whether the best seen / best unseen pair is right."""
    z = np.load(run / "scores.npz")
    seen_ids, unseen_ids = np.flatnonzero(~unseen_mask), np.flatnonzero(unseen_mask)
    out = {}
    for split in ("test_seen", "test_unseen"):
        s, y = z[f"{split}_scores"], z[f"{split}_labels"]
        out[split] = (s[:, seen_ids].max(1) - s[:, unseen_ids].max(1),
                      seen_ids[s[:, seen_ids].argmax(1)] == y, unseen_ids[s[:, unseen_ids].argmax(1)] == y)
    return out


def acc(gap, cs, cu, gamma):
    return np.where(gamma > gap, cu, cs).mean()


def fast_ausuc(seen, unseen):
    """Same curve as dpd.evaluate.seen_unseen_curve + ausuc, via sorting (O(n log n))."""
    (gs, css, cus), (gu, csu, cuu) = seen, unseen
    t = np.unique(np.concatenate([gs, gu]))
    t = np.concatenate([[t[0] - 1.0], t, [t[-1] + 1.0]])

    def curve(gap, cs, cu):
        o = np.argsort(gap, kind="stable")
        g, cs_c, cu_c = gap[o], np.concatenate([[0], np.cumsum(cs[o])]), np.concatenate([[0], np.cumsum(cu[o])])
        k = np.searchsorted(g, t, side="left")
        return (cu_c[k] + (cs_c[-1] - cs_c[k])) / len(gap)

    a, b = curve(gs, css, cus), curve(gu, csu, cuu)
    o = np.lexsort((b, a))
    return float(np.trapezoid(b[o], a[o]))


def metrics(d, gamma, i_s, i_u):
    s = tuple(x[i_s] for x in d["test_seen"])
    u = tuple(x[i_u] for x in d["test_unseen"])
    hm = lambda a, b: 0.0 if a + b == 0 else 2 * a * b / (a + b)
    return {
        "ausuc": fast_ausuc(s, u),
        "closed_seen": s[1].mean(), "closed_unseen": u[2].mean(),
        "hm_g0": hm(acc(*s, 0.0), acc(*u, 0.0)),
        "hm_cal": hm(acc(*s, gamma), acc(*u, gamma)),
    }


def main():
    pairs = pair_table(read_metadata())
    unseen_mask = pairs["is_unseen"].to_numpy()
    runs = {}
    for p in RUNS.glob("*/*"):
        if (p / "scores.npz").exists() and (p / "metrics.json").exists() and p.parent.name != "cache":
            cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
            runs[cfg.get("version") or p.name] = p
    steps = [(p, c) for c, p in PARENT.items() if c in runs and p in runs] + [s for s in EXTRA if s[0] in runs and s[1] in runs]
    cache = {}

    def load(name):
        if name not in cache:
            g = float(json.loads((runs[name] / "metrics.json").read_text()).get("gamma", 0.0))
            cache[name] = (per_image(runs[name], unseen_mask), g)
        return cache[name]

    # sanity: the fast AUSUC equals dpd.evaluate.ausuc on the full test sets
    z = np.load(runs["v5.5"] / "scores.npz")
    ref = ausuc(seen_unseen_curve(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"], pairs))
    d55 = load("v5.5")[0]
    assert abs(fast_ausuc(d55["test_seen"], d55["test_unseen"]) - ref) < 1e-9, "fast AUSUC disagrees with dpd.evaluate"

    rng = np.random.default_rng(0)
    n_s, n_u = len(d55["test_seen"][0]), len(d55["test_unseen"][0])
    draws = [(rng.integers(0, n_s, n_s), rng.integers(0, n_u, n_u)) for _ in range(B)]
    full = (np.arange(n_s), np.arange(n_u))
    rows = []
    for parent, child in steps:
        (dp, gp), (dc, gc) = load(parent), load(child)
        point = {k: metrics(dc, gc, *full)[k] - v for k, v in metrics(dp, gp, *full).items()}
        boot = pd.DataFrame([{k: v - metrics(dp, gp, i_s, i_u)[k] for k, v in metrics(dc, gc, i_s, i_u).items()} for i_s, i_u in draws])
        row = {"step": f"{parent} -> {child}"}
        for k in point:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            row[f"d_{k}"], row[f"{k}_lo"], row[f"{k}_hi"] = point[k], lo, hi
            row[f"{k}_sig"] = "+" if lo > 0 else ("-" if hi < 0 else "")
        rows.append(row)
        print(row["step"], {k: f"{row[f'd_{k}']*100:+.1f} [{row[f'{k}_lo']*100:+.1f},{row[f'{k}_hi']*100:+.1f}]" for k in point}, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
