"""vlm2 report: each controlled run against the common base, on the same validation images (config/vlm2/README.md).

Usage:  python -m train.vlm2_report [--boot 1000]

Reads every runs/vlm2/<name>/ with a config carrying `_vlm2` and saved validation scores (validation.save_scores),
takes each run's scores at its selected epoch (and post-hoc knobs), and compares it with s0_base on the same images:
  - gamma-free: AUSUC, closed-set seen Top-1, closed-set unseen Top-1 (mean of the 3 splits; real unseen pairs hidden)
  - HM at the run's own validation gamma (the protocol number; selection is by HM)
Differences get a paired bootstrap 95% interval (images resampled within each split, val-seen and pseudo-unseen
separately, the same draw for both runs). The noise floor is |s0_base_r2 - s0_base|. A change is "real" when its
interval excludes 0 and |delta AUSUC| >= max(1.5 pts, 2 x noise floor).

Also scored here without training: C2 = s0_base + primitive fusion weights chosen on validation (by HM, with the
epoch), from s0_base's saved crop / disease scores.

Runs with two or more factors (stage 3+) get the interaction I = delta(A+B) - delta(A) - delta(B) against the single-
factor runs: complementary when I's interval contains 0, overlapping when delta(A+B) is within noise of
max(delta(A), delta(B)) and I < 0, antagonistic when delta(A+B) < max(delta(A), delta(B)) beyond noise.
Writes <runs-dir>/vlm2_report.csv and docs/vlm/vlm2_report.md (--report).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.evaluate import hm_top1_over_gammas
from dpd.fusion import fuse

RUNS = PROJECT_ROOT / "runs" / "vlm2"
BASE, REPLICATE = "s0_base", "s0_base_r2"
FUSION_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
METRICS = ("ausuc", "closed_seen", "closed_unseen", "hm")


def split_views(scores, labels, n_vs, trained, hidden):
    """Per-image arrays for one split: gap (best trained - best pseudo-unseen), whether the best trained / best
    pseudo-unseen pair is right; seen and unseen images separately."""
    pseudo = ~trained & ~hidden
    t_ids, p_ids = np.flatnonzero(trained), np.flatnonzero(pseudo)
    best_t, best_p = scores[:, t_ids].max(1), scores[:, p_ids].max(1)
    ok_t = t_ids[scores[:, t_ids].argmax(1)] == labels
    ok_p = p_ids[scores[:, p_ids].argmax(1)] == labels
    gap = best_t - best_p
    return (gap[:n_vs], ok_t[:n_vs], ok_p[:n_vs]), (gap[n_vs:], ok_t[n_vs:], ok_p[n_vs:])


def acc(view, gamma):
    gap, ok_seen, ok_unseen = view
    return np.where(gamma > gap, ok_unseen, ok_seen).mean()


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def ausuc(seen, unseen):
    """Area under the seen/unseen curve from per-image gaps (as dpd.evaluate.seen_unseen_curve + ausuc)."""
    t = np.unique(np.concatenate([seen[0], unseen[0]]))
    t = np.concatenate([[t[0] - 1.0], t, [t[-1] + 1.0]])

    def curve(view):
        gap, ok_seen, ok_unseen = view
        o = np.argsort(gap, kind="stable")
        cs, cu = np.concatenate([[0], np.cumsum(ok_seen[o])]), np.concatenate([[0], np.cumsum(ok_unseen[o])])
        k = np.searchsorted(gap[o], t, side="left")  # images with gap < t take their unseen pick
        return (cu[k] + cs[-1] - cs[k]) / len(gap)

    a, b = curve(seen), curve(unseen)
    o = np.lexsort((b, a))
    return float(np.trapezoid(b[o], a[o]))


def metrics(run, draw=None):
    """Mean over splits of the four metrics; `draw` = per split (seen index, unseen index) resample."""
    out = {m: [] for m in METRICS}
    for k, (seen, unseen) in enumerate(run["views"]):
        if draw is not None:
            i_s, i_u = draw[k]
            seen, unseen = tuple(x[i_s] for x in seen), tuple(x[i_u] for x in unseen)
        out["ausuc"].append(ausuc(seen, unseen))
        out["closed_seen"].append(seen[1].mean())
        out["closed_unseen"].append(unseen[2].mean())
        out["hm"].append(hm(acc(seen, run["gamma"]), acc(unseen, run["gamma"])))
    return {m: float(np.mean(v)) for m, v in out.items()}


def load_split(path, epoch_idx, pairs, knobs=None):
    z = np.load(path)
    s = z["pair"][epoch_idx].astype(np.float64)
    if knobs and (knobs.get("crop_weight") or knobs.get("disease_weight")):
        s = fuse(s, z["crop"][epoch_idx], z["disease"][epoch_idx], pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(),
                 knobs.get("crop_weight", 0.0), knobs.get("disease_weight", 0.0))
    trained = np.isin(pairs["pair_id"].to_numpy(), z["trained_pairs"])
    return s, z["labels"], int(z["n_val_seen"]), trained, z["files"]


def load_run(d: Path, pairs) -> dict | None:
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    meta = (d / "metrics.json") if (d / "metrics.json").exists() else (d / "val_metrics.json")
    if "_vlm2" not in cfg or not meta.exists() or not list(d.glob("val_scores_split*.npz")):
        return None
    m = json.loads(meta.read_text())
    knobs = m.get("knobs") or {"crop_weight": m.get("fusion_crop_weight", 0.0), "disease_weight": m.get("fusion_disease_weight", 0.0)}
    hidden = pairs["is_unseen"].to_numpy()  # real unseen pairs: in no validation label space
    views, files = [], []
    for k in range(len(list(d.glob("val_scores_split*.npz")))):
        s, y, n_vs, trained, f = load_split(d / f"val_scores_split{k + 1}.npz", m["selected_epoch"] - 1, pairs, knobs)
        views.append(split_views(s, y, n_vs, trained & ~hidden, hidden))
        files.append(f)
    return {"name": d.name, "group": cfg["_vlm2"]["group"], "factors": cfg["_vlm2"]["factors"], "stage": cfg["_vlm2"]["stage"],
            "gamma": float(m["gamma"]), "epoch": m["selected_epoch"], "views": views, "files": files, "knobs": knobs}


def fusion_from_base(d: Path, pairs, gammas) -> dict:
    """C2: s0_base's saved scores with primitive fusion weights; (epoch, a, b, gamma) chosen by mean validation HM."""
    hidden = pairs["is_unseen"].to_numpy()
    zs = [np.load(p) for p in sorted(d.glob("val_scores_split*.npz"))]
    best = (-1.0, None)
    for e in range(zs[0]["pair"].shape[0]):
        for a, b in itertools.product(FUSION_GRID, FUSION_GRID):
            hms = []
            for z in zs:
                s = fuse(z["pair"][e].astype(np.float64), z["crop"][e], z["disease"][e], pairs["crop_id"].to_numpy(),
                         pairs["disease_id"].to_numpy(), a, b).copy()
                s[:, hidden] = -1e9
                n_vs, trained = int(z["n_val_seen"]), np.isin(pairs["pair_id"].to_numpy(), z["trained_pairs"])
                vp = pairs.assign(is_unseen=~trained)
                y = z["labels"]
                hms.append(hm_top1_over_gammas(s[:n_vs], y[:n_vs], s[n_vs:], y[n_vs:], vp, gammas))
            mean = np.mean(hms, axis=0)
            if mean.max() > best[0]:
                best = (float(mean.max()), (e, a, b, float(gammas[mean.argmax()])))
    e, a, b, g = best[1]
    views, files = [], []
    for k in range(len(zs)):
        s, y, n_vs, trained, f = load_split(sorted(d.glob("val_scores_split*.npz"))[k], e, pairs, {"crop_weight": a, "disease_weight": b})
        views.append(split_views(s, y, n_vs, trained & ~hidden, hidden))
        files.append(f)
    return {"name": "s2_C2_posthoc_fusion (from s0_base)", "group": "C", "factors": ["C2"], "stage": 2, "gamma": g, "epoch": e + 1,
            "views": views, "files": files, "knobs": {"crop_weight": a, "disease_weight": b}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--runs-dir", default=str(RUNS), help="folder holding the vlm2 runs (default runs/vlm2)")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "docs" / "vlm" / "vlm2_report.md"))
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    sys.stdout.reconfigure(encoding="utf-8")  # the table uses Δ; the Windows console default (cp1252) cannot print it
    pairs = pair_table(read_metadata())
    runs = {r["name"]: r for d in sorted(runs_dir.glob("*")) if d.is_dir() and (r := load_run(d, pairs))}
    assert BASE in runs, f"{BASE} has no saved validation scores yet"
    base_cfg = json.loads((runs_dir / BASE / "config.json").read_text())
    c2 = fusion_from_base(runs_dir / BASE, pairs, np.round(np.arange(*base_cfg["validation"]["gamma_grid"]), 6))
    runs[c2["name"]] = c2
    base = runs[BASE]
    for r in runs.values():  # paired comparisons need the same images in the same order
        assert all((a == b).all() for a, b in zip(r["files"], base["files"])), f"{r['name']}: validation images differ from {BASE}"

    rng = np.random.default_rng(0)
    draws = [[(rng.integers(0, len(s[0]), len(s[0])), rng.integers(0, len(u[0]), len(u[0]))) for s, u in base["views"]]
             for _ in range(args.boot)]
    point = {n: metrics(r) for n, r in runs.items()}
    boot = {n: [metrics(r, dr) for dr in draws] for n, r in runs.items()}

    def delta(a, b):  # point delta and 95% interval of a - b per metric
        out = {}
        for m in METRICS:
            diffs = np.array([x[m] - y[m] for x, y in zip(boot[a], boot[b])])
            out[m] = (point[a][m] - point[b][m], *np.percentile(diffs, [2.5, 97.5]))
        return out

    noise = abs(point[REPLICATE]["ausuc"] - point[BASE]["ausuc"]) if REPLICATE in runs else float("nan")
    threshold = max(0.015, 2 * noise) if np.isfinite(noise) else 0.015
    rows = []
    for n, r in runs.items():
        row = {"run": n, "stage": r["stage"], "group": r["group"], "factors": "+".join(r["factors"]), "epoch": r["epoch"],
               "gamma": r["gamma"], **{f"{m}": point[n][m] for m in METRICS}}
        if n != BASE:
            d = delta(n, BASE)
            for m in METRICS:
                row[f"d_{m}"], row[f"{m}_lo"], row[f"{m}_hi"] = d[m]
            excl = d["ausuc"][1] > 0 or d["ausuc"][2] < 0
            row["verdict"] = ("noise floor" if n == REPLICATE else
                              ("real +" if d["ausuc"][0] > 0 else "real -") if excl and abs(d["ausuc"][0]) >= threshold else "not real")
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["stage", "group", "run"])

    # Interactions: every multi-factor run against its single-factor parts.
    single = {r["factors"][0]: n for n, r in runs.items() if len(r["factors"]) == 1 and r["stage"] == 2}  # stage-2 add-one runs
    inter = []
    for n, r in runs.items():
        if len(r["factors"]) < 2 or not all(f in single for f in r["factors"]):
            continue
        parts = [single[f] for f in r["factors"]]
        i_boot = np.array([(b[n]["ausuc"] - b[BASE]["ausuc"]) - sum(b[p]["ausuc"] - b[BASE]["ausuc"] for p in parts)
                           for b in ({k: boot[k][i] for k in boot} for i in range(args.boot))])
        d_all = point[n]["ausuc"] - point[BASE]["ausuc"]
        d_parts = [point[p]["ausuc"] - point[BASE]["ausuc"] for p in parts]
        i_point = d_all - sum(d_parts)
        lo, hi = np.percentile(i_boot, [2.5, 97.5])
        best_part = max(d_parts)
        kind = ("complementary" if lo <= 0 <= hi else
                "antagonistic" if d_all < best_part - threshold else
                "overlapping" if i_point < 0 else "synergistic")
        inter.append({"run": n, "parts": " + ".join(parts), "d_combined": d_all, "d_parts": d_parts, "sum_parts": sum(d_parts),
                      "interaction": i_point, "interaction_lo": lo, "interaction_hi": hi, "kind": kind})

    table.to_csv(runs_dir / "vlm2_report.csv", index=False, float_format="%.4f")
    pct = lambda v: f"{v * 100:+.1f}"
    lines = ["# vlm2 report (validation, mean over the splits; generated by train/vlm2_report.py)", "",
             f"Base {BASE}: AUSUC {point[BASE]['ausuc']:.2%}, closed seen {point[BASE]['closed_seen']:.2%}, "
             f"closed unseen {point[BASE]['closed_unseen']:.2%}, HM {point[BASE]['hm']:.2%}.",
             f"Noise floor |{REPLICATE} - {BASE}| AUSUC: {noise * 100:.2f} pts; decision threshold {threshold * 100:.2f} pts "
             f"(and the 95% interval must exclude 0). Bootstrap draws: {args.boot}.", "",
             "| Run | Group | Δ AUSUC [95% CI] | Δ closed seen | Δ closed unseen | Δ HM (own γ) | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        if r.run == BASE:
            continue
        lines.append(f"| {r.run} | {r.group} | {pct(r.d_ausuc)} [{pct(r.ausuc_lo)}, {pct(r.ausuc_hi)}] | {pct(r.d_closed_seen)} | "
                     f"{pct(r.d_closed_unseen)} | {pct(r.d_hm)} | {r.verdict} |")
    if inter:
        lines += ["", "## Interactions (AUSUC)", "", "| Run | Parts | Δ combined | Δ parts | I [95% CI] | Kind |", "|---|---|---|---|---|---|"]
        for x in inter:
            lines.append(f"| {x['run']} | {x['parts']} | {pct(x['d_combined'])} | {', '.join(pct(v) for v in x['d_parts'])} | "
                         f"{pct(x['interaction'])} [{pct(x['interaction_lo'])}, {pct(x['interaction_hi'])}] | {x['kind']} |")
    (PROJECT_ROOT / "docs" / "vlm" / "vlm2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
