"""Gamma-0 audit: every finished run re-scored with calibration off, plus gamma-free diagnostics.

Reads only saved artefacts (no retraining):
  - runs/<family>/<name>/scores.npz      raw (uncalibrated) test score matrices
  - runs/<family>/<name>/val_hm_grid.npy validation HM over (splits x epochs x [knobs] x gammas) or (weights x gammas)
  - runs/<family>/<name>/metrics.json    the selected gamma / epoch / weight

Writes docs/vlm/gamma0_audit.csv (one row per run) and docs/vlm/gamma0_audit_deltas.csv (one row per iteration step).
Run from the project root:  .venv/Scripts/python scratch/gamma0_audit.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dpd.data import pair_table, read_metadata
from dpd.evaluate import ausuc, evaluate, parse_gamma_grid, seen_unseen_curve

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "vlm"

# Parent of each iteration (the run it changed ONE thing from, per its config _note).
PARENT = {
    "v2": "v1", "v3": "v2", "v3.1": "v3", "v3.2": "v3", "v3.3": "v3", "v3.4": "v3", "v3.5": "v3.4", "v3.6": "v3.4",
    "v3.7": "v3.6", "v3.8": "v3", "v3.9": "v3",
    "v4": "v3", "v4.1": "v4", "v4.2": "v4.1", "v4.3": "v4.2",
    "v5": "v4.2", "v5.1": "v5", "v5.2": "v5", "v5.3": "v5", "v5.4": "v5", "v5.5": "v5",
    "v5.6": "v5.5", "v5.7": "v5.5", "v5.8": "v5.5",
    "v5.9": "v5.8", "v5.10": "v5.8", "v5.11": "v5.8", "v5.12": "v5.8", "v5.13": "v5.8", "v5.14": "v5.13",
    "v7": "v5.5", "v7.1": "v5.5", "v7.2": "v5.5", "v9": "v5.5",
    "v10": "v5.5", "v10.1": "v5.5", "v10.2": "v5.5", "v10.3": "v5.8", "v12": "v10.3",
    "v13": "v5.5", "v13.1": "v13",
    "v11.1": "v11", "v11.2": "v11", "v11.3": "v11", "v11.4": "v11.1", "v11.5": "v11.3",
    "v6": "v5", "v6.1": "v6", "v6.2": "v6.1", "v6.3": "v6.1", "v6.4": "v6.3", "v6.5": "v6.3", "v6.6": "v6.4",
    "v6.7": "v6.6", "v6.8": "v6.3", "v14": "v6.3",
    "aniq_cropnorm_v1": "aniq_cropnorm_v0", "aniq_cropnorm_v2": "aniq_cropnorm_v1", "aniq_cropnorm_v3": "aniq_cropnorm_v2",
    "aniq_cropnorm_v4": "aniq_cropnorm_v3",
    "aniq_baseline_independent_convnext_base": "aniq_baseline_independent_cnn",
    "aniq_independent_convnext_base_336_1split": "aniq_baseline_independent_convnext_base",
}
# v10.3 is v10.1 rerun on 3 splits; its like-for-like base on the same 3 splits is v5.8 (= v5.5 on 3 splits).


def gammas_of(cfg):
    grid = (cfg.get("validation") or {}).get("gamma_grid")
    if grid is None:
        return None
    if cfg.get("method", "").startswith("cnn") or cfg.get("method") in ("multi_task_cnn",):
        return parse_gamma_grid(grid)
    return np.round(np.arange(*grid), 6)


def val_numbers(run: Path, cfg: dict, metrics: dict) -> dict:
    """Validation HM at the selected gamma, at gamma 0 for the same epoch/knobs, and the best gamma-0 HM over epochs/knobs."""
    path = run / "val_hm_grid.npy"
    gam = gammas_of(cfg)
    if not path.exists() or gam is None:
        return {}
    grid = np.load(path)
    if grid.shape[-1] != len(gam):
        return {"val_note": f"grid has {grid.shape[-1]} gammas, config grid {len(gam)}"}
    i0 = np.flatnonzero(np.isclose(gam, 0.0))
    if len(i0) == 0:
        return {"val_note": "gamma 0 not in grid"}
    i0 = int(i0[0])
    ensemble = cfg.get("method") == "ensemble"
    n_splits = 1 if ensemble else grid.shape[0]
    mean = grid if ensemble else grid.mean(axis=0)
    flat = mean.reshape(-1, mean.shape[-1])
    sel_row, sel_g = np.unravel_index(flat.argmax(), flat.shape)
    out = {
        "val_splits": n_splits,
        "val_hm_cal": float(flat[sel_row, sel_g]),
        "val_gamma": float(gam[sel_g]),
        "val_hm_g0_same_epoch": float(flat[sel_row, i0]),
        "val_hm_g0_best": float(flat[:, i0].max()),
        "val_gamma0_index": i0,
    }
    if not ensemble:
        # epoch picked if gamma were off (first non-split, non-gamma axis is the epoch)
        per_epoch = mean[..., i0].reshape(mean.shape[0], -1).max(axis=1)
        out["val_epoch_cal"] = int(np.unravel_index(mean.argmax(), mean.shape)[0]) + 1
        out["val_epoch_g0"] = int(per_epoch.argmax()) + 1
        if n_splits > 1:  # split 1 alone: comparable with the 1-split runs (same seed-42 split)
            s1 = grid[0].reshape(-1, grid.shape[-1])
            out["val_split1_hm_cal_same_epoch"] = float(s1[sel_row, sel_g])
            out["val_split1_hm_g0_same_epoch"] = float(s1[sel_row, i0])
    stored = (metrics.get("val") or {}).get("val_hm_g0_mean")
    if stored is not None:
        out["val_hm_g0_stored"] = float(stored)
    return out


def test_numbers(run: Path, metrics: dict, pairs: pd.DataFrame) -> dict:
    z = np.load(run / "scores.npz")
    ss, ys, su, yu = z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"]
    gamma = float(metrics.get("gamma", 0.0))
    cal = evaluate(ss, ys, su, yu, pairs, gamma)
    g0 = evaluate(ss, ys, su, yu, pairs, 0.0)
    unseen = pairs["is_unseen"].to_numpy()
    seen_ids, unseen_ids = np.flatnonzero(~unseen), np.flatnonzero(unseen)
    # Gamma-free views: rank each split only among its own pairs (what the representation knows, minus the seen/unseen bias).
    closed_seen = float((seen_ids[ss[:, seen_ids].argmax(1)] == ys).mean())
    closed_unseen = float(cal["diagnostic_unseen_only_top1"])
    curve = seen_unseen_curve(ss, ys, su, yu, pairs)
    hm_curve = np.where(curve.seen + curve.unseen > 0, 2 * curve.seen * curve.unseen / (curve.seen + curve.unseen).clip(1e-12), 0)
    # Seen bias: on unseen-test images, how far the best seen score sits above the best unseen score (median), the gap gamma must close.
    gap_u = su[:, seen_ids].max(1) - su[:, unseen_ids].max(1)
    gap_s = ss[:, seen_ids].max(1) - ss[:, unseen_ids].max(1)
    row = {
        "gamma": gamma,
        "hm_cal": cal["harmonic_mean_top1"], "seen_cal": cal["test_seen"]["top1"], "unseen_cal": cal["test_unseen"]["top1"],
        "hm5_cal": cal["harmonic_mean_top5"],
        "hm_g0": g0["harmonic_mean_top1"], "seen_g0": g0["test_seen"]["top1"], "unseen_g0": g0["test_unseen"]["top1"],
        "hm5_g0": g0["harmonic_mean_top5"], "seen5_g0": g0["test_seen"]["top5"], "unseen5_g0": g0["test_unseen"]["top5"],
        "seen_crop_g0": g0["test_seen"]["crop_top1"], "seen_disease_g0": g0["test_seen"]["disease_top1"],
        "unseen_crop_g0": g0["test_unseen"]["crop_top1"], "unseen_disease_g0": g0["test_unseen"]["disease_top1"],
        "seen_to_unseen_g0": g0["test_seen"]["predicted_as_unseen"], "unseen_to_unseen_g0": g0["test_unseen"]["predicted_as_unseen"],
        "seen_to_unseen_cal": cal["test_seen"]["predicted_as_unseen"], "unseen_to_unseen_cal": cal["test_unseen"]["predicted_as_unseen"],
        "closed_seen_top1": closed_seen, "closed_unseen_top1": closed_unseen,
        "ausuc": ausuc(curve), "hm_test_oracle": float(hm_curve.max()),
        "bias_gap_unseen_median": float(np.median(gap_u)), "bias_gap_seen_median": float(np.median(gap_s)),
    }
    stored = metrics.get("harmonic_mean_top1")
    row["check_hm_cal_matches_metrics"] = stored is None or abs(stored - row["hm_cal"]) < 1e-6
    return row


def main():
    pairs = pair_table(read_metadata())
    rows = []
    for run in sorted(p for p in RUNS.glob("*/*") if p.is_dir()):
        if run.parent.name == "cache" or not (run / "scores.npz").exists() or not (run / "metrics.json").exists():
            continue
        metrics = json.loads((run / "metrics.json").read_text())
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        name = cfg.get("version") or run.name
        row = {"run": name, "family": run.parent.name, "title": cfg.get("title", ""), "method": cfg.get("method", ""),
               "copied_from": metrics.get("copied_from", ""), "parent": PARENT.get(name, "")}
        row.update(test_numbers(run, metrics, pairs))
        row.update(val_numbers(run, cfg, metrics))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gamma0_audit.csv", index=False, float_format="%.4f")

    by = df.set_index("run")
    steps = []
    for r in df.itertuples():
        if not r.parent or r.parent not in by.index:
            continue
        p = by.loc[r.parent]
        d = {"step": f"{r.parent} -> {r.run}", "title": r.title}
        for col in ("val_hm_cal", "val_hm_g0_same_epoch", "val_hm_g0_best", "hm_cal", "hm_g0", "seen_g0", "unseen_g0",
                    "closed_seen_top1", "closed_unseen_top1", "ausuc", "bias_gap_unseen_median"):
            a, b = getattr(r, col, np.nan), p.get(col, np.nan)
            d[f"d_{col}"] = a - b if pd.notna(a) and pd.notna(b) else np.nan
        steps.append(d)
    pd.DataFrame(steps).to_csv(OUT / "gamma0_audit_deltas.csv", index=False, float_format="%.4f")
    print(df.to_string(max_colwidth=18))


if __name__ == "__main__" and "--val-members" not in __import__("sys").argv:
    main()


def val_member_numbers():
    """Gamma-free metrics on VALIDATION for every member whose validation scores an ensemble cached (val_scores_*.npy).

    These are the only validation score matrices on disk; they let the gamma-free comparison be made without test data.
    """
    from dpd.ensemble import to_logprob
    from dpd.splits import make_validation_split

    df = read_metadata()
    pairs = pair_table(df)
    train = df[df["split"] == "train"]
    rows, seen = [], set()
    for ens in sorted(RUNS.glob("vlm/v6*")):
        cfg = json.loads((ens / "config.json").read_text())
        v = cfg["validation"]
        key = (cfg["seed"], v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
               tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
        fit_df, vs, vu, _ = make_validation_split(train, *key[1:4], cfg["seed"], key[4], key[5])
        n_vs = len(vs)
        y = np.concatenate([vs["pair_id"].to_numpy(), vu["pair_id"].to_numpy()])
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        hidden = pairs["is_unseen"].to_numpy()
        pseudo = val_pairs["is_unseen"].to_numpy() & ~hidden
        trained = ~val_pairs["is_unseen"].to_numpy()
        for f in sorted(ens.glob("val_scores_*.npy")):
            member = f.stem.removeprefix("val_scores_")
            if (member, key) in seen:
                continue
            seen.add((member, key))
            raw = np.load(f)
            run_name = member.split("_w0.5")[0]
            run_dir = next(d for d in RUNS.glob(f"*/{run_name}") if d.is_dir())
            method = json.loads((run_dir / "config.json").read_text())["method"]
            s = to_logprob(raw, method).copy()
            s[:, hidden] = -1e9
            ss, su, ys, yu = s[:n_vs], s[n_vs:], y[:n_vs], y[n_vs:]
            g0 = evaluate(ss, ys, su, yu, val_pairs, 0.0, (1,))
            curve = seen_unseen_curve(ss, ys, su, yu, val_pairs)
            hm = np.where(curve.seen + curve.unseen > 0, 2 * curve.seen * curve.unseen / (curve.seen + curve.unseen).clip(1e-12), 0)
            tid, pid = np.flatnonzero(trained), np.flatnonzero(pseudo)
            rows.append({
                "member": member, "split_seed": cfg["seed"], "n_val_seen": n_vs, "n_val_pseudo_unseen": len(yu),
                "val_hm_g0": g0["harmonic_mean_top1"], "val_seen_g0": g0["test_seen"]["top1"], "val_unseen_g0": g0["test_unseen"]["top1"],
                "val_hm_best_gamma": float(hm.max()),
                "val_closed_seen_top1": float((tid[ss[:, tid].argmax(1)] == ys).mean()),
                "val_closed_unseen_top1": float((pid[su[:, pid].argmax(1)] == yu).mean()),
                "val_ausuc": ausuc(curve),
                "val_bias_gap_unseen_median": float(np.median(su[:, tid].max(1) - su[:, pid].max(1))),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gamma0_audit_val_members.csv", index=False, float_format="%.4f")
    print(out.round(4).to_string())


if __name__ == "__main__" and "--val-members" in __import__("sys").argv:
    val_member_numbers()
