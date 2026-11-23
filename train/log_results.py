"""Append a finished run's uncalibrated (gamma 0) and calibrated test rows to docs/all_results.csv (AGENTS.md rule).

Usage:  python -m train.log_results runs/vlm2/s4_final [--type clip_prompt_lora] [--name vlm2_s4_final]

Reads the run's scores.npz and metrics.json, evaluates at gamma 0 and at the run's validation gamma with the shared
protocol, and appends two rows using the file's existing columns only. Refuses to add a run that is already logged.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.evaluate import evaluate

CSV = PROJECT_ROOT / "docs" / "all_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--type", help="the csv's type column (default: the config's method)")
    parser.add_argument("--name", help="experiment_name (default: <family>_<run folder>)")
    args = parser.parse_args()
    run = (PROJECT_ROOT / args.run).resolve()
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    m = json.loads((run / "metrics.json").read_text())
    name = args.name or f"{run.parent.name}_{run.name}"
    with open(CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header, logged = reader.fieldnames, {r["experiment_name"] for r in reader}
    if name in logged:
        raise SystemExit(f"{name} is already in {CSV.name}")
    pairs = pair_table(read_metadata())
    z = np.load(run / "scores.npz")
    when = datetime.fromtimestamp((run / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    config_path = next((p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "config").rglob(f"{run.name}_config.json")), "")
    val_hm = (m.get("val") or {}).get("val_hm_mean")
    rows = []
    for gamma, note in ((0.0, "Uncalibrated (gamma=0.0)"), (float(m.get("gamma", 0.0)), f"Calibrated (gamma={m.get('gamma', 0.0)} applied)")):
        r = evaluate(z["test_seen_scores"], z["test_seen_labels"], z["test_unseen_scores"], z["test_unseen_labels"], pairs, gamma)
        s, u = r["test_seen"], r["test_unseen"]
        rows.append({
            "datetime": when, "type": args.type or cfg["method"], "experiment_name": name,
            "harmonic_mean_top1": r["harmonic_mean_top1"], "harmonic_mean_top5": r["harmonic_mean_top5"],
            "seen_top_1": s["top1"], "seen_top1_class_mean": s["top1_class_mean"], "seen_top5": s["top5"],
            "seen_top5_class_mean": s["top5_class_mean"], "seen_crop_top1": s["crop_top1"], "seen_disease_top1": s["disease_top1"],
            "unseen_top_1": u["top1"], "unseen_top1_class_mean": u["top1_class_mean"], "unseen_top5": u["top5"],
            "unseen_top5_class_mean": u["top5_class_mean"], "unseen_crop_top1": u["crop_top1"], "unseen_disease_top1": u["disease_top1"],
            "config": config_path,
            "notes": f"{note}; val HM {val_hm}; epoch {m.get('selected_epoch')}",
        })
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header, lineterminator="\n").writerows({k: row.get(k, "") for k in header} for row in rows)
    print(f"appended 2 rows for {name} to {CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
