"""Ensemble of saved final models (no training). Rule fixed BEFORE reading test:
- PRIMARY = all 4 models (control s42, control s45, same-crop mixing s42, s45), equal weights, average of the members'
  raw pair scores (log P(crop) + log P(disease)); gamma = mean of the members' validation-chosen gammas.
- Sub-ensembles (2 controls, 2 mixing models) are printed as diagnostics only and are not used to choose anything.
Compare with: control 70.71 (s42) / 68.85 (s45), independent ConvNeXt-Base 71.44, team #1 ensemble 74.99."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402

pairs = pair_table(read_metadata())
R = REPO / "runs" / "contrastive"
MEMBERS = {"control_s42": "supcon_ce_convnext_base_control_no_supcon", "control_s45": "control_full_s45",
           "scmix_s42": "full_scmix", "scmix_s45": "full_scmix_s45"}
Z = {k: np.load(R / v / "scores.npz") for k, v in MEMBERS.items()}
G = {k: json.load(open(R / v / "metrics.json"))["gamma"] for k, v in MEMBERS.items()}
for k in Z:  # same images in the same order
    assert (Z[k]["test_seen_labels"] == Z["control_s42"]["test_seen_labels"]).all()
    assert (Z[k]["test_unseen_labels"] == Z["control_s42"]["test_unseen_labels"]).all()


def ens(keys):
    s = np.mean([Z[k]["test_seen_scores"] for k in keys], 0)
    u = np.mean([Z[k]["test_unseen_scores"] for k in keys], 0)
    g = float(np.mean([G[k] for k in keys]))
    r = evaluate(s, Z[keys[0]]["test_seen_labels"], u, Z[keys[0]]["test_unseen_labels"], pairs, gamma=g)
    return r, g


def show(name, keys):
    r, g = ens(keys)
    print(f"{name:34s} gamma {g:.4g}  HM {r['harmonic_mean_top1']*100:.2f}  top5 {r['harmonic_mean_top5']*100:.2f}  "
          f"seen {r['test_seen']['top1']*100:.1f}  unseen {r['test_unseen']['top1']*100:.1f}")
    return r, g


print("members:")
for k in Z:
    show(k, [k])
print("\nPRIMARY (fixed before looking):")
r, g = show("4-model ensemble", list(Z))
json.dump({"members": MEMBERS, "gamma": g, "harmonic_mean_top1": r["harmonic_mean_top1"], "harmonic_mean_top5": r["harmonic_mean_top5"],
           "test_seen": r["test_seen"], "test_unseen": r["test_unseen"]},
          open(REPO / "runs" / "contrastive" / "ensemble4_metrics.json", "w"), indent=2)
print("\ndiagnostics only:")
show("2 controls", ["control_s42", "control_s45"])
show("2 mixing", ["scmix_s42", "scmix_s45"])
