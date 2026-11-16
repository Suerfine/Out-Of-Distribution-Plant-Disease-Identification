"""6-model ensemble. Rule fixed BEFORE reading test (2026-11-15): the four members of the best model (Run 23: control and
same-crop mixing 0.5, seeds 42 and 45) plus the two resolution-336 controls (seeds 42 and 45), equal weights, mean log-probs
per head, crop weight 1.75 and gamma 7.875 exactly as fixed for Run 23. One test read. If it does not beat Run 23 (75.03),
Run 23 stays."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate
pairs = pair_table(read_metadata())
cid, did = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
R = REPO / "runs" / "contrastive"
RUNS = ["supcon_ce_convnext_base_control_no_supcon", "control_full_s45", "full_scmix", "full_scmix_s45", "full_res336", "full_res336_s45"]
W, G = 1.75, 7.875
t = [np.load(R / r / "test_parts.npz") for r in RUNS]
for x in t:
    assert (x["test_seen_labels"] == t[0]["test_seen_labels"]).all() and (x["test_unseen_labels"] == t[0]["test_unseen_labels"]).all()
S = {s: W * np.mean([x[f"{s}_logp_crop"] for x in t], 0)[:, cid] + np.mean([x[f"{s}_logp_disease"] for x in t], 0)[:, did] for s in ("test_seen", "test_unseen")}
out = {}
for g in (0.0, G):
    r = evaluate(S["test_seen"], t[0]["test_seen_labels"], S["test_unseen"], t[0]["test_unseen_labels"], pairs, g); out[str(g)] = r
    print(f"6 models, w {W} gamma {g}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
json.dump({"crop_weight": W, "gamma": G, "members": RUNS, "results": out}, open(R / "ensemble6_res336_metrics.json", "w"), indent=2, default=float)
