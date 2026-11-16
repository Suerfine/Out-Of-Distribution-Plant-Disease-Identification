"""4-model ensemble with the crop weight. Rule (the pre-fixed gamma rule of ensemble_val.py extended to the weight, but
written AFTER the per-seed test numbers of queue 9 were seen - disclose): w and gamma = mean of the two per-seed
ensembles' validation-chosen variant-B values (s42: w 2.0 gamma 8.0; s45: w 1.5 gamma 7.75). One test evaluation."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402

pairs = pair_table(read_metadata())
cid, did = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
R = REPO / "runs" / "contrastive"
RUNS = ["supcon_ce_convnext_base_control_no_supcon", "control_full_s45", "full_scmix", "full_scmix_s45"]
W, G = (2.0 + 1.5) / 2, (8.0 + 7.75) / 2
t = [np.load(R / r / "test_parts.npz") for r in RUNS]
S = {s: W * np.mean([x[f"{s}_logp_crop"] for x in t], 0)[:, cid] + np.mean([x[f"{s}_logp_disease"] for x in t], 0)[:, did]
     for s in ("test_seen", "test_unseen")}
out = {}
for g in (0.0, G):
    r = evaluate(S["test_seen"], t[0]["test_seen_labels"], S["test_unseen"], t[0]["test_unseen_labels"], pairs, g)
    out[str(g)] = r
    print(f"w {W} gamma {g}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
json.dump({"crop_weight": W, "gamma": G, "members": RUNS, "results": out}, open(REPO / "runs" / "contrastive" / "ensemble4_weighted_metrics.json", "w"), indent=2, default=float)
