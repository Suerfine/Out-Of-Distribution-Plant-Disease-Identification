"""10-model ensemble. Rule fixed BEFORE reading test (2026-11-13 night): ALL ten saved full-protocol models of the two-seed
runs (control, same-crop mixing 0.5, same-crop mixing 0.25, taxonomy heads, SAM; seeds 42 and 45), equal weights, mean
log-probs per head, crop weight 1.75 and gamma 7.875 as already fixed for the 4-model ensemble. One test read. If it does
not beat the 4-model ensemble (75.03) the 4-model ensemble stays."""
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
RUNS = ["supcon_ce_convnext_base_control_no_supcon", "control_full_s45", "full_scmix", "full_scmix_s45", "full_scmix_p25",
        "full_scmix_p25_s45", "full_taxonomy", "full_taxonomy_s45", "full_sam", "full_sam_s45"]
W, G = 1.75, 7.875
t = [np.load(R / r / "test_parts.npz") for r in RUNS]
for x in t:
    assert (x["test_seen_labels"] == t[0]["test_seen_labels"]).all() and (x["test_unseen_labels"] == t[0]["test_unseen_labels"]).all()
S = {s: W * np.mean([x[f"{s}_logp_crop"] for x in t], 0)[:, cid] + np.mean([x[f"{s}_logp_disease"] for x in t], 0)[:, did] for s in ("test_seen", "test_unseen")}
out = {}
for g in (0.0, G):
    r = evaluate(S["test_seen"], t[0]["test_seen_labels"], S["test_unseen"], t[0]["test_unseen_labels"], pairs, g); out[str(g)] = r
    print(f"10 models, w {W} gamma {g}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
json.dump({"crop_weight": W, "gamma": G, "members": RUNS, "results": out}, open(REPO / "runs" / "contrastive" / "ensemble10_weighted_metrics.json", "w"), indent=2, default=float)
