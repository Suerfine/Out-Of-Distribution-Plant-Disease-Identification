"""Write the standard report files (dpd/report.py: metrics.json, confusion_test_seen.csv, predictions_<split>.csv, scores.npz)
and config.json for the final mix described by config/contrastive/mix_res336_v103.json. The mix weight and gamma in that config
were chosen on validation by mix_v103.py; no choice is made here, and the result must reproduce runs/contrastive/mix_v103_metrics.json.
With config.json in place the Streamlit app lists the mix (method supcon_mix, dpd/inference.py SupconMixRunModel).
members.clip may be one run or a list (averaged in log-probabilities, as mix_v103_two_seeds.py does).
Usage (from the repo root, DPD_ROOT set):  python train/supcon_eval/mix_v103_report.py [config json]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.ensemble import to_logprob  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402
from dpd.report import save_test_outputs  # noqa: E402

CONFIG = REPO / (sys.argv[1] if len(sys.argv) > 1 else "config/contrastive/mix_res336_v103.json")
cfg = json.load(open(CONFIG, encoding="utf-8"))
A, W, G = cfg["ensemble"]["clip_weight"], cfg["ensemble"]["crop_weight"], cfg["ensemble"]["gamma"]
out = REPO / cfg["output_dir"]
out.mkdir(parents=True, exist_ok=True)
df = read_metadata()
pairs = pair_table(df)
cid, did = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
ts = [np.load(REPO / r / "test_parts.npz") for r in cfg["members"]["convnext"]]
clip_paths = [cfg["members"]["clip"]] if isinstance(cfg["members"]["clip"], str) else cfg["members"]["clip"]
cls = [np.load(REPO / p / "scores.npz") for p in clip_paths]
scores, labels = {}, {}
for s in ("test_seen", "test_unseen"):
    lc, ld = np.mean([t[f"{s}_logp_crop"] for t in ts], 0), np.mean([t[f"{s}_logp_disease"] for t in ts], 0)
    cnn = to_logprob(W * lc[:, cid] + ld[:, did], "cnn_independent")
    scores[s] = A * np.mean([to_logprob(c[f"{s}_scores"], "clip_prompt_lora") for c in cls], 0) + (1 - A) * cnn
    labels[s] = ts[0][f"{s}_labels"]
    assert all((t[f"{s}_labels"] == labels[s]).all() for t in ts) and all((c[f"{s}_labels"] == labels[s]).all() for c in cls)
    assert (df[df["split"] == s]["pair_id"].to_numpy() == labels[s]).all()
files = {s: df[df["split"] == s]["file"].to_numpy() for s in scores}
top_k = tuple(cfg["evaluation"]["top_k"])
res = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, G, top_k)
g0 = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
ref = json.load(open(REPO / cfg["ensemble"]["metrics_file"]))["results"][str(G)]["harmonic_mean_top1"]
assert abs(res["harmonic_mean_top1"] - ref) < 1e-9, (res["harmonic_mean_top1"], ref)
res.update(version=cfg["version"], gamma=G, weight=A, crop_weight=W, members=cfg["members"]["convnext"] + clip_paths,
           val={"weight": A, "gamma": G, **cfg["ensemble"]["validation_result"]},
           harmonic_mean_top1_gamma0=g0["harmonic_mean_top1"])
save_test_outputs(out, res, scores, labels, files, pairs, G)
shutil.copy(CONFIG, out / "config.json")
print(f"HM {res['harmonic_mean_top1']*100:.2f} (seen {res['test_seen']['top1']*100:.1f}, unseen {res['test_unseen']['top1']*100:.1f}); files:",
      sorted(p.name for p in out.iterdir()))
