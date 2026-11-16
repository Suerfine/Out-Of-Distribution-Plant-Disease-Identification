"""Validation-chosen ensemble and crop/disease weighting (no training, CPU). Rule fixed BEFORE reading test:
For each seed s in (42, 45): ensemble E_s = control_s + same-crop-mixing_s (equal weights, mean log-probs per head).
On that seed's 3 validation splits (same splits for both members, each member at its own selected epoch), choose
  A) gamma only (crop weight 1), and B) crop weight w in {0.5 .. 2.0} together with gamma,
by the mean validation HM. Then apply to test once. Also B for each single control (weighting alone, no ensemble).
Headline: all 4 models with gamma = mean of the two per-seed ensembles' variant-A gammas.
Validation = the stage-1 reruns with validation.min_other_crops 2 (pseudo-unseen pairs matched to the real ones).
Pass rule (as before): a variant replaces the control only if it beats it on test on BOTH seeds
(control 70.71 on seed 42, 68.85 on seed 45). Needs val_parts_split*.npz (stage-1 reruns) and test_parts.npz."""
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
unseen_real = pairs["is_unseen"].to_numpy()
R = REPO / "runs" / "contrastive"
GAMMAS = np.round(np.arange(0.0, 15.0, 0.25), 6)
WEIGHTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
SEEDS = {42: {"control": "supcon_ce_convnext_base_control_no_supcon", "scmix": "full_scmix"},
         45: {"control": "control_full_s45", "scmix": "full_scmix_s45"}}
CONTROL = {42: 0.7071, 45: 0.6885}


def compose(lc, ld, w):
    return w * lc[:, cid] + ld[:, did]


def val_parts(run, split):
    """Validation parts of the stage-1 rerun of `run` at the epoch the ORIGINAL run selected."""
    epoch = json.load(open(R / run / "metrics.json"))["selected_epoch"]
    z = np.load(R / f"val_{run}" / f"val_parts_split{split}.npz")
    return z["logp_crop"][epoch - 1], z["logp_disease"][epoch - 1], z


def val_grid(runs, split):
    lcs, lds, z = zip(*(val_parts(r, split) for r in runs))
    for zz in z[1:]:
        assert (zz["fit_pairs"] == z[0]["fit_pairs"]).all() and (zz["y_val_seen"] == z[0]["y_val_seen"]).all()
    lc, ld, z = np.mean(lcs, 0), np.mean(lds, 0), z[0]
    vp = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(z["fit_pairs"])))
    n = len(z["y_val_seen"])
    g = np.zeros((len(WEIGHTS), len(GAMMAS)))
    for i, w in enumerate(WEIGHTS):
        S = compose(lc, ld, w)
        S[:, unseen_real] = -1e9  # real unseen pairs are outside the validation label space (as in training)
        g[i] = [evaluate(S[:n], z["y_val_seen"], S[n:], z["y_val_unseen"], vp, gm, (1,))["harmonic_mean_top1"] for gm in GAMMAS]
    return g


def test_eval(runs, w, gamma):
    t = [np.load(R / r / "test_parts.npz") for r in runs]
    S = {s: compose(np.mean([x[f"{s}_logp_crop"] for x in t], 0), np.mean([x[f"{s}_logp_disease"] for x in t], 0), w)
         for s in ("test_seen", "test_unseen")}
    return evaluate(S["test_seen"], t[0]["test_seen_labels"], S["test_unseen"], t[0]["test_unseen_labels"], pairs, gamma)


rows, gA = [], {}
for seed, m in SEEDS.items():
    for label, runs in ((f"ensemble control+mixing s{seed}", [m["control"], m["scmix"]]), (f"control s{seed} alone", [m["control"]]),
                        (f"mixing s{seed} alone", [m["scmix"]])):
        try:
            grid = np.mean([val_grid(runs, k) for k in (1, 2, 3)], 0)
        except FileNotFoundError as e:
            print(f"{label}: skipped, missing {e.filename}")
            continue
        i1 = WEIGHTS.index(1.0)
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        for vname, i, j in (("A gamma only", i1, int(grid[i1].argmax())), ("B weight + gamma", int(i), int(j))):
            r = test_eval(runs, WEIGHTS[i], float(GAMMAS[j]))
            d = (r["harmonic_mean_top1"] - CONTROL[seed]) * 100
            if vname[0] == "A" and len(runs) == 2:
                gA[seed] = float(GAMMAS[j])
            print(f"{label:32s} {vname:17s} w {WEIGHTS[i]:<5} gamma {GAMMAS[j]:<5} val {grid[i, j]*100:.2f} | TEST HM "
                  f"{r['harmonic_mean_top1']*100:.2f} ({d:+.2f} vs control s{seed}) seen {r['test_seen']['top1']*100:.1f} "
                  f"unseen {r['test_unseen']['top1']*100:.1f} top5 {r['harmonic_mean_top5']*100:.2f}", flush=True)
if len(gA) == 2:  # headline: all 4 models, gamma = mean of the two per-seed ensembles' validation gammas (rule fixed beforehand)
    g = float(np.mean(list(gA.values())))
    r = test_eval([x for m in SEEDS.values() for x in m.values()], 1.0, g)
    print(f"4-MODEL ENSEMBLE (fixed validation) gamma {g:.4g} | TEST HM {r['harmonic_mean_top1']*100:.2f} seen {r['test_seen']['top1']*100:.1f} "
          f"unseen {r['test_unseen']['top1']*100:.1f} top5 {r['harmonic_mean_top5']*100:.2f}  (old validation: 71.25; baseline 71.44)")
    json.dump({"gamma": g, "harmonic_mean_top1": r["harmonic_mean_top1"], "harmonic_mean_top5": r["harmonic_mean_top5"],
               "test_seen": r["test_seen"], "test_unseen": r["test_unseen"]},
              open(REPO / "runs" / "contrastive" / "ensemble4_fixedval_metrics.json", "w"), indent=2)
