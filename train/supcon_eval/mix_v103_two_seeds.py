"""Mix of the four resolution-336 ConvNeXt models with v10.3 on TWO seeds (42 and 45). Rule fixed BEFORE the test read (2026-11-16):
  ConvNeXt side: unchanged from mix_v103.py (four 336 models, crop weight 1.75; the 384 runs failed the both-seeds rule and are not used).
  CLIP side: mean of the two v10.3 runs' log-probabilities over the 333 pairs (runs/vlm/v10.3 and runs/vlm/v10.3_s45).
  score = a * CLIP + (1 - a) * ConvNeXt, gamma subtracted from seen pairs.
  a (0 to 1, step 0.05) and gamma (0 to 15, step 0.25) are chosen on the mean validation HM over SIX splits: seeds 42, 43, 44 with the
  seed-42 ConvNeXt models and v10.3, and seeds 45, 46, 47 with the seed-45 ConvNeXt models and v10.3_s45 (each split uses the models
  that were trained without its validation images). This is the planned completion of the mix (both seeds on both sides, weight chosen
  with all four ConvNeXt models); it becomes the final mix whether its test score is above or below the one-seed mix (76.38), and
  both are reported. Test is read once.
Usage (repo root, DPD_ROOT set): python train/supcon_eval/mix_v103_two_seeds.py val | test A G"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.ensemble import to_logprob  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402

pairs = pair_table(read_metadata())
cid, did, un = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(), pairs["is_unseen"].to_numpy()
pid = pairs["pair_id"].to_numpy()
idx = {p: i for i, p in enumerate(pid)}
R, V = REPO / "runs" / "contrastive", REPO / "runs" / "vlm"
W = 1.75
AS = np.round(np.arange(0.0, 1.0001, 0.05), 6)
GAM = np.round(np.arange(0.0, 15.0, 0.25), 6)
SIDES = [(["full_res336", "full_scmix_res336"], "v10.3"), (["full_res336_s45", "full_scmix_res336_s45"], "v10.3_s45")]
TEST_RUNS = ["full_res336", "full_res336_s45", "full_scmix_res336", "full_scmix_res336_s45"]
CLIPS = ["v10.3", "v10.3_s45"]


def hm(a, u):
    return 2 * a * u / (a + u + 1e-12)


def cnn_logp(lc, ld):
    return to_logprob(W * lc[:, cid] + ld[:, did], "cnn_independent")


def validate():
    grids = []
    for runs, clip in SIDES:
        eps = [json.load(open(R / r / "metrics.json"))["selected_epoch"] for r in runs]
        for k in (1, 2, 3):
            parts = [np.load(R / r / f"val_parts_split{k}.npz") for r in runs]
            z = parts[0]
            assert all((p["y_val_seen"] == z["y_val_seen"]).all() and (p["y_val_unseen"] == z["y_val_unseen"]).all() for p in parts)
            cnn = cnn_logp(np.mean([p["logp_crop"][e - 1] for p, e in zip(parts, eps)], 0),
                           np.mean([p["logp_disease"][e - 1] for p, e in zip(parts, eps)], 0))
            cl = to_logprob(np.load(V / clip / f"val_scores_split{k}.npy"), "clip_prompt_lora")
            assert cl.shape[0] == cnn.shape[0]
            fit = np.isin(pid, z["fit_pairs"])
            ys = np.array([idx[p] for p in z["y_val_seen"]])
            yu = np.array([idx[p] for p in z["y_val_unseen"]])
            n = len(ys)
            g = np.zeros((len(AS), len(GAM)))
            for i, a in enumerate(AS):
                S0 = a * cl + (1 - a) * cnn
                S0[:, un] = -1e9
                for j, gam in enumerate(GAM):
                    pred = (S0 - fit * gam).argmax(1)
                    g[i, j] = hm((pred[:n] == ys).mean(), (pred[n:] == yu).mean()) * 100
            grids.append(g)
    grid = np.mean(grids, 0)
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    print(f"ConvNeXt side alone (a 0): best validation HM {grid[0].max():.2f} at gamma {GAM[grid[0].argmax()]}")
    print(f"CLIP alone (a 1):          best validation HM {grid[-1].max():.2f} at gamma {GAM[grid[-1].argmax()]}")
    print(f"BEST MIX over 6 splits: a {AS[i]:.2f}, gamma {GAM[j]}: validation HM {grid[i, j]:.2f} (gain over a 0: {grid[i, j] - grid[0].max():+.2f})")
    print("per split at the choice:", [round(float(g[i, j]), 2) for g in grids])
    print("one-seed choice (a 0.35, gamma 7.75) on these 6 splits:", round(float(grid[list(AS).index(0.35), list(GAM).index(7.75)]), 2))
    print("best validation HM by a:", {float(a): round(float(grid[k].max()), 2) for k, a in enumerate(AS) if k % 2 == 0})


def test(a, g):
    ts = [np.load(R / r / "test_parts.npz") for r in TEST_RUNS]
    cls = [np.load(V / c / "scores.npz") for c in CLIPS]
    out = {}
    for s in ("test_seen", "test_unseen"):
        assert all((t[f"{s}_labels"] == ts[0][f"{s}_labels"]).all() for t in ts) and all((c[f"{s}_labels"] == ts[0][f"{s}_labels"]).all() for c in cls)
        cnn = cnn_logp(np.mean([t[f"{s}_logp_crop"] for t in ts], 0), np.mean([t[f"{s}_logp_disease"] for t in ts], 0))
        clip = np.mean([to_logprob(c[f"{s}_scores"], "clip_prompt_lora") for c in cls], 0)
        out[s] = a * clip + (1 - a) * cnn
    res = {}
    for gg in (0.0, g):
        r = evaluate(out["test_seen"], ts[0]["test_seen_labels"], out["test_unseen"], ts[0]["test_unseen_labels"], pairs, gg, (1, 5))
        res[str(gg)] = r
        print(f"TWO-SEED MIX test, a {a}, gamma {gg}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} "
              f"seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
    json.dump({"a": a, "gamma": g, "members": TEST_RUNS + [f"runs/vlm/{c}" for c in CLIPS], "results": res},
              open(R / "mix_v103_two_seeds_metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test(float(sys.argv[2]), float(sys.argv[3]))
