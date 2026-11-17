"""Mix of the four resolution-336 models (control and same-crop mixing, seeds 42 and 45) with the team's v10.3 (CLIP with LoRA),
retrained here from its unchanged config. Rule fixed BEFORE any test read (2026-11-16):
  both sides become log-probabilities over the 333 pairs (train/ensemble.py's convention):
    ConvNeXt side: log_softmax over pairs of 1.75 * mean log P(crop) + mean log P(disease), mean over the four 336 models
    CLIP side:     log_softmax(100 * cosine) of v10.3
  score = a * CLIP + (1 - a) * ConvNeXt, gamma subtracted from seen pairs.
  a (0 to 1, step 0.05) and gamma (0 to 15, step 0.25) are chosen on the mean validation HM of the three validation splits
  (seeds 42, 43, 44). Validation uses the seed-42 models of the ConvNeXt side (control and same-crop mixing at 336) at their
  selected epochs, because only they share those splits with v10.3; the chosen (a, gamma) is applied to all four on test.
  The mix is adopted only if its validation HM beats the ConvNeXt side alone (a = 0, its own best gamma) by at least 0.5 points.
  Test is read once, with the chosen (a, gamma), only if that holds.
Usage: python mix_v103.py val      (CPU, prints the validation choice, never reads test)
       python mix_v103.py test A G (one test read with the given a and gamma)"""
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
R = REPO / "runs" / "contrastive"
V = REPO / "runs" / "vlm" / "v10.3"
W = 1.75
AS = np.round(np.arange(0.0, 1.0001, 0.05), 6)
GAM = np.round(np.arange(0.0, 15.0, 0.25), 6)
VAL_RUNS = ["full_res336", "full_scmix_res336"]
TEST_RUNS = ["full_res336", "full_res336_s45", "full_scmix_res336", "full_scmix_res336_s45"]


def hm(a, u):
    return 2 * a * u / (a + u + 1e-12)


def cnn_logp(lc, ld):
    return to_logprob(W * lc[:, cid] + ld[:, did], "cnn_independent")


def validate():
    grid = np.zeros((len(AS), len(GAM)))
    for k in (1, 2, 3):
        parts = [np.load(R / r / f"val_parts_split{k}.npz") for r in VAL_RUNS]
        eps = [json.load(open(R / r / "metrics.json"))["selected_epoch"] for r in VAL_RUNS]
        z = parts[0]
        for p in parts:
            assert (p["y_val_seen"] == z["y_val_seen"]).all() and (p["y_val_unseen"] == z["y_val_unseen"]).all()
        lc = np.mean([p["logp_crop"][e - 1] for p, e in zip(parts, eps)], 0)
        ld = np.mean([p["logp_disease"][e - 1] for p, e in zip(parts, eps)], 0)
        cnn = cnn_logp(lc, ld)
        clip = to_logprob(np.load(V / f"val_scores_split{k}.npy"), "clip_prompt_lora")
        fit = np.isin(pid, z["fit_pairs"])
        ys = np.array([idx[p] for p in z["y_val_seen"]])
        yu = np.array([idx[p] for p in z["y_val_unseen"]])
        n = len(ys)
        for i, a in enumerate(AS):
            S0 = a * clip + (1 - a) * cnn
            S0[:, un] = -1e9
            for j, g in enumerate(GAM):
                pred = (S0 - fit * g).argmax(1)
                grid[i, j] += hm((pred[:n] == ys).mean(), (pred[n:] == yu).mean()) / 3
    grid *= 100
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    base = grid[0].max()
    print(f"ConvNeXt side alone (a 0): best validation HM {base:.2f} at gamma {GAM[grid[0].argmax()]}")
    print(f"v10.3 alone (a 1):         best validation HM {grid[-1].max():.2f} at gamma {GAM[grid[-1].argmax()]}")
    print(f"BEST MIX: a {AS[i]:.2f}, gamma {GAM[j]}: validation HM {grid[i, j]:.2f}  (gain over a 0: {grid[i, j] - base:+.2f}; rule needs >= +0.50)")
    print("best validation HM by a:", {float(a): round(float(grid[k].max()), 2) for k, a in enumerate(AS) if k % 2 == 0})
    return AS[i], GAM[j]


def test(a, g):
    ts = [np.load(R / r / "test_parts.npz") for r in TEST_RUNS]
    cl = np.load(REPO / "runs" / "vlm" / "v10.3" / "scores.npz")
    out = {}
    for s in ("test_seen", "test_unseen"):
        for t in ts:
            assert (t[f"{s}_labels"] == ts[0][f"{s}_labels"]).all()
        assert (cl[f"{s}_labels"] == ts[0][f"{s}_labels"]).all()
        cnn = cnn_logp(np.mean([t[f"{s}_logp_crop"] for t in ts], 0), np.mean([t[f"{s}_logp_disease"] for t in ts], 0))
        out[s] = a * to_logprob(cl[f"{s}_scores"], "clip_prompt_lora") + (1 - a) * cnn
    res = {}
    for gg in (0.0, g):
        r = evaluate(out["test_seen"], ts[0]["test_seen_labels"], out["test_unseen"], ts[0]["test_unseen_labels"], pairs, gg)
        res[str(gg)] = r
        print(f"MIX test, a {a}, gamma {gg}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} "
              f"seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
    json.dump({"a": a, "gamma": g, "members": TEST_RUNS + ["runs/vlm/v10.3"], "results": res},
              open(R / "mix_v103_metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test(float(sys.argv[2]), float(sys.argv[3]))
