"""Final mix + CLIP disease-only evidence (2026-11-16 night). Rule fixed BEFORE any result:
  ConvNeXt side: w log P_cnn(crop) + (1 - b) log P_cnn(disease) + b log P_clip(disease), w = 1.75 as before, where
    log P_clip(disease) = log_softmax over the 175 diseases of 100 x cosine(image, v10.3's learned disease prompt).
  Then as the two-seed mix: both sides to log-probabilities over the 333 pairs, a x CLIP pair side + (1 - a) x ConvNeXt side,
  gamma subtracted from seen pairs.
  b in {0, 0.1, ..., 1}, a in {0, 0.05, ..., 1}, gamma in {0, 0.25, ..., 14.75}, chosen on the mean validation HM of six splits
  (seeds 42-44: the seed-42 ConvNeXt models + v10.3 retrained per split; seeds 45-47: the seed-45 models + v10.3_s45 retrained per
  split). b = 0 is the current final mix. Adopted only if the best validation HM beats the best with b = 0 by at least 0.5;
  then the test set is read once. On test the CLIP terms are the means of the two v10.3 runs.
Usage (repo root, DPD_ROOT set): python train/supcon_eval/mix_clip_disease.py val | test B A G"""
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
CID, DID, UN = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(), pairs["is_unseen"].to_numpy()
PID = pairs["pair_id"].to_numpy()
IDX = {p: i for i, p in enumerate(PID)}
R, V = REPO / "runs" / "contrastive", REPO / "runs" / "vlm"
W = 1.75
BS = np.round(np.arange(0.0, 1.0001, 0.1), 6)
AS = np.round(np.arange(0.0, 1.0001, 0.05), 6)
GAM = np.round(np.arange(0.0, 15.0, 0.25), 6)
SIDES = [(["full_res336", "full_scmix_res336"], "v10.3"), (["full_res336_s45", "full_scmix_res336_s45"], "v10.3_s45")]
TEST_RUNS = ["full_res336", "full_res336_s45", "full_scmix_res336", "full_scmix_res336_s45"]


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def hm(a, u):
    return 2 * a * u / (a + u + 1e-12)


def cnn_side(lc, ld, ldc, b):
    return to_logprob(W * lc[:, CID] + (1 - b) * ld[:, DID] + b * ldc[:, DID], "cnn_independent")


def gamma_sweep(S, seen_mask, cand_unseen, ys, yu, n):
    """Seen / unseen accuracy for every gamma at once: an image goes to its best seen pair unless its best candidate unseen
    pair beats it after the penalty."""
    Ss, Su = np.where(seen_mask, S, -np.inf), np.where(cand_unseen, S, -np.inf)
    bs, bu = Ss.argmax(1), Su.argmax(1)
    vs, vu = Ss.max(1), Su.max(1)
    y = np.concatenate([ys, yu])
    pred = np.where(vs[None, :] - GAM[:, None] >= vu[None, :], bs[None, :], bu[None, :])
    hit = pred == y[None, :]
    return hit[:, :n].mean(1), hit[:, n:].mean(1)


def validate():
    grids, g0 = [], []
    for runs, clip in SIDES:
        eps = [json.load(open(R / r / "metrics.json"))["selected_epoch"] for r in runs]
        for k in (1, 2, 3):
            parts = [np.load(R / r / f"val_parts_split{k}.npz") for r in runs]
            c = np.load(V / clip / f"val_primitives_split{k}.npz")
            z = parts[0]
            assert all((p["y_val_seen"] == c["y_val_seen"]).all() and (p["y_val_unseen"] == c["y_val_unseen"]).all() for p in parts)
            lc = np.mean([p["logp_crop"][e - 1] for p, e in zip(parts, eps)], 0)
            ld = np.mean([p["logp_disease"][e - 1] for p, e in zip(parts, eps)], 0)
            ldc = lsm(100 * c["dis"])
            cp = to_logprob(c["pair"], "clip_prompt_lora")
            seen_mask = np.isin(PID, z["fit_pairs"])
            cand_unseen = ~seen_mask & ~UN  # pseudo-unseen candidates (the real unseen pairs are out of the label space)
            ys = np.array([IDX[p] for p in z["y_val_seen"]])
            yu = np.array([IDX[p] for p in z["y_val_unseen"]])
            n = len(ys)
            g = np.zeros((len(BS), len(AS), len(GAM)))
            for i, b in enumerate(BS):
                cs = cnn_side(lc, ld, ldc, b)
                for j, a in enumerate(AS):
                    acc_s, acc_u = gamma_sweep(a * cp + (1 - a) * cs, seen_mask, cand_unseen, ys, yu, n)
                    g[i, j] = hm(acc_s, acc_u) * 100
            grids.append(g)
            dis_u = (ldc[n:].argmax(1) == DID[yu]).mean() * 100
            dis_cnn = (ld[n:].argmax(1) == DID[yu]).mean() * 100
            g0.append((dis_u, dis_cnn))
    grid = np.mean(grids, 0)
    print("disease named right on pseudo-unseen validation images, CLIP disease-only vs ConvNeXt disease head, per split:",
          [(round(a, 1), round(b, 1)) for a, b in g0])
    for i, b in enumerate(BS):
        j, k = np.unravel_index(grid[i].argmax(), grid[i].shape)
        print(f"  b {b:.1f}: best validation HM {grid[i, j, k]:.2f} at a {AS[j]:.2f}, gamma {GAM[k]}; HM at gamma 0 with that a: {grid[i, j, 0]:.2f}")
    i, j, k = np.unravel_index(grid.argmax(), grid.shape)
    base = grid[0].max()
    print(f"BEST: b {BS[i]:.1f}, a {AS[j]:.2f}, gamma {GAM[k]}: validation HM {grid[i, j, k]:.2f}; b = 0 (current mix) {base:.2f}; "
          f"gain {grid[i, j, k] - base:+.2f} (rule needs >= +0.50) -> {'ADOPT, read test once' if grid[i, j, k] - base >= 0.5 else 'NOT ADOPTED'}")
    print("per split at the choice:", [round(float(x[i, j, k]), 2) for x in grids])


def test(b, a, g):
    ts = [np.load(R / r / "test_parts.npz") for r in TEST_RUNS]
    cl = [np.load(V / c / "test_primitives.npz") for c in ("v10.3", "v10.3_s45")]
    out = {}
    for s, tag in (("test_seen", "ts"), ("test_unseen", "tu")):
        assert all((t[f"{s}_labels"] == ts[0][f"{s}_labels"]).all() for t in ts) and all((c[f"y_{tag}"] == ts[0][f"{s}_labels"]).all() for c in cl)
        lc, ld = np.mean([t[f"{s}_logp_crop"] for t in ts], 0), np.mean([t[f"{s}_logp_disease"] for t in ts], 0)
        ldc = np.mean([lsm(100 * c[f"dis_{tag}"]) for c in cl], 0)
        cp = np.mean([to_logprob(c[f"pair_{tag}"], "clip_prompt_lora") for c in cl], 0)
        out[s] = a * cp + (1 - a) * cnn_side(lc, ld, ldc, b)
        if s == "test_unseen":
            y = np.array([IDX[p] for p in ts[0][f"{s}_labels"]])
            print(f"test_unseen: disease named right by CLIP disease-only {(ldc.argmax(1) == DID[y]).mean() * 100:.1f}%, "
                  f"by the ConvNeXt disease heads {(ld.argmax(1) == DID[y]).mean() * 100:.1f}%")
    res = {}
    for gg in (0.0, g):
        r = evaluate(out["test_seen"], ts[0]["test_seen_labels"], out["test_unseen"], ts[0]["test_unseen_labels"], pairs, gg, (1, 5))
        res[str(gg)] = r
        print(f"MIX + CLIP disease (b {b}, a {a}) gamma {gg}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} "
              f"seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
    json.dump({"b": b, "a": a, "gamma": g, "results": res}, open(R / "mix_clip_disease_metrics.json", "w"), indent=2, default=float)
    np.savez_compressed(R / "mix_clip_disease_scores.npz", test_seen_scores=out["test_seen"], test_unseen_scores=out["test_unseen"],
                        test_seen_labels=ts[0]["test_seen_labels"], test_unseen_labels=ts[0]["test_unseen_labels"])


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
