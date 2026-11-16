"""Final mix + disease evidence from the frozen-CLIP disease head with crop-anchored GAN features (clip_gan_disease.py; Hein's idea).
Rule fixed BEFORE any result (same as mix_clip_disease.py, only the disease source changes):
  ConvNeXt side: 1.75 log P_cnn(crop) + (1 - b) log P_cnn(disease) + b log P_head(disease); then a x CLIP pair side (v10.3) +
  (1 - a) x ConvNeXt side, gamma subtracted from seen pairs; b, a, gamma chosen on the mean validation HM of six splits; b = 0 is the
  current final mix; adopted only if it beats b = 0 by at least 0.5 on validation, then the test set is read once.
Usage (repo root, DPD_ROOT set): python train/supcon_eval/mix_gan_disease.py val | test B A G"""
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
H = REPO / "runs" / "clip_gan_disease"
W = 1.75
BS = np.round(np.arange(0.0, 1.0001, 0.1), 6)
AS = np.round(np.arange(0.0, 1.0001, 0.05), 6)
GAM = np.round(np.arange(0.0, 15.0, 0.25), 6)
SIDES = [(["full_res336", "full_scmix_res336"], "v10.3", 42), (["full_res336_s45", "full_scmix_res336_s45"], "v10.3_s45", 45)]
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
    for runs, clip, base in SIDES:
        eps = [json.load(open(R / r / "metrics.json"))["selected_epoch"] for r in runs]
        for k in (1, 2, 3):
            parts = [np.load(R / r / f"val_parts_split{k}.npz") for r in runs]
            c = np.load(V / clip / f"val_primitives_split{k}.npz")
            z = parts[0]
            assert all((p["y_val_seen"] == c["y_val_seen"]).all() and (p["y_val_unseen"] == c["y_val_unseen"]).all() for p in parts)
            lc = np.mean([p["logp_crop"][e - 1] for p, e in zip(parts, eps)], 0)
            ld = np.mean([p["logp_disease"][e - 1] for p, e in zip(parts, eps)], 0)
            hd = np.load(H / f"val_split_s{base + k - 1}.npz")
            assert (hd["y_val_seen"] == c["y_val_seen"]).all() and (hd["y_val_unseen"] == c["y_val_unseen"]).all()
            ldc = hd["ld"]
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
    print("disease named right on pseudo-unseen validation images, GAN disease head vs ConvNeXt disease head, per split:",
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
        hd = np.load(H / "test.npz")
        assert (hd[f"y_{tag}"] == ts[0][f"{s}_labels"]).all()
        ldc = hd[f"ld_{tag}"]
        cp = np.mean([to_logprob(c[f"pair_{tag}"], "clip_prompt_lora") for c in cl], 0)
        out[s] = a * cp + (1 - a) * cnn_side(lc, ld, ldc, b)
        if s == "test_unseen":
            y = np.array([IDX[p] for p in ts[0][f"{s}_labels"]])
            print(f"test_unseen: disease named right by the GAN disease head {(ldc.argmax(1) == DID[y]).mean() * 100:.1f}%, "
                  f"by the ConvNeXt disease heads {(ld.argmax(1) == DID[y]).mean() * 100:.1f}%")
    res = {}
    for gg in (0.0, g):
        r = evaluate(out["test_seen"], ts[0]["test_seen_labels"], out["test_unseen"], ts[0]["test_unseen_labels"], pairs, gg, (1, 5))
        res[str(gg)] = r
        print(f"MIX + GAN disease head (b {b}, a {a}) gamma {gg}: HM {r['harmonic_mean_top1']*100:.2f} top5 {r['harmonic_mean_top5']*100:.2f} "
              f"seen {r['test_seen']['top1']*100:.1f} unseen {r['test_unseen']['top1']*100:.1f}")
    json.dump({"b": b, "a": a, "gamma": g, "results": res}, open(R / "mix_gan_disease_metrics.json", "w"), indent=2, default=float)
    np.savez_compressed(R / "mix_gan_disease_scores.npz", test_seen_scores=out["test_seen"], test_unseen_scores=out["test_unseen"],
                        test_seen_labels=ts[0]["test_seen_labels"], test_unseen_labels=ts[0]["test_unseen_labels"])


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
