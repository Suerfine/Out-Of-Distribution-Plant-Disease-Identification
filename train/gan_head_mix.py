"""Mix the GAN disease head (train/gan_disease_head.py) into the v6 ensemble members as extra disease evidence.

score(pair) = sum_k w_k * logprob_k(pair) + b * log P_head(disease of pair)
w on the simplex (step 0.1) over the members below, b in 0..1.5 (step 0.1), gamma 0..15 (step 0.1), all chosen on the v6
validation split (seed 42). b = 0 contains every plain ensemble of these members (v6.7 included).
Rule fixed before any result: adopt only if validation HM beats v6.7 (68.66%) by more than 2 points (> 70.66%); then read
test once. Usage: [GAN_HEAD=<folder in runs/vlm>] python -m train.gan_head_mix val | test

Result (2026-11-17, validation only, test never read): not adopted with either feature source. Best validation HM is
68.79% at b = 0 (no head; 0.1 v5.13, 0.1 v5.7, 0.5 ConvNeXt-Base, 0.3 ConvNeXt-Base 336, gamma 6.5), +0.13 over v6.7;
every b > 0 is lower (b 0.1: 68.24% frozen CLIP / 68.20% LoRA; b 0.5: 62.65% / 63.99%). The head names the disease of only
5.3% (frozen OpenAI CLIP) / 6.2% (v10.3's LoRA-adapted encoder) of pseudo-unseen validation images, although every
pseudo-unseen disease occurs in training on other crops (val-seen: 72.5% / 84.2%). Hein's 45% on the real unseen test
images did not reproduce on this split with Katsuki's re-implementation.

Aligned recipe (2026-11-22, gan_disease_head.py --recipe aligned --features lora:v10.3, GAN_HEAD=gan_disease_head_aligned;
validation only, test never read): Hein's healthy-crop anchors, pair-text condition and text-initialised cosine head raise the
head's pseudo-unseen disease accuracy from 6.2% to 20.3% (val-seen 81.4%). Still not adopted: best validation HM 69.10% at
b = 0.1 (0.2 v5.13, 0.5 ConvNeXt-Base, 0.3 ConvNeXt-Base 336, gamma 6.7), +0.31 over the same sweep without the head and
+0.44 over v6.7, both inside run-to-run noise (rule needs > 70.66%); every b >= 0.2 is lower than b = 0.
Same recipe on v5.13's LoRA features (GAN_HEAD=gan_disease_head_aligned_v5.13): head pseudo-unseen disease accuracy 22.9%;
best validation HM 69.20% at b = 0.1 (0.1 v5.13, 0.1 v5.7, 0.4 / 0.4 ConvNeXt-Base 224 / 336, gamma 6.5), +0.41 over b = 0
and +0.54 over v6.7. Not adopted."""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.ensemble import to_logprob  # noqa: E402
from dpd.evaluate import evaluate  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402

V = ROOT / "runs/vlm"
HEAD = V / os.environ.get("GAN_HEAD", "gan_disease_head")  # the head's output folder (e.g. gan_disease_head_lora)
MEMBERS = {  # name: (cached validation scores, run folder with test scores.npz, method)
    "v5.13": (V / "v6.7/val_scores_v5.13.npy", V / "v5.13", "clip_prompt_lora"),
    "v5.7": (V / "v6.2/val_scores_v5.7.npy", V / "v5.7", "clip_prompt_lora"),
    "cnx224": (V / "v6.7/val_scores_aniq_baseline_independent_convnext_base_w0.5_tau0.0.npy",
               ROOT / "runs/baseline/aniq_baseline_independent_convnext_base", "cnn_independent"),
    "cnx336": (V / "v6.7/val_scores_aniq_independent_convnext_base_336_1split_w0.5_tau0.0.npy",
               ROOT / "runs/baseline/aniq_independent_convnext_base_336_1split", "cnn_independent"),
}
BS = np.round(np.arange(0.0, 1.5001, 0.1), 4)
GAMMAS = np.round(np.arange(0.0, 15.0, 0.1), 4)
BASE, MARGIN = 68.66, 2.0

df = read_metadata()
pairs = pair_table(df)
PID, DID = pairs["pair_id"].to_numpy(), pairs["disease_id"].to_numpy()
HIDDEN = pairs["is_unseen"].to_numpy()
NAMES = list(MEMBERS)


def combos():
    n = 10
    for c in itertools.product(range(n + 1), repeat=len(NAMES) - 1):
        if sum(c) <= n:
            yield np.array(c + (n - sum(c),)) / n


def validate():
    cfg = json.load(open(ROOT / "config/vlm/v6.7_three_member_ensemble_config.json"))
    v = cfg["validation"]
    fit, vs, vu, _ = make_validation_split(df[df["split"] == "train"], v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
                                           cfg["seed"], tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
    h = np.load(HEAD / "val.npz")
    assert (h["y_val_seen"] == vs["pair_id"].to_numpy()).all() and (h["y_val_unseen"] == vu["pair_id"].to_numpy()).all()
    head = h["ld"][:, DID]
    n = len(vs)
    idx = {p: i for i, p in enumerate(PID)}
    y = np.array([idx[p] for p in np.concatenate([vs["pair_id"], vu["pair_id"]])])
    val_unseen = ~np.isin(PID, fit["pair_id"].unique()) & ~HIDDEN
    val_seen = ~val_unseen & ~HIDDEN
    S = {k: to_logprob(np.load(f), m) for k, (f, _, m) in MEMBERS.items()}

    def sweep(s):
        ss, su = np.where(val_seen, s, -np.inf), np.where(val_unseen, s, -np.inf)
        pred = np.where(ss.max(1)[None] >= su.max(1)[None] + GAMMAS[:, None], ss.argmax(1)[None], su.argmax(1)[None])
        hit = pred == y[None]
        a, u = hit[:, :n].mean(1), hit[:, n:].mean(1)
        return 2 * a * u / (a + u + 1e-12) * 100, a * 100, u * 100

    # Check the fast sweep against the shared evaluate() on v6.7's choice.
    s = 0.2 * S["v5.13"] + 0.3 * S["cnx224"] + 0.5 * S["cnx336"]
    s2 = s.copy()
    s2[:, HIDDEN] = -1e9
    val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit["pair_id"])))
    ref = evaluate(s2[:n], vs["pair_id"].to_numpy(), s2[n:], vu["pair_id"].to_numpy(), val_pairs, 6.4, (1,))["harmonic_mean_top1"] * 100
    assert abs(sweep(s)[0][64] - ref) < 1e-6, (sweep(s)[0][64], ref)

    print(f"head alone, disease named right: pseudo-unseen {(h['ld'][n:].argmax(1) == DID[y[n:]]).mean():.1%}, "
          f"val-seen {(h['ld'][:n].argmax(1) == DID[y[:n]]).mean():.1%}")
    rows = []
    for w in combos():
        base = sum(wk * S[k] for wk, k in zip(w, NAMES) if wk > 0)
        for b in BS:
            hm, a, u = sweep(base + b * head)
            j = hm.argmax()
            rows.append(dict(zip(NAMES, w)) | {"b": b, "hm": hm[j], "seen": a[j], "unseen": u[j], "gamma": GAMMAS[j]})
    r = pd.DataFrame(rows).sort_values("hm", ascending=False)
    pd.set_option("display.width", 200)
    print("\nbest validation HM for each head weight b:")
    print(r.groupby("b").head(1).sort_values("b").round(2).to_string(index=False))
    best, plain = r.iloc[0], r[r.b == 0].iloc[0]
    print(f"\nBEST: {best.round(2).to_dict()}")
    print(f"best without the head (b = 0): {plain.hm:.2f}; gain from the head {best.hm - plain.hm:+.2f}; vs v6.7 {best.hm - BASE:+.2f} "
          f"-> {'ADOPT, read test once' if best.hm > BASE + MARGIN else 'NOT ADOPTED (rule needs > %.2f)' % (BASE + MARGIN)}")
    r.to_csv(HEAD / "val_search.csv", index=False)


def test(ws, b, gamma):
    t = np.load(HEAD / "test.npz")
    out = {}
    for split, tag in (("test_seen", "ts"), ("test_unseen", "tu")):
        sc = 0
        for wk, (k, (_, run, m)) in zip(ws, MEMBERS.items()):
            z = np.load(run / "scores.npz")
            assert (z[f"{split}_labels"] == t[f"y_{tag}"]).all()
            sc = sc + wk * to_logprob(z[f"{split}_scores"], m)
        out[split] = sc + b * t[f"ld_{tag}"][:, DID]
    res = {}
    for g in (0.0, gamma):
        r = evaluate(out["test_seen"], t["y_ts"], out["test_unseen"], t["y_tu"], pairs, g, (1, 5))
        res[str(g)] = r
        print(f"gamma {g}: HM {r['harmonic_mean_top1']:.2%}  seen {r['test_seen']['top1']:.2%}  unseen {r['test_unseen']['top1']:.2%}")
    json.dump({"members": NAMES, "weights": ws, "b": b, "gamma": gamma, "results": res}, open(HEAD / "test_metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "val":
        validate()
    else:
        test([float(x) for x in sys.argv[2].split(",")], float(sys.argv[3]), float(sys.argv[4]))
