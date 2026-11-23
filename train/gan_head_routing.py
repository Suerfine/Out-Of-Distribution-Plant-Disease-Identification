"""Score GAN disease heads (train/gan_disease_head.py) in Hein's hard-routing pipeline on the v6 validation split (seed 42).

Same equations as Disease_Transfer_Experiments/clip_anchored_gan_v1/clip_disease_only.py:
  unseen-candidate score(pair) = ConvNeXt crop logit + head disease logit, over the fit split's seen pairs + its 20
                                 pseudo-unseen pairs (the 10 real unseen pairs are excluded, as in the label space here)
  gate                          route to seen if max seen score - max pseudo-unseen score >= gamma (0..15, step 0.1)
  seen expert                   ConvNeXt crop + ConvNeXt disease over the seen pairs
  unseen expert                 the best pseudo-unseen pair by the candidate score
Log-probs are used instead of logits; that shifts every pair of an image by one constant, so no prediction changes.
ConvNeXt: ConvNeXt-Base 336 (runs/baseline/aniq_independent_convnext_base_336_1split), whose split-1 validation log-probs are
saved for exactly this split; crop epoch 8 and disease epoch 18 as it selected on validation. (Hein used ConvNeXt-Base 224.)
Arms: each head folder given, plus 'convnext' (the ConvNeXt disease head as the unseen expert's disease evidence).
Usage: python -m train.gan_head_routing gan_disease_head_aligned gan_disease_head_lora gan_disease_head

Result (2026-11-23, validation only, test never read; runs/vlm/gan_head_routing_val.json). The 'convnext' arm reproduces the
ConvNeXt-Base 336 run's own recorded validation HM (63.58% at gamma 6.6, 6.51% at 0), which checks this script.
  arm                 HM gamma 0   HM calibrated (gamma)   unseen expert alone
  convnext              6.51%        63.58% (6.6)             341/533
  aligned head         39.20%        63.99% (2.8)             360/533
  Katsuki, LoRA        15.44%        60.85% (4.6)             330/533
  Katsuki, frozen      11.59%        52.68% (3.9)             288/533
The aligned head matches Hein's own result on this split (63.6% calibrated, clip_disease_only/development_results.json)
but gains only +0.41 over ConvNeXt alone in the same pipeline and is 4.7 points below the v6.7 soft ensemble (68.66%).
Not adopted. The large gamma-0 gain does not survive calibration.
v5.13 features (2026-11-23, gan_disease_head_aligned_v5.13; runs/vlm/gan_head_routing_val_v5.13.json): HM 43.12% at gamma 0,
63.81% calibrated (gamma 2.6), unseen expert alone 358/533, head disease accuracy pseudo-unseen 22.9% / val-seen 79.9%.
Within noise of the v10.3-feature head. Not adopted."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402

CNX = ROOT / "runs/baseline/aniq_independent_convnext_base_336_1split"
GAMMAS = np.round(np.arange(0.0, 15.01, 0.1), 2)


def score(pred, y, n):
    hit = pred == y
    s, u = hit[:n].mean(), hit[n:].mean()
    return {"seen": float(s), "unseen": float(u), "hm": float(2 * s * u / (s + u) if s + u else 0.0),
            "seen_correct": int(hit[:n].sum()), "unseen_correct": int(hit[n:].sum())}


def main(heads):
    cfg = json.load(open(ROOT / "config/vlm/v6.7_three_member_ensemble_config.json"))
    v = cfg["validation"]
    df = read_metadata()
    pairs = pair_table(df)
    fit, vs, vu, held = make_validation_split(df[df["split"] == "train"], v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
                                              cfg["seed"], tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
    held = np.sort(np.asarray(held))
    y = np.r_[vs["pair_id"].to_numpy(), vu["pair_id"].to_numpy()]
    n = len(vs)
    ci, di = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    si = np.array(sorted(set(fit["pair_id"])))
    assert len(held) == 20 and not set(held) & set(si) and not pairs["is_unseen"].to_numpy()[np.r_[si, held]].any()
    ids = np.sort(np.r_[si, held])
    s = np.isin(ids, si)

    m = json.load(open(CNX / "metrics.json"))["selected_epochs"]
    z = np.load(CNX / "val_logp_split1.npz")
    crop, cnx_disease = z["crop"][m["crop_net"] - 1], z["disease"][m["disease_net"] - 1]
    assert len(crop) == len(y)
    seen_pred = si[(crop[:, ci[si]] + cnx_disease[:, di[si]]).argmax(1)]

    arms = {"convnext": cnx_disease}
    for h in heads:
        f = np.load(ROOT / "runs/vlm" / h / "val.npz")
        assert (f["y_val_seen"] == vs["pair_id"].to_numpy()).all() and (f["y_val_unseen"] == vu["pair_id"].to_numpy()).all()
        arms[h] = f["ld"]
    results = {}
    for name, disease in arms.items():
        cand = crop[:, ci[ids]] + disease[:, di[ids]]
        unseen_pred = ids[~s][cand[:, ~s].argmax(1)]
        gap = cand[:, s].max(1) - cand[:, ~s].max(1)
        grid = [(g, score(np.where(gap >= g, seen_pred, unseen_pred), y, n)) for g in GAMMAS]
        g, best = max(grid, key=lambda t: t[1]["hm"])
        results[name] = {"gamma_0": grid[0][1], "gamma": float(g), "calibrated": best,
                         "unseen_expert_alone_correct": int((unseen_pred[n:] == y[n:]).sum()),
                         "disease_named_right_pseudo_unseen": float((disease[n:].argmax(1) == di[y[n:]]).mean())}
        print(f"{name:28s} gamma 0: HM {grid[0][1]['hm']:.2%} (seen {grid[0][1]['seen']:.1%}, unseen {grid[0][1]['unseen']:.1%}) | "
              f"gamma {g}: HM {best['hm']:.2%} (seen {best['seen']:.1%}, unseen {best['unseen']:.1%}) | "
              f"unseen expert alone {results[name]['unseen_expert_alone_correct']}/{len(vu)}")
    out = ROOT / "runs/vlm/gan_head_routing_val.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
