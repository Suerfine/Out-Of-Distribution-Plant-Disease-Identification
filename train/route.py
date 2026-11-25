"""Gated routing between two finished runs (v15+): a seen/unseen gate sends each image to a seen expert or an unseen
expert, each picked for its own branch.

Usage:  python -m train.route --config config/vlm/v15_gated_router_config.json [--validation-only]

The ensemble (train/ensemble.py) mixes two members with one weight over all 333 pairs and lets gamma decide between
seen and unseen classes. Here the three jobs are separated, after COSMO (Atzmon & Chechik, CVPR 2019):
  gate           P(unseen | image): logistic regression on label-space-agnostic features of the members' scores
                 (router.gate.features, see gate_features(): best seen / best unseen log-probability, confidence
                 inside each branch, agreement between members; optionally the same for the one-weight mix).
  seen expert    w_s * A + (1 - w_s) * B over the seen pairs only, w_s = best val-seen accuracy among seen pairs
  unseen expert  w_u * A + (1 - w_u) * B over the unseen pairs only, w_u = best pseudo-unseen accuracy among unseen pairs
  routing        hard: unseen expert if gate logit + bias > 0, else seen expert
                 soft: score(y) = log P(branch of y | image) + log P_expert(y | branch), argmax over all pairs
The gate is trained on the validation images (val-seen vs pseudo-unseen). To keep validation honest, validation HM
uses out-of-fold gate predictions (GroupKFold by pair, so the gate is always scored on pairs it was not trained on);
the gate applied to test is refit on all validation images. Experts, bias and hard / soft are chosen on validation.
Members' validation and test scores are obtained exactly as in train/ensemble.py (same cache files; the config may
name folders whose caches are reused, validation.reuse_val_scores_from). Test is scored once, at the end.
"""

import argparse
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dpd.config import load_config, output_dir
from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.ensemble import to_logprob
from dpd.evaluate import evaluate, format_summary, harmonic_mean
from dpd.report import save_test_outputs
from dpd.splits import make_validation_split

NEG = -1e9


def log_softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def branch_logp(logp, mask):
    """log P(y | image, y in mask): the scores renormalised inside one branch; NEG outside it."""
    return log_softmax(np.where(mask, logp, NEG))


def gate_features(members, seen, unseen, kind):
    """(features, names). They do not depend on which pairs are in each branch, so a gate trained on the
    pseudo-unseen pairs of validation can be applied to the real unseen pairs of test. `members` = (A, B, mix), the
    mix being the one-weight ensemble's scores. kind (router.gate.features):
      "members"      per member A, B: best seen, best unseen, confidence inside each branch; + the two agreements
      "members+mix"  the same, plus those four features of the mix
      "margins"      best unseen - best seen of A, B and the mix only (3 features)"""
    names, cols, top = [], [], []
    use = {"members": "AB", "members+mix": "ABM", "margins": "ABM"}[kind]
    for tag, lp in zip("ABM", members):
        if tag not in use:
            continue
        s, u = np.where(seen, lp, NEG), np.where(unseen, lp, NEG)
        if kind == "margins":
            cols.append(u.max(1) - s.max(1))
            names.append(f"{tag} margin")
            continue
        cols += [s.max(1), u.max(1), branch_logp(lp, seen).max(1), branch_logp(lp, unseen).max(1)]
        names += [f"{tag} best seen", f"{tag} best unseen", f"{tag} conf in seen", f"{tag} conf in unseen"]
        top.append((s.argmax(1), u.argmax(1)))
    if kind != "margins":
        cols += [(top[0][0] == top[1][0]).astype(float), (top[0][1] == top[1][1]).astype(float)]
        names += ["agree on seen", "agree on unseen"]
    return np.stack(cols, axis=1), names


def new_gate(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000))


def routed_scores(z, seen_lp, unseen_lp, seen, mode):
    """Final scores over all pairs from the gate logit z (P(unseen), bias already added) and the two experts'
    within-branch log-probabilities. Predict with argmax (gamma 0)."""
    if mode == "soft":
        log_pu, log_ps = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    else:  # hard: the routed branch always wins
        log_pu, log_ps = np.where(z > 0, 0.0, -1e6), np.where(z > 0, -1e6, 0.0)
    return np.where(seen, seen_lp + log_ps[:, None], unseen_lp + log_pu[:, None])


def hm_of(scores, y, n_vs):
    hit = scores.argmax(1) == y
    return harmonic_mean(hit[:n_vs].mean(), hit[n_vs:].mean()), hit[:n_vs].mean(), hit[n_vs:].mean()


def load_member(run, out, extra_dirs, val_df, cfg, vcfg, fit_df, pairs):
    """One member's validation scores (cached or retrained, as train/ensemble.py) and saved test scores, as
    log-probabilities over the 333 pairs."""
    run_dir = PROJECT_ROOT / run
    mcfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    tag = f"_w{metrics['crop_weight']}_tau{metrics['tau']}" if "crop_weight" in metrics else ""
    name = f"val_scores_{run_dir.name}{tag}.npy"
    found = next((d / name for d in [out, *extra_dirs] if (d / name).exists()), None)
    if found is not None:
        val = np.load(found)
        print(f"{run_dir.name}: validation scores from {found.relative_to(PROJECT_ROOT)}", flush=True)
    else:
        from train.ensemble import member_val_scores

        epochs = metrics["selected_epoch"]
        print(f"{run_dir.name}: no cached validation scores, retraining {epochs} epochs on the validation split", flush=True)
        keys = ("n_holdout_pairs", "min_holdout_images", "val_fraction", "exclude_holdout_diseases", "max_images_per_holdout_pair")
        same_split = mcfg["seed"] == cfg["seed"] and all(mcfg["validation"].get(k) == vcfg[k] for k in keys)
        val = member_val_scores(run_dir, mcfg, metrics, fit_df, val_df, epochs, pairs, same_split)
        np.save(out / name, val)
    assert val.shape == (len(val_df), len(pairs)), f"{name}: shape {val.shape} does not match this validation split"
    test = np.load(run_dir / "scores.npz")
    return {"name": run_dir.name, "val": to_logprob(val, mcfg["method"]),
            "test": {s: to_logprob(test[f"{s}_scores"], mcfg["method"]) for s in ("test_seen", "test_unseen")},
            "labels": {s: test[f"{s}_labels"] for s in ("test_seen", "test_unseen")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-only", action="store_true", help="stop after the validation report; test is not scored")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "router", f"{args.config} is a '{cfg['method']}' config"
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")
    vcfg, rcfg, ev = cfg["validation"], cfg["router"], cfg["evaluation"]

    df = read_metadata()
    pairs = pair_table(df)
    train = df[df["split"] == "train"]
    fit_df, val_seen, val_unseen, _ = make_validation_split(
        train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], cfg["seed"],
        tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
    )
    val_df = pd.concat([val_seen, val_unseen])
    n_vs = len(val_seen)
    y = val_df["pair_id"].to_numpy()
    is_unseen_img = np.arange(len(val_df)) >= n_vs

    extra = [PROJECT_ROOT / d for d in vcfg.get("reuse_val_scores_from", [])]
    assert len(cfg["members"]) == 2, "this script routes between exactly two members"
    a, b = (load_member(m["run"], out, extra, val_df, cfg, vcfg, fit_df, pairs) for m in cfg["members"])

    # Validation label space: real unseen pairs hidden; pseudo-unseen = pairs absent from the fit part.
    real_unseen = pairs["is_unseen"].to_numpy()
    hidden = real_unseen if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    v_unseen = ~pairs["pair_id"].isin(set(fit_df["pair_id"])).to_numpy() & ~hidden
    v_seen = ~v_unseen & ~hidden
    val_pairs = pairs.assign(is_unseen=v_unseen | hidden)
    va, vb = (log_softmax(np.where(hidden, NEG, m["val"])) for m in (a, b))
    weights = np.round(np.arange(*rcfg["weight_grid"]), 4)
    mix = lambda w, p, q: w * p + (1 - w) * q

    # ---- baselines on the same split: each member and the one-weight ensemble, gamma-calibrated ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 4)

    def gamma_hm(s):
        s = np.where(hidden, NEG, s)
        return max(evaluate(s[:n_vs], y[:n_vs], s[n_vs:], y[n_vs:], val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas)

    ens = {float(w): gamma_hm(mix(w, a["val"], b["val"])) for w in weights}  # exactly train/ensemble.py's scoring
    w_ens = max(ens, key=ens.get)
    report = {f"{a['name']} alone (gamma)": ens[1.0], f"{b['name']} alone (gamma)": ens[0.0], f"ensemble, one weight {w_ens} (gamma)": ens[w_ens]}

    # ---- experts: each branch's weight from accuracy inside that branch (independent of the gate) ----
    acc_s = [(branch_logp(mix(w, va, vb), v_seen)[:n_vs].argmax(1) == y[:n_vs]).mean() for w in weights]
    acc_u = [(branch_logp(mix(w, va, vb), v_unseen)[n_vs:].argmax(1) == y[n_vs:]).mean() for w in weights]
    w_s, w_u = float(weights[int(np.argmax(acc_s))]), float(weights[int(np.argmax(acc_u))])
    print(f"seen expert: weight {w_s} on {a['name']} (val-seen accuracy among seen pairs {max(acc_s):.2%}; "
          f"{a['name']} alone {acc_s[-1]:.2%}, {b['name']} alone {acc_s[0]:.2%})")
    print(f"unseen expert: weight {w_u} on {a['name']} (pseudo-unseen accuracy among unseen pairs {max(acc_u):.2%}; "
          f"{a['name']} alone {acc_u[-1]:.2%}, {b['name']} alone {acc_u[0]:.2%})", flush=True)
    perfect = harmonic_mean(max(acc_s), max(acc_u))

    # ---- gate: out-of-fold logits on validation (folds by pair) ----
    kind = rcfg["gate"]["features"]
    x, feature_names = gate_features((va, vb, log_softmax(mix(w_ens, va, vb))), v_seen, v_unseen, kind)
    z_oof = np.zeros(len(x))
    for tr, te in GroupKFold(n_splits=rcfg["gate"]["cv_folds"]).split(x, is_unseen_img, groups=y):
        z_oof[te] = new_gate(rcfg["gate"]["C"]).fit(x[tr], is_unseen_img[tr]).decision_function(x[te])
    from sklearn.metrics import roc_auc_score

    margin = lambda lp: np.where(v_unseen, lp, NEG).max(1) - np.where(v_seen, lp, NEG).max(1)
    auc = {"learned gate (out-of-fold)": roc_auc_score(is_unseen_img, z_oof), f"{a['name']} margin": roc_auc_score(is_unseen_img, margin(va)),
           f"{b['name']} margin": roc_auc_score(is_unseen_img, margin(vb)), "ensemble margin": roc_auc_score(is_unseen_img, margin(mix(w_ens, va, vb)))}
    print("gate AUROC on validation: " + ", ".join(f"{k} {v:.3f}" for k, v in auc.items()), flush=True)

    biases = np.round(np.arange(*rcfg["bias_grid"]), 4)

    def best_routing(z, seen_lp, unseen_lp, modes, biases=biases):
        return max(((*hm_of(routed_scores(z + bb, seen_lp, unseen_lp, v_seen, m), y, n_vs), m, float(bb)) for m in modes for bb in biases),
                   key=lambda r: r[0])

    exp_s, exp_u = branch_logp(mix(w_s, va, vb), v_seen), branch_logp(mix(w_u, va, vb), v_unseen)
    one = mix(w_ens, va, vb)
    modes = tuple(rcfg["modes"])
    full = best_routing(z_oof, exp_s, exp_u, modes)
    # Ablations: which part of the router earns the gain over the ensemble.
    # A margin gate's bias is gamma (hard routing on the ensemble margin with one weight is exactly the ensemble).
    report["ensemble-margin gate + per-branch experts"] = best_routing(margin(one), exp_s, exp_u, ("hard",), gammas)[0]
    report["learned gate + one-weight expert"] = best_routing(z_oof, branch_logp(one, v_seen), branch_logp(one, v_unseen), modes)[0]
    for m in modes:
        report[f"router, {m} routing"] = best_routing(z_oof, exp_s, exp_u, (m,))[0]
    report["perfect gate + per-branch experts (upper bound)"] = perfect
    hm, acc_seen, acc_unseen, mode, bias = full
    print("\nValidation HM (one split, seed %d):" % cfg["seed"])
    for k, v in report.items():
        print(f"  {k:52s} {v:.2%}")
    print(f"selected: {mode} routing, bias {bias}: val HM {hm:.2%} (seen {acc_seen:.2%}, unseen {acc_unseen:.2%})", flush=True)

    gate = new_gate(rcfg["gate"]["C"]).fit(x, is_unseen_img)  # the gate used on test: fit on all validation images
    import joblib

    joblib.dump(gate, out / "gate.joblib")  # scaler + logistic regression, for RouterRunModel (dpd/inference.py)
    coef = dict(zip(feature_names, np.round(gate[-1].coef_[0], 3).tolist()))
    print("gate coefficients (standardised features; + = looks unseen): " + ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))
    val = {"val_hm": round(hm, 4), "val_seen": round(float(acc_seen), 4), "val_unseen": round(float(acc_unseen), 4), "mode": mode, "bias": bias,
           "seen_expert_weight": w_s, "unseen_expert_weight": w_u, "ensemble_weight": w_ens, "gate_features": kind, "gate_auroc": {k: round(float(v), 4) for k, v in auc.items()},
           "val_hm_report": {k: round(float(v), 4) for k, v in report.items()}, "gate_coefficients": coef}
    (out / "validation.json").write_text(json.dumps(val, indent=2))
    if args.validation_only:
        print(f"validation only: test not scored. Saved {out / 'validation.json'}")
        return

    # ---- test once ----
    t_unseen, t_seen = real_unseen, ~real_unseen
    scores, labels = {}, a["labels"]
    assert all((labels[s] == b["labels"][s]).all() for s in labels), "members' test images are not in the same order"
    gate_stats = {}
    for s in ("test_seen", "test_unseen"):
        ta, tb = a["test"][s], b["test"][s]
        z = gate.decision_function(gate_features((ta, tb, log_softmax(mix(w_ens, ta, tb))), t_seen, t_unseen, kind)[0]) + bias
        gate_stats[s] = float((z > 0).mean())
        scores[s] = routed_scores(z, branch_logp(mix(w_s, ta, tb), t_seen), branch_logp(mix(w_u, ta, tb), t_unseen), t_seen, mode)
    files = {s: df[df["split"] == s]["file"].to_numpy() for s in scores}
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, tuple(ev["top_k"]))
    result.update(version=cfg["version"], members=[a["name"], b["name"]], val=val,
                  gate_sent_to_unseen={k: round(v, 4) for k, v in gate_stats.items()})
    print(f"\ngate sent to the unseen expert: {gate_stats['test_seen']:.1%} of test-seen images, {gate_stats['test_unseen']:.1%} of test-unseen images")
    print("\nTest (routing chosen on validation; no gamma):\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, 0.0)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
