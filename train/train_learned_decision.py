"""Run 3: a learned seen-versus-unseen decision in place of one global gamma (method cnn_supcon).

The model is the final model of an earlier run (config `decision.source_run`); nothing is retrained for the test. Steps:
  1. Re-train the validation splits (same seeds and split procedure as the trainer, 20-epoch cosine schedule, stopped at
     the source run's selected epoch) and keep each validation image's scores. Held-out pairs play the unseen pairs.
  2. Compare gamma with a learned decision by leave-one-split-out on those validation splits only: choose gamma, or fit the
     decision on two splits, and measure the harmonic mean on the third.
  3. Fit the decision on all validation splits and apply it once to the test sets with the source run's final model.

The decision is a logistic regression on facts about the scores (not the image source): the best seen and best unseen
pair score, their gap, the margin of the best over the second best in each group, the entropy and top probability of the
crop and disease heads, and the difference of the log-sum-exp of each group. It predicts "belongs to an unseen pair";
the image then gets the best pair of the predicted group. The threshold is chosen for the validation harmonic mean.
Using only the gap between the two best scores gives exactly a gamma rule, so gamma is a special case.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dpd.cnn import ImageLabels, build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from dpd.supcon import ContrastiveNet, embed
from train.train_cnn import seed_everything
from train.train_supcon import train_network

FEATURES = ["best_seen", "best_unseen", "gap_seen_minus_unseen", "margin_seen", "margin_unseen",
            "entropy_crop", "entropy_disease", "pmax_crop", "pmax_disease", "lse_seen_minus_unseen"]


def features(logp_c, logp_d, S, seen_m, unseen_m):
    s, u = np.sort(S[:, seen_m], axis=1), np.sort(S[:, unseen_m], axis=1)
    ent = lambda lp: -(np.exp(lp) * lp).sum(1)
    cols = [s[:, -1], u[:, -1], s[:, -1] - u[:, -1], s[:, -1] - s[:, -2], u[:, -1] - u[:, -2], ent(logp_c), ent(logp_d),
            np.exp(logp_c.max(1)), np.exp(logp_d.max(1)),
            logsumexp(S[:, seen_m], axis=1) - logsumexp(S[:, unseen_m], axis=1)]
    return np.stack(cols, 1).astype(np.float64)


def group_view(logp_c, logp_d, S, seen_m, unseen_m):
    """Features plus each group's best pair id, for a block of images."""
    seen_ids, unseen_ids = np.flatnonzero(seen_m), np.flatnonzero(unseen_m)
    return {"F": features(logp_c, logp_d, S, seen_m, unseen_m),
            "pred_seen": seen_ids[S[:, seen_m].argmax(1)], "pred_unseen": unseen_ids[S[:, unseen_m].argmax(1)]}


def hm_of(d, y, pred_seen, pred_unseen, from_unseen):
    """Harmonic mean of seen and unseen Top-1 when images with d True are sent to the unseen group."""
    a = np.mean((~d & (pred_seen == y))[~from_unseen])
    b = np.mean((d & (pred_unseen == y))[from_unseen])
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def pick_gamma(splits, gammas):
    mean = [np.mean([hm_of(sp["F"][:, 2] < g, sp["y"], sp["pred_seen"], sp["pred_unseen"], sp["from_unseen"]) for sp in splits]) for g in gammas]
    return float(gammas[int(np.argmax(mean))]), float(np.max(mean))


def fit_decision(splits, thresholds):
    X = np.concatenate([sp["F"] for sp in splits])
    t = np.concatenate([sp["from_unseen"] for sp in splits]).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X, t)
    mean = [np.mean([hm_of(clf.predict_proba(sp["F"])[:, 1] > th, sp["y"], sp["pred_seen"], sp["pred_unseen"], sp["from_unseen"]) for sp in splits]) for th in thresholds]
    return clf, float(thresholds[int(np.argmax(mean))]), float(np.max(mean))


def collect_validation(cfg, df, pairs, image_dir, tfs, epoch, quick):
    rt, vcfg, tcfg = cfg["runtime"], cfg["validation"], cfg["training"]
    device, amp = rt["device"], rt["amp"]
    train = df[df["split"] == "train"]
    out = []
    for k in range(vcfg.get("n_splits", 1)):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, _ = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"])
        if quick:
            fit_df, val_seen, val_unseen = fit_df.sample(n=1200, random_state=0), val_seen.head(250), val_unseen.head(250)
        val_df = pd.concat([val_seen, val_unseen])
        val_loader = DataLoader(ImageLabels(val_df, image_dir, tfs[1]), batch_size=128, shuffle=False,
                                num_workers=rt["num_workers"], pin_memory=True)
        state = {"n": 0}

        def on_epoch(model):
            state["n"] += 1
            return embed(model, val_loader, device, amp) if state["n"] == epoch else None

        def log(rec):
            print(f"collect split{k + 1} " + " ".join(f"{a}={b}" for a, b in rec.items()), flush=True)

        seed_everything(split_seed)
        _, res = train_network(fit_df, epoch, cfg, image_dir, tfs, on_epoch, log, schedule_epochs=tcfg["max_epochs"])
        emb = res[-1]
        S = compose_pair_scores(emb["logp_crop"], emb["logp_disease"], pairs)
        seen_m = pairs["pair_id"].isin(fit_df["pair_id"]).to_numpy()
        unseen_m = pairs["pair_id"].isin(val_unseen["pair_id"].unique()).to_numpy()
        sp = group_view(emb["logp_crop"], emb["logp_disease"], S, seen_m, unseen_m)
        sp["y"] = val_df["pair_id"].to_numpy()
        sp["from_unseen"] = np.r_[np.zeros(len(val_seen), bool), np.ones(len(val_unseen), bool)]
        out.append(sp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true", help="pipeline check: 1 split, tiny data, 1 epoch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "cnn_supcon", f"{args.config} is a '{cfg['method']}' config"
    dcfg = cfg["decision"]
    src = dcfg["source_run"]
    src_metrics = json.load(open(f"{src}/metrics.json"))
    epoch = 1 if args.quick else src_metrics["selected_epoch"]
    if args.quick:
        cfg["validation"] = {**cfg["validation"], "n_splits": 1}
    rt = cfg["runtime"]
    device, amp = rt["device"], rt["amp"]
    out = output_dir(cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])

    # ---- 1. validation scores ----
    cache = out / "val_collect.npz"
    if cache.exists() and not args.quick:
        z = np.load(cache)
        splits = [{k.split("__")[1]: z[k] for k in z.files if k.startswith(f"s{i}__")} for i in range(len({k.split("__")[0] for k in z.files}))]
        print(f"validation scores loaded from {cache}", flush=True)
    else:
        t0 = time.time()
        splits = collect_validation(cfg, df, pairs, image_dir, tfs, epoch, args.quick)
        if not args.quick:
            np.savez(cache, **{f"s{i}__{k}": v for i, sp in enumerate(splits) for k, v in sp.items()})
        print(f"validation scores collected in {time.time() - t0:.0f}s", flush=True)
    gammas = np.round(np.arange(*cfg["validation"]["gamma_grid"]), 6)
    thresholds = np.round(np.arange(0.02, 0.99, 0.01), 4)

    # ---- 2. leave-one-split-out comparison on validation ----
    loso = []
    if len(splits) > 1:
        for j in range(len(splits)):
            tr_, te_ = [s for i, s in enumerate(splits) if i != j], splits[j]
            g, _ = pick_gamma(tr_, gammas)
            clf, th, _ = fit_decision(tr_, thresholds)
            hg = hm_of(te_["F"][:, 2] < g, te_["y"], te_["pred_seen"], te_["pred_unseen"], te_["from_unseen"])
            hl = hm_of(clf.predict_proba(te_["F"])[:, 1] > th, te_["y"], te_["pred_seen"], te_["pred_unseen"], te_["from_unseen"])
            loso.append({"held_out_split": j + 1, "gamma": g, "hm_gamma": round(hg, 4), "threshold": th, "hm_learned": round(hl, 4)})
            print(f"leave-one-split-out, held out split {j + 1}: gamma {g} -> HM {hg:.2%} | learned (threshold {th}) -> HM {hl:.2%}", flush=True)
        print(f"mean held-out validation HM: gamma {np.mean([r['hm_gamma'] for r in loso]):.2%} | learned {np.mean([r['hm_learned'] for r in loso]):.2%}", flush=True)

    # ---- 3. fit on all validation splits, apply once to test with the source run's final model ----
    gamma_all, hm_gamma_all = pick_gamma(splits, gammas)
    clf, th, hm_learned_all = fit_decision(splits, thresholds)
    print(f"all splits: gamma {gamma_all} (val HM {hm_gamma_all:.2%}) | learned threshold {th} (val HM {hm_learned_all:.2%}, in-sample)", flush=True)
    if args.quick:
        print("quick mode: stopping before the test step")
        return
    m = cfg["model"]
    net = ContrastiveNet(m["backbone"], False, m["proj_dim"], m.get("proj_hidden"), m.get("ce_dropout"), m.get("drop_path_rate", 0.0)).to(device)
    net.load_state_dict(torch.load(f"{src}/supcon_net.pt", map_location=device))
    unseen_m = pairs["is_unseen"].to_numpy()
    seen_m = ~unseen_m
    scores, labels, files, tests = {}, {}, {}, {}
    for s in ("test_seen", "test_unseen"):
        sub = df[df["split"] == s]
        loader = DataLoader(ImageLabels(sub, image_dir, tfs[1]), batch_size=128, shuffle=False, num_workers=rt["num_workers"], pin_memory=True)
        emb = embed(net, loader, device, amp)
        S = compose_pair_scores(emb["logp_crop"], emb["logp_disease"], pairs).astype(np.float64)
        v = group_view(emb["logp_crop"], emb["logp_disease"], S, seen_m, unseen_m)
        to_unseen = clf.predict_proba(v["F"])[:, 1] > th
        # The decision as a per-image gamma: shift this image's seen scores so the decided group wins. Top-1 is then the
        # decision's pick, and Top-5 is ranked over all 333 pairs after calibration, like the gamma runs.
        margin = v["F"][:, 2]  # best seen minus best unseen
        g_img = np.where(to_unseen, margin + 1e-3, margin - 1e-3)
        S2 = S.copy()
        S2[:, seen_m] -= g_img[:, None]
        scores[s], labels[s], files[s] = S2, sub["pair_id"].to_numpy(), sub["file"].to_numpy()
        tests[s] = {**v, "S": S, "y": labels[s]}
    top_k = tuple(cfg["evaluation"]["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    # sanity: the source run's gamma rule applied to the same scores reproduces its recorded test result
    g_src = src_metrics["gamma"]
    hm_src = hm_of(tests["test_seen"]["F"][:, 2] < g_src, tests["test_seen"]["y"], tests["test_seen"]["pred_seen"], tests["test_seen"]["pred_unseen"], np.zeros(len(tests["test_seen"]["y"]), bool))
    seen_acc_src = np.mean((~(tests["test_seen"]["F"][:, 2] < g_src)) & (tests["test_seen"]["pred_seen"] == tests["test_seen"]["y"]))
    print(f"sanity: source run recorded seen Top-1 {src_metrics['test_seen']['top1']:.4f}, recomputed from scores {seen_acc_src:.4f}", flush=True)
    result.update(version=cfg["version"], selected_epoch=epoch, source_run=src, gamma_of_source_run=g_src,
                  val={"learned_val_hm_in_sample": round(hm_learned_all, 4), "gamma_val_hm": round(hm_gamma_all, 4), "threshold": th, "loso": loso},
                  harmonic_mean_top1_gamma0=None)
    print("\nTest, learned decision:\n" + format_summary(result))
    print(f"\nSource run (gamma {g_src}) test HM {src_metrics['harmonic_mean_top1']:.4f} | learned decision test HM {result['harmonic_mean_top1']:.4f}", flush=True)
    save_test_outputs(out, result, scores, labels, files, pairs, 0.0)
    coef = dict(zip(FEATURES, clf[-1].coef_[0].round(3).tolist()))
    (out / "decision.json").write_text(json.dumps({"features": FEATURES, "coefficients_standardised": coef, "threshold": th, "loso": loso}, indent=2), encoding="utf-8")
    print("coefficients (standardised):", coef)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
