"""Score-level ensemble of finished runs (e.g. v5 + the independent-CNN baseline), evaluated with the shared protocol.

Usage:  python -m train.ensemble --config config/vlm/v6_cnn_ensemble_config.json

Each member's scores are put on a common scale, log-probabilities over the 333 pairs:
    CLIP runs:  log_softmax(logit_scale * cosine)
    CNN runs:   log_softmax(log P(crop) + log P(disease))     (renormalised over the 333 pairs)
and combined as  w * member_1 + (1 - w) * member_2.  The weight w and gamma are chosen on the validation split.

Validation scores: each member is retrained on the validation split's training part (same split procedure and seed
as the members used) for the epoch count it selected, and its scores on the validation images are computed. They are
cached in the output folder, so re-running with another weight grid costs nothing. Test scores come from each
member's saved scores.npz (its final model, refit on all training data), so no member is retrained on test data.
"""

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch

from dpd.config import load_config, output_dir
from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.ensemble import to_logprob
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split

def compose_cnn_member(logp: dict, pairs, metrics: dict, fit_df) -> np.ndarray:
    """CNN member pair scores with the composition the member chose on validation (train_cnn.py "composition"):
    crop weight and logit-adjustment tau from its metrics.json, defaulting to the plain log P(crop) + log P(disease)
    for runs without them. Priors come from `fit_df`, the images these validation networks were trained on, just as
    train_cnn.py uses the fit split in stage 1 (its saved test scores used all of train, matching its final fit)."""
    from dpd.cnn import compose_pair_scores, log_priors

    w, tau = float(metrics.get("crop_weight", 0.5)), float(metrics.get("tau", 0.0))
    prior = log_priors(fit_df) if tau else None
    return compose_pair_scores(logp["crop"], logp["disease"], pairs, w, tau, prior)


def member_val_scores(run_dir, mcfg, metrics, fit_df, val_df, epochs, pairs):
    """Retrain one member on the split's fit part for `epochs` and score the validation images (all 333 pairs),
    scored exactly the way the member's saved test scores were."""
    method = mcfg["method"]
    if method == "clip_prompt_lora":
        from dpd.clip_utils import clip_transforms
        from train import train_lora as tl

        image_dir = ensure_resized(pd.concat([fit_df, val_df])["file"], mcfg["input"]["resize_cache"])
        tfs = clip_transforms(mcfg["input"], mcfg["augmentation"])
        tl.seed_everything(mcfg["seed"])
        model, lora = tl.build(mcfg, pairs, mcfg["runtime"]["device"])
        tl.fit(model, lora, fit_df, sorted(set(fit_df["pair_id"])), epochs, mcfg, image_dir, tfs,
               log=lambda r: print(f"  {run_dir.name} epoch {r['epoch']}/{epochs} loss {r['loss']}", flush=True))
        tta = tuple(mcfg["evaluation"].get("tta", ["identity"]))
        feats = tl.encode(model, tl.loader(val_df, image_dir, tfs[1], mcfg, shuffle=False), "cuda", mcfg["runtime"]["amp"], tta)
        return tl.pair_scores(model, feats, mcfg["runtime"]["amp"])
    if method in ("cnn_independent", "cnn_multitask"):
        from dpd.cnn import NETWORKS, build_transforms
        from train import train_cnn as tc

        image_dir = ensure_resized(pd.concat([fit_df, val_df])["file"], mcfg["input"]["resize_cache"])
        tfs = build_transforms(mcfg["input"], mcfg["augmentation"])
        val_loader = tc.make_loader(val_df, image_dir, tfs[1], mcfg, shuffle=False)
        logp = {}
        for name, heads in NETWORKS[method].items():
            tc.seed_everything(mcfg["seed"])
            model, _ = tc.train_network(heads, fit_df, epochs, mcfg, image_dir, tfs,
                                        log=lambda r, n=name: print(f"  {run_dir.name}/{n} epoch {r['epoch']}/{epochs} loss {r['loss']}", flush=True))
            logp.update(tc.predict(model, val_loader, mcfg["runtime"]["device"], mcfg["runtime"]["amp"]))
            del model
            torch.cuda.empty_cache()
        print(f"  {run_dir.name}: crop weight {metrics.get('crop_weight', 0.5)}, tau {metrics.get('tau', 0.0)} "
              f"(the member's validation-chosen scoring)", flush=True)
        return compose_cnn_member(logp, pairs, metrics, fit_df)
    raise ValueError(f"ensemble does not support member method {method!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "ensemble", f"{args.config} is a '{cfg['method']}' config"
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")
    vcfg, ev = cfg["validation"], cfg["evaluation"]

    df = read_metadata()
    pairs = pair_table(df)
    train = df[df["split"] == "train"]
    fit_df, val_seen, val_unseen, _ = make_validation_split(
        train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], cfg["seed"],
        tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
    )
    val_df = pd.concat([val_seen, val_unseen])
    n_vs = len(val_seen)

    members = []
    for m in cfg["members"]:
        run_dir = PROJECT_ROOT / m["run"]
        mcfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text())
        epochs = metrics["selected_epoch"]
        # The cache name includes the member's scoring, so a stale file can never be reused with different settings.
        tag = f"_w{metrics['crop_weight']}_tau{metrics['tau']}" if "crop_weight" in metrics else ""
        cache = out / f"val_scores_{run_dir.name}{tag}.npy"
        if cache.exists():
            val = np.load(cache)
        else:
            print(f"validation scores for {run_dir.name}: retraining {epochs} epochs on the validation split", flush=True)
            val = member_val_scores(run_dir, mcfg, metrics, fit_df, val_df, epochs, pairs)
            np.save(cache, val)
        test = np.load(run_dir / "scores.npz")
        members.append({"name": run_dir.name, "method": mcfg["method"], "val": to_logprob(val, mcfg["method"]),
                        "test": {s: to_logprob(test[f"{s}_scores"], mcfg["method"]) for s in ("test_seen", "test_unseen")},
                        "labels": {s: test[f"{s}_labels"] for s in ("test_seen", "test_unseen")}})
    assert len(members) == 2, "this script mixes exactly two members"
    a, b = members

    # ---- choose the weight and gamma on validation ----
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
    y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
    weights = np.round(np.arange(*cfg["ensemble"]["weight_grid"]), 4)
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 4)
    grid = np.zeros((len(weights), len(gammas)))
    for i, w in enumerate(weights):
        s = w * a["val"] + (1 - w) * b["val"]
        s[:, hidden] = -1e9
        grid[i] = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    w, gamma = float(weights[i]), float(gammas[j])
    alone = {a["name"]: float(grid[np.argmin(np.abs(weights - 1))].max()), b["name"]: float(grid[np.argmin(np.abs(weights))].max())}
    print(f"selected weight {w} on {a['name']} ({1 - w:.2f} on {b['name']}), gamma {gamma}: val HM {grid[i, j]:.2%}  "
          f"(members alone, same split: {', '.join(f'{k} {v:.2%}' for k, v in alone.items())})", flush=True)

    # ---- test once ----
    scores = {s: w * a["test"][s] + (1 - w) * b["test"][s] for s in ("test_seen", "test_unseen")}
    labels = a["labels"]
    assert all((labels[s] == b["labels"][s]).all() for s in labels), "members' test images are not in the same order"
    files = {s: df[df["split"] == s]["file"].to_numpy() for s in scores}
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], weight=w, members=[m["name"] for m in members],
                  val={"weight": w, "gamma": gamma, "val_hm": round(float(grid[i, j]), 4), "members_alone_val_hm": alone},
                  harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", grid)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
