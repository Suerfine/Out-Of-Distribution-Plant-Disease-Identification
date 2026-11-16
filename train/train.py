"""Train compositional prompts on a frozen CLIP, then evaluate with the shared protocol.

Usage:  python train.py --config config/vlm/v4.2_multi_split_val_config.json

Stage 1 (selection): for each of validation.n_splits splits (dpd/splits.py, one seed each), fit on train minus
the split; after every epoch, score the validation images and sweep the calibration gamma. Pick the (epoch, gamma)
with the best validation harmonic mean averaged over the splits.
Stage 2 (final fit, if training.final_fit == "all_train"): refit from scratch on all training images for the
chosen number of epochs, keep the chosen gamma, and evaluate once on both test sets.
"""

import argparse
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dpd.clip_utils import encode_images, load_clip
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.prompt_learner import build_model
from dpd.report import save_test_outputs
from dpd.splits import make_validation_split


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def text_embeddings(model, group, amp):
    with torch.autocast("cuda", enabled=amp):
        return model.encode(group)


def fit(model, feats, y_pair, y_crop, y_disease, trained_pairs, epochs, tcfg, runtime, on_epoch_end=None, history_path=None):
    """Train the crop/disease vectors. `trained_pairs`: pair ids the pair loss may use (the rest stay unseen)."""
    device = runtime["device"]
    params = [model.crop_vecs, model.disease_vecs, *model.ctx.values()]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n = len(feats)
    steps_per_epoch = int(np.ceil(n / tcfg["batch_size"]))
    # lr_schedule_epochs: fixed cosine horizon, so the stage-2 refit follows the first epochs of stage 1 (see train_lora.py).
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.get("lr_schedule_epochs", epochs) * steps_per_epoch)
    scale = tcfg["logit_scale"]
    trained = torch.as_tensor(trained_pairs, device=device)
    # Map pair id -> column in the trained-pairs logits.
    col = torch.full((int(trained.max()) + 1,), -1, device=device, dtype=torch.long)
    col[trained] = torch.arange(len(trained), device=device)
    feats, y_pair, y_crop, y_disease = (torch.as_tensor(a, device=device) for a in (feats, y_pair, y_crop, y_disease))
    lw = tcfg["loss_weights"]
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        totals = {"loss": 0.0, "pair": 0.0, "crop": 0.0, "disease": 0.0}
        for s in range(steps_per_epoch):
            idx = perm[s * tcfg["batch_size"] : (s + 1) * tcfg["batch_size"]]
            v = feats[idx]
            t_pair = text_embeddings(model, "pairs", runtime["amp"])[trained]
            losses = {"pair": F.cross_entropy(scale * v @ t_pair.T, col[y_pair[idx]])}
            if lw["crop"] > 0:
                losses["crop"] = F.cross_entropy(scale * v @ text_embeddings(model, "crops", runtime["amp"]).T, y_crop[idx])
            if lw["disease"] > 0:
                losses["disease"] = F.cross_entropy(scale * v @ text_embeddings(model, "diseases", runtime["amp"]).T, y_disease[idx])
            loss = sum(lw[k] * l for k, l in losses.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            totals["loss"] += loss.item() / steps_per_epoch
            for k, l in losses.items():
                totals[k] += l.item() / steps_per_epoch
        record = {"epoch": epoch, **{k: round(v, 4) for k, v in totals.items()}}
        if on_epoch_end:
            record.update(on_epoch_end(epoch))
        history.append(record)
        print(" ".join(f"{k}={v}" for k, v in record.items()), flush=True)
        if history_path:  # written every epoch so progress can be checked while training runs
            pd.DataFrame(history).to_csv(history_path, index=False)
    return history


@torch.no_grad()
def all_pair_scores(model, feats, amp):
    model.eval()
    t = text_embeddings(model, "pairs", amp).float().cpu().numpy()
    return feats @ t.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "clip_prompt_learning", f"{args.config} is a '{cfg['method']}' config"
    runtime, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")
    seed_everything(cfg["seed"])

    df = read_metadata()
    pairs = pair_table(df)
    clip_model, preprocess, tokenizer = load_clip(cfg["model"], runtime["device"])
    clip_model.transformer.grad_checkpointing = runtime["grad_checkpointing"]

    feats = {}
    for split in ("train", "test_seen", "test_unseen"):
        sub = df[df["split"] == split]
        feats[split] = encode_images(clip_model, preprocess, sub["file"], cfg["model"], runtime, cache_name=split)
    train = df[df["split"] == "train"].copy()
    train["row"] = np.arange(len(train))

    # ---- Stage 1: choose epoch count and gamma on validation split(s) ----
    # With n_splits > 1, each split uses a different seed (different pseudo-unseen pairs and val images). The
    # validation HM is averaged over splits for every (epoch, gamma), and the best average is chosen, so one
    # unrepresentative split cannot set gamma on its own.
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    # Optionally hide the real unseen pairs during validation, so validation has only the pseudo-unseen pairs as
    # untrained candidates, as the test set has only the 10 real ones.
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(gammas)))  # validation HM per split, epoch, gamma
    history, holdouts = [], []
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        holdouts.append(pairs.loc[holdout, "pair"].tolist())
        print(f"validation split {k + 1}/{n_splits} (seed {split_seed}): fit {len(fit_df)} images, val-seen {len(val_seen)}, "
              f"pseudo-unseen {len(val_unseen)} ({len(holdout)} pairs: {', '.join(holdouts[-1])})")
        # In validation, "unseen" = every pair the model was not trained on.
        trained_ids = sorted(set(fit_df["pair_id"]))
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(trained_ids))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()

        def validate(epoch, k=k, val_seen=val_seen, val_unseen=val_unseen, val_pairs=val_pairs, y_vs=y_vs, y_vu=y_vu):
            s_seen = all_pair_scores(model, feats["train"][val_seen["row"]], runtime["amp"])
            s_unseen = all_pair_scores(model, feats["train"][val_unseen["row"]], runtime["amp"])
            s_seen[:, hidden] = s_unseen[:, hidden] = -1e9
            results = [evaluate(s_seen, y_vs, s_unseen, y_vu, val_pairs, g, (1,)) for g in gammas]
            hm_grid[k, epoch - 1] = [r["harmonic_mean_top1"] for r in results]
            best = results[int(hm_grid[k, epoch - 1].argmax())]
            return {
                "split": k + 1,
                "val_hm_g0": round(results[0]["harmonic_mean_top1"], 4),
                "val_hm": round(best["harmonic_mean_top1"], 4),
                "gamma": best["gamma"],
                "val_seen": round(best["test_seen"]["top1"], 4),
                "val_unseen": round(best["test_unseen"]["top1"], 4),
            }

        seed_everything(split_seed)
        model = build_model(cfg, clip_model, tokenizer, pairs).to(runtime["device"])
        rows = fit_df["row"].to_numpy()
        history += fit(
            model, feats["train"][rows], fit_df["pair_id"].to_numpy(), fit_df["crop_id"].to_numpy(), fit_df["disease_id"].to_numpy(),
            trained_ids, tcfg["max_epochs"], tcfg, runtime, on_epoch_end=validate,
            history_path=out / f"history_split{k + 1}.csv" if n_splits > 1 else out / "history.csv",
        )
    mean_grid = hm_grid.mean(axis=0)  # epochs x gammas
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {
        "epoch": best_epoch,
        "gamma": gamma,
        "val_hm_mean": round(float(mean_grid[e, g]), 4),
        "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]],
        "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4),
        "best_gamma_per_split": [float(gammas[hm_grid[s, e].argmax()]) for s in range(n_splits)],
    }
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%} "
          f"(per split {best['val_hm_per_split']}; each split's own best gamma {best['best_gamma_per_split']})")

    # ---- Stage 2: final fit on all training images ----
    if tcfg["final_fit"] == "all_train":
        seed_everything(cfg["seed"])
        model = build_model(cfg, clip_model, tokenizer, pairs).to(runtime["device"])
        fit(model, feats["train"], train["pair_id"].to_numpy(), train["crop_id"].to_numpy(), train["disease_id"].to_numpy(),
            sorted(set(train["pair_id"])), best_epoch, tcfg, runtime)
    elif tcfg["final_fit"] != "none":
        raise ValueError(f"unknown final_fit {tcfg['final_fit']!r}")
    else:
        raise NotImplementedError("final_fit 'none' would need the best-epoch weights; not implemented")

    # ---- Test (once) ----
    scores, labels, files = {}, {}, {}
    for split in ("test_seen", "test_unseen"):
        sub = df[df["split"] == split]
        scores[split], labels[split], files[split] = all_pair_scores(model, feats[split], runtime["amp"]), sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    np.save(out / "val_hm_grid.npy", hm_grid)  # splits x epochs x gammas, for plotting the selection
    with torch.no_grad():  # what the app needs to score new images without rebuilding the model
        np.save(out / "pair_text_embeddings.npy", text_embeddings(model, "pairs", runtime["amp"]).float().cpu().numpy())
    torch.save({"crop_vecs": model.crop_vecs.detach().cpu(), "disease_vecs": model.disease_vecs.detach().cpu(), "gamma": gamma}, out / "prompt_vectors.pt")
    (out / "validation_split.json").write_text(json.dumps({"gammas": gammas.tolist(), "holdout_pairs_per_split": holdouts}, indent=1))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
