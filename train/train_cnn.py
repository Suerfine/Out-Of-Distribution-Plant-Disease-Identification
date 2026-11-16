"""CNN baselines (spec section 3.1), evaluated with the same protocol as the CLIP models.

Usage:  python train_cnn.py --config config/baseline/aniq_baseline_independent_cnn.json

method "cnn_independent": two separate CNNs, one predicting the crop (55), one the disease (175).
method "cnn_multitask":   one CNN backbone with a crop head and a disease head, trained jointly.
Pair scores = log P(crop) + log P(disease) over all 333 pairs (dpd/cnn.py), so unseen pairs can be predicted.
An optional "composition" config section adds a crop/disease weight and post-hoc logit adjustment (dpd/cnn.py),
both chosen on validation; without it the score is the plain sum above.

Stage 1: train on train minus a validation split (dpd/splits.py, same as train.py); after every epoch, score the
validation images and sweep gamma (and crop weight / tau when configured); pick the (epoch, weight, tau, gamma)
with the best validation HM (mean over splits).
For the independent baseline both CNNs are scored at the same epoch.
Stage 2: refit every network on all training images for the chosen epochs, then evaluate once on test.
"""

import argparse
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpd.cnn import HEAD_SIZES, NETWORKS, ImageLabels, MultiHeadNet, build_transforms, compose_pair_scores, log_priors
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary, hm_top1_over_gammas, parse_gamma_grid
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# cuDNN autotuning helps; channels_last does NOT: on this setup (RTX 4050 Laptop, torch 2.11 + cuDNN 9.19) a
# ResNet-50 AMP training step ran at 60 img/s with channels_last vs 269 img/s without, so it is not used.
torch.backends.cudnn.benchmark = True


def make_loader(df, image_dir, transform, cfg, shuffle):
    rt = cfg["runtime"]
    return DataLoader(
        ImageLabels(df, image_dir, transform), batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
        num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=shuffle,
    )


@torch.no_grad()
def predict(model, loader, device, amp) -> dict[str, np.ndarray]:
    """Log-probabilities per head for every image in the loader."""
    model.eval()
    out = {k: [] for k in model.heads}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        for k, v in logits.items():
            out[k].append(F.log_softmax(v.float(), dim=-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def train_network(heads, train_df, epochs, cfg, image_dir, tfs, eval_loader=None, log=None):
    """Train one network; if eval_loader is given, return its log-probs after every epoch."""
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    mcfg, ocfg, lcfg, tcfg = cfg["model"], cfg["optimizer"], cfg["loss"], cfg["training"]
    model = MultiHeadNet(mcfg["backbone"], mcfg["pretrained"], {h: HEAD_SIZES[h] for h in heads}, mcfg["head_dropout"],
                         mcfg.get("drop_path_rate", 0.0))
    model = model.to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
            {"params": model.heads.parameters(), "lr": ocfg["lr_head"]},
        ],
        weight_decay=ocfg["weight_decay"],
    )
    loader = make_loader(train_df, image_dir, tfs[0], cfg, shuffle=True)
    # Optional training keys (absent in older configs, which keep the original per-step cosine to 0 over `epochs`):
    # accumulation_steps (physical batch_size x steps = effective batch), gradient_clip (max grad norm),
    # warmup_epochs (linear warmup), min_lr_ratio (cosine floor), lr_schedule_epochs (a fixed cosine horizon, so a
    # refit for e epochs follows the same learning rates as the first e epochs of the validation run).
    accum, clip = tcfg.get("accumulation_steps", 1), tcfg.get("gradient_clip")
    steps = len(loader) // accum
    if any(k in tcfg for k in ("warmup_epochs", "min_lr_ratio", "lr_schedule_epochs")):
        warmup, floor = tcfg.get("warmup_epochs", 0) * steps, tcfg.get("min_lr_ratio", 0.0)
        horizon = tcfg.get("lr_schedule_epochs", epochs) * steps

        def lr_factor(s):
            if s < warmup:
                return (s + 1) / warmup
            t = min(1.0, (s - warmup) / max(1, horizon - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)  # drops any partial accumulation left from the previous epoch
        t0, tot, correct, n = time.time(), 0.0, {h: 0 for h in heads}, 0
        for step, (x, y_crop, y_dis) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            targets = {"crop": y_crop.to(device), "disease": y_dis.to(device)}
            with torch.autocast(device, enabled=amp):
                logits = model(x)
                loss = sum(
                    lcfg["weights"][h] * F.cross_entropy(logits[h].float(), targets[h], label_smoothing=lcfg["label_smoothing"])
                    for h in heads
                )
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                if clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot += loss.item() * len(x)
            n += len(x)
            for h in heads:
                correct[h] += (logits[h].argmax(1) == targets[h]).sum().item()
        rec = {"epoch": epoch, "loss": round(tot / n, 4), **{f"train_{h}_acc": round(correct[h] / n, 4) for h in heads},
               "sec": round(time.time() - t0)}
        if eval_loader is not None:
            per_epoch.append(predict(model, eval_loader, device, amp))
        if log:
            log(rec)
    return model, per_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] in NETWORKS, f"{args.config} is a '{cfg['method']}' config"
    networks = NETWORKS[cfg["method"]]
    runtime, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")

    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    train = df[df["split"] == "train"]
    history = []

    def logger(prefix):
        def log(rec):
            rec = {"stage": prefix, **rec}
            history.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        return log

    # ---- Stage 1: choose epoch, scoring (crop weight, logit adjustment) and gamma on validation ----
    gammas = parse_gamma_grid(vcfg.get("gamma_grid"))
    if len(gammas) == 1 and gammas[0] == 0.0:
        print("Gamma tuning disabled (gamma_grid=0): evaluating uncalibrated (gamma=0.0) across all stages.", flush=True)
    # Optional "composition" section; without it the score is the baselines' log P(crop) + log P(disease).
    ccfg = cfg.get("composition", {})
    weights = np.round(np.arange(*ccfg["crop_weight_grid"]), 6) if "crop_weight_grid" in ccfg else np.array([0.5])
    taus = np.round(np.arange(*ccfg["tau_grid"]), 6) if "tau_grid" in ccfg else np.array([0.0])
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    # validation.separate_epochs (cnn_independent only): the crop and disease networks are separate models, so each
    # gets its own epoch count; every (crop epoch, disease epoch) combination is scored. Otherwise they share one.
    separate = vcfg.get("separate_epochs", False)
    assert not separate or set(networks) == {"crop_net", "disease_net"}, "separate_epochs needs two independent networks"
    n_ep = tcfg["max_epochs"]
    combos = [(a, b) for a in range(n_ep) for b in range(n_ep)] if separate else [(e, e) for e in range(n_ep)]
    hm_grid = np.zeros((n_splits, len(combos), len(weights), len(taus), len(gammas)))
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        val_df = pd.concat([val_seen, val_unseen])
        n_vs = len(val_seen)
        eval_loader = make_loader(val_df, image_dir, tfs[1], cfg, shuffle=False)
        val_logp = {}  # head -> list over epochs of log-prob arrays
        for name, heads in networks.items():
            seed_everything(split_seed)
            _, per_epoch = train_network(heads, fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, eval_loader, logger(f"split{k + 1}/{name}"))
            for h in heads:
                val_logp[h] = [p[h] for p in per_epoch]
        if k == 0 and tcfg.get("lr_schedule_epochs"):
            # With a fixed lr horizon, epoch e of this run equals a fresh e-epoch run, so train/ensemble.py can use these
            # split-1 validation predictions instead of retraining the networks (val_seen rows first, then val_unseen).
            np.savez_compressed(out / "val_logp_split1.npz", **{h: np.stack(v).astype(np.float32) for h, v in val_logp.items()})
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        prior = log_priors(fit_df)  # the frequencies the validation networks were trained on
        for c, (ec, ed) in enumerate(combos):
            for i, w in enumerate(weights):
                for j, t in enumerate(taus):
                    s = compose_pair_scores(val_logp["crop"][ec], val_logp["disease"][ed], pairs, w, t, prior)
                    s[:, hidden] = -1e9
                    hm_grid[k, c, i, j] = hm_top1_over_gammas(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, gammas)
            if not separate:
                i, j, g = np.unravel_index(hm_grid[k, c].argmax(), hm_grid[k, c].shape)
                print(f"split {k + 1} epoch {ec + 1}: best val HM {hm_grid[k, c, i, j, g]:.2%} at crop weight {weights[i]}, "
                      f"tau {taus[j]}, gamma {gammas[g]}", flush=True)
        if separate:
            c, i, j, g = np.unravel_index(hm_grid[k].argmax(), hm_grid[k].shape)
            print(f"split {k + 1}: best val HM {hm_grid[k, c, i, j, g]:.2%} at crop epoch {combos[c][0] + 1}, disease epoch "
                  f"{combos[c][1] + 1}, crop weight {weights[i]}, tau {taus[j]}, gamma {gammas[g]}", flush=True)
    mean_grid = hm_grid.mean(axis=0)
    c, i, j, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    epochs = {"crop_net": combos[c][0] + 1, "disease_net": combos[c][1] + 1} if separate else {n: combos[c][0] + 1 for n in networks}
    best_epoch, crop_weight, tau, gamma = max(epochs.values()), float(weights[i]), float(taus[j]), float(gammas[g])
    best = {"epoch": best_epoch, "epochs": epochs, "crop_weight": crop_weight, "tau": tau, "gamma": gamma,
            "val_hm_mean": round(float(mean_grid[c, i, j, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, c, i, j, g]],
            "val_hm_g0_mean": round(float(mean_grid[c, i, j, 0]), 4)}
    # In-run ablation: the same networks scored the baselines' way (w 0.5, tau 0), with its own best epoch and gamma.
    if 0.5 in weights and 0.0 in taus:
        plain = mean_grid[:, list(weights).index(0.5), list(taus).index(0.0)]
        pe, pg = np.unravel_index(plain.argmax(), plain.shape)
        best["plain_scoring"] = {"epochs": [combos[pe][0] + 1, combos[pe][1] + 1], "gamma": float(gammas[pg]),
                                 "val_hm_mean": round(float(plain[pe, pg]), 4)}
    print(f"selected epochs {epochs}, crop weight {crop_weight}, tau {tau}, gamma {gamma}: "
          f"mean val HM {best['val_hm_mean']:.2%}", flush=True)
    if "plain_scoring" in best:
        print(f"  (plain log P(crop) + log P(disease) scoring: {best['plain_scoring']})", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    test_loaders = {s: make_loader(df[df["split"] == s], image_dir, tfs[1], cfg, shuffle=False) for s in ("test_seen", "test_unseen")}
    test_logp = {s: {} for s in test_loaders}
    for name, heads in networks.items():
        seed_everything(cfg["seed"])
        model, _ = train_network(heads, train, epochs[name], cfg, image_dir, tfs, log=logger(f"final/{name}"))
        torch.save(model.state_dict(), out / f"{name}.pt")
        for s, loader in test_loaders.items():
            test_logp[s].update(predict(model, loader, runtime["device"], runtime["amp"]))
        del model
        torch.cuda.empty_cache()

    scores, labels, files = {}, {}, {}
    prior = log_priors(train)
    for s in test_loaders:
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(test_logp[s]["crop"], test_logp[s]["disease"], pairs, crop_weight, tau, prior)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, selected_epochs=epochs, crop_weight=crop_weight, tau=tau, val=best,
                  harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)  # splits x epoch combos x crop weights x taus x gammas
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
