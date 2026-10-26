"""CNN baselines (spec section 3.1), evaluated with the same protocol as the CLIP models.

Usage:  python train_cnn.py --config config/baseline/aniq_baseline_independent_cnn.json

method "cnn_independent": two separate CNNs, one predicting the crop (55), one the disease (175).
method "cnn_multitask":   one CNN backbone with a crop head and a disease head, trained jointly.
Pair scores = log P(crop) + log P(disease) over all 333 pairs (dpd/cnn.py), so unseen pairs can be predicted.

Stage 1: train on train minus a validation split (dpd/splits.py, same as train.py); after every epoch, score the
validation images and sweep gamma; pick the (epoch, gamma) with the best validation HM (mean over splits).
For the independent baseline both CNNs are scored at the same epoch.
Stage 2: refit every network on all training images for the chosen epochs, then evaluate once on test.
"""

import argparse
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpd.cnn import ImageLabels, MultiHeadNet, build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split

HEAD_SIZES = {"crop": 55, "disease": 175}
NETWORKS = {
    "cnn_independent": {"crop_net": ["crop"], "disease_net": ["disease"]},
    "cnn_multitask": {"multitask_net": ["crop", "disease"]},
}


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
    mcfg, ocfg, lcfg = cfg["model"], cfg["optimizer"], cfg["loss"]
    model = MultiHeadNet(mcfg["backbone"], mcfg["pretrained"], {h: HEAD_SIZES[h] for h in heads}, mcfg["head_dropout"])
    model = model.to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
            {"params": model.heads.parameters(), "lr": ocfg["lr_head"]},
        ],
        weight_decay=ocfg["weight_decay"],
    )
    loader = make_loader(train_df, image_dir, tfs[0], cfg, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, {h: 0 for h in heads}, 0
        for x, y_crop, y_dis in loader:
            x = x.to(device, non_blocking=True)
            targets = {"crop": y_crop.to(device), "disease": y_dis.to(device)}
            with torch.autocast(device, enabled=amp):
                logits = model(x)
                loss = sum(
                    lcfg["weights"][h] * F.cross_entropy(logits[h].float(), targets[h], label_smoothing=lcfg["label_smoothing"])
                    for h in heads
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
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

    # ---- Stage 1: choose epoch and gamma on validation ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(gammas)))
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
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        for e in range(tcfg["max_epochs"]):
            s = compose_pair_scores(val_logp["crop"][e], val_logp["disease"][e], pairs)
            s[:, hidden] = -1e9
            hm_grid[k, e] = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
            print(f"split {k + 1} epoch {e + 1}: val HM {hm_grid[k, e, 0]:.2%} at gamma 0, best {hm_grid[k, e].max():.2%} "
                  f"at gamma {gammas[hm_grid[k, e].argmax()]}", flush=True)
    mean_grid = hm_grid.mean(axis=0)
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]], "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4)}
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    test_loaders = {s: make_loader(df[df["split"] == s], image_dir, tfs[1], cfg, shuffle=False) for s in ("test_seen", "test_unseen")}
    test_logp = {s: {} for s in test_loaders}
    for name, heads in networks.items():
        seed_everything(cfg["seed"])
        model, _ = train_network(heads, train, best_epoch, cfg, image_dir, tfs, log=logger(f"final/{name}"))
        torch.save(model.state_dict(), out / f"{name}.pt")
        for s, loader in test_loaders.items():
            test_logp[s].update(predict(model, loader, runtime["device"], runtime["amp"]))
        del model
        torch.cuda.empty_cache()

    scores, labels, files = {}, {}, {}
    for s in test_loaders:
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(test_logp[s]["crop"], test_logp[s]["disease"], pairs)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
