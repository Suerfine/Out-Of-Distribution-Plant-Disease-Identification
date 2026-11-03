"""Disentangled multi-task CNN (Victor's proposed approach), evaluated with the same protocol as every other model.

Usage:  python -m train.train_disentangle --config config/disentangle/<name>.json

method "cnn_disentangle": one backbone, a crop branch and a disease branch, spies behind a gradient reversal layer
and an optional decorrelation penalty (dpd/disentangle.py). Loss per batch:
    CE(crop) + CE(disease) + CE(spy_crop) + CE(spy_disease) + decorrelation_weight * decorrelation(z_crop, z_dis)
The spies always learn (their accuracy is logged as a leakage measure); adversarial_weight only sets how strongly the
reversed gradient pushes the branches, so adversarial_weight 0 + decorrelation_weight 0 is the "branches only" control.

Same two stages as train/train_cnn.py: pick (epoch, gamma) on 3 validation splits with pseudo-unseen pairs, then refit
on all training images for that many epochs and evaluate once on test.
"""

import argparse
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dpd.cnn import build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.disentangle import DisentangledNet, adversarial_ramp, decorrelation_loss
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from train.train_cnn import make_loader, seed_everything

METHOD = "cnn_disentangle"
NETWORK = "disentangle_net"


def build_model(cfg, pretrained=None):
    m = cfg["model"]
    return DisentangledNet(m["backbone"], m["pretrained"] if pretrained is None else pretrained, m["z_dim"], m["spy_hidden"], m["head_dropout"])


@torch.no_grad()
def predict(model, loader, device, amp) -> dict[str, np.ndarray]:
    """Log-probabilities of the crop and disease heads for every image in the loader."""
    model.eval()
    out = {"crop": [], "disease": []}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            o = model(x.to(device, non_blocking=True))
        for k in out:
            out[k].append(F.log_softmax(o[k].float(), dim=-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def train_network(train_df, epochs, cfg, image_dir, tfs, eval_loader=None, log=None):
    """Train one DisentangledNet; if eval_loader is given, return its head log-probs after every epoch."""
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    ocfg, lcfg, dcfg = cfg["optimizer"], cfg["loss"], cfg["disentangle"]
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
            {"params": model.head_parameters(), "lr": ocfg["lr_head"]},
        ],
        weight_decay=ocfg["weight_decay"],
    )
    loader = make_loader(train_df, image_dir, tfs[0], cfg, shuffle=True)
    total_steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    # The adversarial ramp counts in epochs out of max_epochs, so the stage-2 refit applies the same strength at a given
    # epoch as stage 1 did when that epoch was selected (a ramp over the refit's own length reached full strength early).
    ramp_steps = cfg["training"]["max_epochs"] * len(loader)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    ls, w = lcfg["label_smoothing"], lcfg["weights"]
    per_epoch, step = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, n = time.time(), 0
        sums = dict.fromkeys(["loss", "decor", "crop_acc", "disease_acc", "spy_crop_acc", "spy_disease_acc"], 0.0)
        for x, y_crop, y_dis in loader:
            x, y_crop, y_dis = x.to(device, non_blocking=True), y_crop.to(device), y_dis.to(device)
            ramp = adversarial_ramp(step / ramp_steps) if dcfg["adversarial_ramp"] else 1.0
            with torch.autocast(device, enabled=amp):
                o = model(x)
                spies = model.spies(o["z_crop"], o["z_dis"], dcfg["adversarial_weight"] * ramp)
                decor = decorrelation_loss(o["z_crop"].float(), o["z_dis"].float())
                loss = (
                    w["crop"] * F.cross_entropy(o["crop"].float(), y_crop, label_smoothing=ls)
                    + w["disease"] * F.cross_entropy(o["disease"].float(), y_dis, label_smoothing=ls)
                    + F.cross_entropy(spies["spy_crop"].float(), y_crop)
                    + F.cross_entropy(spies["spy_disease"].float(), y_dis)
                    + dcfg["decorrelation_weight"] * decor
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            b = len(x)
            n += b
            sums["loss"] += loss.item() * b
            sums["decor"] += decor.item() * b
            for key, logits, y in (("crop_acc", o["crop"], y_crop), ("disease_acc", o["disease"], y_dis),
                                   ("spy_crop_acc", spies["spy_crop"], y_crop), ("spy_disease_acc", spies["spy_disease"], y_dis)):
                sums[key] += (logits.argmax(1) == y).sum().item()
        rec = {"epoch": epoch, **{(k if k in ("loss", "decor") else f"train_{k}"): round(v / n, 4) for k, v in sums.items()},
               "adv_lambda": round(dcfg["adversarial_weight"] * ramp, 4), "sec": round(time.time() - t0)}
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
    assert cfg["method"] == METHOD, f"{args.config} is a '{cfg['method']}' config"
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

    # ---- Stage 1: choose epoch and gamma on validation (same procedure as train/train_cnn.py) ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(gammas)))
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, _ = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        n_vs = len(val_seen)
        eval_loader = make_loader(pd.concat([val_seen, val_unseen]), image_dir, tfs[1], cfg, shuffle=False)
        seed_everything(split_seed)
        _, per_epoch = train_network(fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, eval_loader, logger(f"split{k + 1}/{NETWORK}"))
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        for e, logp in enumerate(per_epoch):
            s = compose_pair_scores(logp["crop"], logp["disease"], pairs)
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
    seed_everything(cfg["seed"])
    model, _ = train_network(train, best_epoch, cfg, image_dir, tfs, log=logger(f"final/{NETWORK}"))
    torch.save(model.state_dict(), out / f"{NETWORK}.pt")

    scores, labels, files = {}, {}, {}
    for s, loader in test_loaders.items():
        logp = predict(model, loader, runtime["device"], runtime["amp"])
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(logp["crop"], logp["disease"], pairs)
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
