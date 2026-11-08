"""Compositional contrastive model (method cnn_compositional): image-to-pair InfoNCE with held-out (pretend-unseen) pairs.

Same two-stage protocol as train/train_supcon.py: stage 1 trains on 3 validation splits built from the training set only and
picks the epoch and gamma by the mean validation harmonic mean; stage 2 refits on all training images for that epoch (on the
same max_epochs cosine schedule, so it matches what was validated) and evaluates once on the test sets.

Loss (every step, both augmented views): scores s = scale * z . p(c, d) over the training pairs; a random `holdout_frac` of
the training pairs play the unseen pairs and every other pair's score is shifted down by `gamma0`, as calibration does at
test time; then cross-entropy (label smoothing) over the training pairs. This is a contrastive loss between an image and
the embeddings of all candidate (crop, disease) pairs.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpd.cnn import ImageLabels, build_transforms
from dpd.compositional import CompositionalNet, pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from train.train_cnn import seed_everything
from train.train_supcon import PILOT_IMAGES_PER_PAIR, PILOT_TEST_IMAGES, apply_pilot, make_two_view_loader


def build_model(cfg, pairs):
    m = cfg["model"]
    return CompositionalNet(m["backbone"], m["pretrained"], pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(), m["emb_dim"],
                            m["hidden"], m["head_dropout"], m.get("drop_path_rate", 0.0), m["init_scale"])


def train_network(train_df, epochs, cfg, image_dir, tfs, pairs, on_epoch=None, log=None, schedule_epochs=None):
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    ocfg, pl = cfg["optimizer"], cfg["loss"]["pair_loss"]
    model = build_model(cfg, pairs).to(device)
    if cfg["runtime"].get("grad_checkpointing") and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
        {"params": model.head_parameters(), "lr": ocfg["lr_head"]},
        {"params": [model.log_scale], "lr": ocfg["lr_head"], "weight_decay": 0.0},
    ], weight_decay=ocfg["weight_decay"])
    loader = make_two_view_loader(train_df, image_dir, tfs[0], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(schedule_epochs or epochs) * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    pair_of = -torch.ones(55, 175, dtype=torch.long)
    pair_of[torch.as_tensor(pairs["crop_id"].to_numpy()), torch.as_tensor(pairs["disease_id"].to_numpy())] = torch.as_tensor(pairs["pair_id"].to_numpy())
    pair_of = pair_of.to(device)
    cand = torch.zeros(len(pairs), dtype=torch.bool, device=device)
    cand[torch.as_tensor(train_df["pair_id"].unique()).to(device)] = True
    cand_idx = cand.nonzero().squeeze(1)
    position = -torch.ones(len(pairs), dtype=torch.long, device=device)
    position[cand_idx] = torch.arange(len(cand_idx), device=device)
    per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for v1, v2, y_crop, y_dis in loader:
            x = torch.cat([v1, v2]).to(device, non_blocking=True)
            y_pair = pair_of[torch.cat([y_crop, y_crop]).to(device), torch.cat([y_dis, y_dis]).to(device)]
            with torch.autocast(device, enabled=amp):
                s = model(x)
            hold = (torch.rand(len(cand), device=device) < pl["holdout_frac"]) & cand
            z = (s.float() - pl["gamma0"] * (cand & ~hold).float())[:, cand_idx]
            loss = F.cross_entropy(z, position[y_pair], label_smoothing=pl["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(v1)
            n += len(v1)
        rec = {"epoch": epoch, "loss": round(tot / n, 4), "scale": round(float(model.log_scale.exp()), 2), "sec": round(time.time() - t0)}
        if on_epoch is not None:
            per_epoch.append(on_epoch(model))
        if log:
            log(rec)
    return model, per_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "cnn_compositional", f"{args.config} is a '{cfg['method']}' config"
    if args.pilot:
        cfg = apply_pilot(cfg)
    rt, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    device, amp = rt["device"], rt["amp"]
    out = output_dir(cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    train = df[df["split"] == "train"]
    if args.pilot:
        train = train.sample(frac=1, random_state=cfg["seed"]).groupby("pair_id").head(PILOT_IMAGES_PER_PAIR)
    history = []

    def logger(prefix):
        def log(rec):
            rec = {"stage": prefix, **rec}
            history.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        return log

    def eval_loader(frame):
        return DataLoader(ImageLabels(frame, image_dir, tfs[1]), batch_size=128, shuffle=False, num_workers=rt["num_workers"], pin_memory=True)

    # ---- Stage 1: choose epoch and gamma on validation (training data only) ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(gammas)))
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, _ = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"])
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        val_loader = eval_loader(pd.concat([val_seen, val_unseen]))
        n_vs = len(val_seen)
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        seed_everything(split_seed)
        _, val_scores = train_network(fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, pairs,
                                      lambda model: pair_scores(model, val_loader, device, amp), logger(f"split{k + 1}"))
        for e, s in enumerate(val_scores):
            s[:, hidden] = -1e9
            hm_grid[k, e] = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
            print(f"split {k + 1} epoch {e + 1}: val HM {hm_grid[k, e, 0]:.2%} at gamma 0, best {hm_grid[k, e].max():.2%} at gamma {gammas[hm_grid[k, e].argmax()]}", flush=True)
    mean_grid = hm_grid.mean(axis=0)
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]], "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4)}
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    seed_everything(cfg["seed"])
    model, _ = train_network(train, best_epoch, cfg, image_dir, tfs, pairs, log=logger("final"), schedule_epochs=tcfg["max_epochs"])
    torch.save(model.state_dict(), out / "compositional_net.pt")
    scores, labels, files = {}, {}, {}
    for s in ("test_seen", "test_unseen"):
        sub = df[df["split"] == s]
        if args.pilot:
            sub = sub.sample(n=min(PILOT_TEST_IMAGES, len(sub)), random_state=cfg["seed"])
        scores[s] = pair_scores(model, eval_loader(sub), device, amp)
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
