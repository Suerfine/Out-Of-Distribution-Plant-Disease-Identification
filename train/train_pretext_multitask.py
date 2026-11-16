"""Tunable Multi-task CNN with Pretext Task.

Usage:
    python -m train.train_pretext_multitask --config config/pretext/pretext_multitask_convnext_simsiam.json
    python -m train.train_pretext_multitask --config config/pretext/pretext_multitask_convnext_simsiam.json --pilot
    python -m train.train_pretext_multitask --config config/pretext/pretext_multitask_convnext_simsiam.json --set runtime.num_workers=2
    python -m train.train_pretext_multitask --config config/pretext/pretext_multitask_convnext_simsiam.json --resume

Features:
- Joint multi-task classification (crop, disease) + auxiliary pretext task (SimSiam, Barlow Twins)
- Two-view training with independent augmentations
- Validates on pseudo-unseen pairs carved from train (dpd/splits.py)
- Automatic gamma sweep (or disabled with gamma_grid=0) on validation split
- Full refit on all train data with standard 333-pair evaluation
- Exports clean MultiHeadNet weights (multitask_net.pt) compatible with dpd/inference.py and Streamlit app
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpd.cnn import HEAD_SIZES, ImageLabels, MultiHeadNet, build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary, parse_gamma_grid
from dpd.pretext import PretextMultiTaskNet, barlow_twins_loss, simsiam_loss
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from dpd.supcon import TwoViewImageLabels
from train.train_multi_task_cnn import (
    CheckpointManager,
    LossWeighter,
    apply_overrides,
    build_scheduler,
    get_rng_states,
    seed_everything,
    set_backbone_trainable,
    set_rng_states,
    sync_run_to_drive,
)

METHOD = "pretext_multitask_cnn"
NETWORK_NAME = "multitask_net"


def apply_pilot(cfg: dict) -> dict:
    """Shrink a config for a quick end-to-end smoke check."""
    cfg = copy.deepcopy(cfg)
    cfg["version"] = cfg.get("version", "pretext") + "_pilot"
    cfg["output_dir"] = "runs/pretext/_pilot"
    cfg["training"]["max_epochs"] = 2
    cfg["training"]["freeze_backbone_epochs"] = 0
    cfg["training"]["early_stopping"] = {"patience": 0, "min_epochs": 1}
    cfg["validation"]["n_splits"] = 1
    cfg["validation"]["n_holdout_pairs"] = 8
    cfg["validation"]["min_holdout_images"] = 4
    cfg["validation"]["max_images_per_holdout_pair"] = 6
    cfg["validation"]["val_fraction"] = 0.25
    cfg["validation"]["gamma_grid"] = 0
    return cfg


def make_two_view_loader(df, image_dir, transform, cfg, shuffle: bool = True):
    rt = cfg["runtime"]
    return DataLoader(
        TwoViewImageLabels(df, image_dir, transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=rt["num_workers"],
        pin_memory=True,
        persistent_workers=rt["num_workers"] > 0,
        drop_last=shuffle,
    )


def make_single_view_loader(df, image_dir, transform, cfg, shuffle: bool = False):
    rt = cfg["runtime"]
    return DataLoader(
        ImageLabels(df, image_dir, transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=rt["num_workers"],
        pin_memory=True,
        persistent_workers=rt["num_workers"] > 0,
        drop_last=False,
    )


def build_optimizer(model: PretextMultiTaskNet, loss_weighter: LossWeighter, cfg: dict) -> torch.optim.Optimizer:
    ocfg = cfg["optimizer"]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.heads.parameters()) + list(model.pretext_head.parameters())
    param_groups = [
        {"params": backbone_params, "lr": ocfg["lr_backbone"]},
        {"params": head_params, "lr": ocfg["lr_head"]},
    ]
    if loss_weighter.mode == "uncertainty":
        param_groups.append({
            "params": loss_weighter.parameters(),
            "lr": ocfg["lr_head"],
            "weight_decay": 0.0,
        })

    name = ocfg.get("name", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=ocfg["weight_decay"])
    elif name == "sgd":
        return torch.optim.SGD(
            param_groups,
            weight_decay=ocfg["weight_decay"],
            momentum=ocfg.get("momentum", 0.9),
            nesterov=ocfg.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer {name!r}")


def train_pretext_epoch(
    model: PretextMultiTaskNet,
    loss_weighter: LossWeighter,
    opt: torch.optim.Optimizer,
    sched,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    heads: tuple[str, ...],
    device: str,
    amp: bool,
    freeze: bool,
    accum_steps: int,
    clip_norm: float,
    pretext_cfg: dict,
    desc: str = "",
):
    model.train()
    set_backbone_trainable(model, not freeze)

    tot_loss, tot_ce, tot_pretext = 0.0, 0.0, 0.0
    correct = {h: 0 for h in heads}
    n = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    lambda_pretext = float(pretext_cfg.get("lambda_pretext", pretext_cfg.get("lambda_ssl", 0.5)))
    pretext_type = model.pretext_type
    n_batches = len(loader)
    log_interval = max(1, n_batches // 5)

    for step_idx, (v1, v2, y_crop, y_dis) in enumerate(loader):
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        targets = {"crop": y_crop.to(device, non_blocking=True), "disease": y_dis.to(device, non_blocking=True)}

        with torch.autocast(device, enabled=amp):
            # Forward pass on both views
            f1 = model.forward_backbone(v1)
            f2 = model.forward_backbone(v2)

            logits1 = model.forward_heads(f1)
            logits2 = model.forward_heads(f2)

            # Symmetrized classification loss
            loss_ce1, stats1 = loss_weighter(logits1, targets)
            loss_ce2, stats2 = loss_weighter(logits2, targets)
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            # Pretext task loss
            if pretext_type == "simsiam":
                z1, p1 = model.forward_pretext(f1)
                z2, p2 = model.forward_pretext(f2)
                loss_pretext = simsiam_loss(p1, z1, p2, z2)
            elif pretext_type == "barlow_twins":
                z1 = model.forward_pretext(f1)
                z2 = model.forward_pretext(f2)
                loss_pretext = barlow_twins_loss(z1, z2, lambda_offdiag=pretext_cfg.get("lambda_offdiag", 0.005))
            else:
                loss_pretext = torch.tensor(0.0, device=device)

            total_loss = loss_ce + lambda_pretext * loss_pretext
            loss_scaled = total_loss / accum_steps

        scaler.scale(loss_scaled).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == n_batches:
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scale_before <= scaler.get_scale() and sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)

        batch_sz = len(v1)
        tot_loss += total_loss.item() * batch_sz
        tot_ce += loss_ce.item() * batch_sz
        tot_pretext += loss_pretext.item() * batch_sz
        n += batch_sz

        for h in heads:
            pred1 = logits1[h].argmax(1)
            pred2 = logits2[h].argmax(1)
            correct[h] += ((pred1 == targets[h]).sum().item() + (pred2 == targets[h]).sum().item()) / 2.0

        if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == n_batches:
            prefix = f"  [{desc}] " if desc else "  "
            acc_str = " ".join(f"{h}_acc={correct[h] / max(1, n):.1%}" for h in heads)
            print(
                f"{prefix}Step {step_idx + 1}/{n_batches} | total_loss={tot_loss / max(1, n):.4f} "
                f"ce={tot_ce / max(1, n):.4f} pretext={tot_pretext / max(1, n):.4f} | {acc_str}",
                flush=True,
            )

    sec = round(time.time() - t0)
    train_acc = {f"train_{h}_acc": round(correct[h] / max(1, n), 4) for h in heads}
    rec = {
        "loss": round(tot_loss / max(1, n), 4),
        "loss_ce": round(tot_ce / max(1, n), 4),
        "loss_pretext": round(tot_pretext / max(1, n), 4),
        **train_acc,
        "sec": sec,
    }
    return rec


@torch.no_grad()
def predict_single_view(model: PretextMultiTaskNet, loader: DataLoader, device: str, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    out = {k: [] for k in model.heads}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        for k, v in logits.items():
            out[k].append(F.log_softmax(v.float(), dim=-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def score_split(model, fit_df, val_seen_df, val_unseen_df, pairs, image_dir, tf_val, cfg, gammas, hidden):
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    loader_seen = make_single_view_loader(val_seen_df, image_dir, tf_val, cfg, shuffle=False)
    loader_unseen = make_single_view_loader(val_unseen_df, image_dir, tf_val, cfg, shuffle=False)

    logp_seen = predict_single_view(model, loader_seen, device, amp)
    logp_unseen = predict_single_view(model, loader_unseen, device, amp)

    scores_seen = compose_pair_scores(logp_seen["crop"], logp_seen["disease"], pairs)
    scores_unseen = compose_pair_scores(logp_unseen["crop"], logp_unseen["disease"], pairs)

    y_seen = val_seen_df["pair_id"].to_numpy()
    y_unseen = val_unseen_df["pair_id"].to_numpy()

    # Fast sweep across gammas
    hms = []
    for g in gammas:
        s_seen_adj = scores_seen.copy()
        s_unseen_adj = scores_unseen.copy()
        if g > 0.0:
            s_seen_adj[:, ~hidden] -= g
            s_unseen_adj[:, ~hidden] -= g
        pred_seen = s_seen_adj.argmax(axis=1)
        pred_unseen = s_unseen_adj.argmax(axis=1)
        acc_s = (pred_seen == y_seen).mean()
        acc_u = (pred_unseen == y_unseen).mean()
        hm = (2.0 * acc_s * acc_u / (acc_s + acc_u)) if (acc_s + acc_u) > 0 else 0.0
        hms.append(hm)
    return np.array(hms)


def main():
    parser = argparse.ArgumentParser(description="Multi-task CNN with Pretext Task")
    parser.add_argument("--config", required=True, help="Path to config json")
    parser.add_argument("--pilot", action="store_true", help="Run a quick 2-epoch smoke test")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="Config overrides like runtime.num_workers=2")
    parser.add_argument("--sync-dir", default=None, help="Directory to sync checkpoints and outputs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pilot:
        cfg = apply_pilot(cfg)
    assert cfg["method"] == METHOD, f"Config method must be {METHOD!r}, got {cfg.get('method')!r}"

    if args.overrides:
        applied = apply_overrides(cfg, args.overrides)
        cfg["_overrides"] = applied
        print("Applied overrides:\n  " + "\n  ".join(applied), flush=True)

    runtime = cfg["runtime"]
    tcfg = cfg["training"]
    vcfg = cfg["validation"]
    mcfg = cfg["model"]
    ocfg = cfg["optimizer"]
    lcfg = cfg["loss"]
    pretext_cfg = lcfg.get("pretext", lcfg.get("ssl", {}))

    device = runtime["device"]
    amp = runtime["amp"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")

    torch.backends.cudnn.benchmark = True
    out = output_dir(cfg)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    sync_base = args.sync_dir or runtime.get("sync_dir")
    sync_target = None
    if sync_base:
        sync_base = Path(sync_base)
        rel_out = Path(cfg["output_dir"])
        sync_target = (sync_base / Path(*rel_out.parts[1:])) if len(rel_out.parts) > 1 and rel_out.parts[0] == "runs" else (sync_base / rel_out)
        sync_target.mkdir(parents=True, exist_ok=True)
        print(f"Live synchronization enabled -> {sync_target}", flush=True)

    sync_ckpt_dir = (sync_target / "checkpoints") if sync_target is not None else None
    ckpt_mgr = CheckpointManager(
        out / "checkpoints",
        n_every=runtime.get("checkpoint_every_n_epochs", 5),
        keep=runtime.get("keep_checkpoints", False),
        sync_ckpt_dir=sync_ckpt_dir,
    )

    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    tf_train, tf_val = tfs[0], tfs[1]
    train_all = df[df["split"] == "train"]

    history = []
    def log_record(rec: dict):
        history.append(rec)
        print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if sync_target is not None:
            try:
                shutil.copy2(out / "history.csv", sync_target / "history.csv")
            except Exception:
                pass

    gammas = parse_gamma_grid(vcfg.get("gamma_grid"))
    if len(gammas) == 1 and gammas[0] == 0.0:
        print("Gamma tuning disabled (gamma_grid=0): evaluating uncalibrated (gamma=0.0) across all stages.", flush=True)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 3)
    max_epochs = tcfg["max_epochs"]
    heads = ("crop", "disease")

    hm_grid = np.full((n_splits, max_epochs, len(gammas)), np.nan)
    start_stage = "stage1"
    start_split = 0
    start_epoch = 1

    # Resume check
    if args.resume:
        ckpt = ckpt_mgr.load()
        if ckpt:
            print(f"Resuming from checkpoint (stage={ckpt['stage']}, split={ckpt.get('split')}, epoch={ckpt['epoch']})", flush=True)
            start_stage = ckpt["stage"]
            start_split = ckpt.get("split", 0)
            start_epoch = ckpt["epoch"] + 1
            hm_grid = ckpt.get("hm_grid", hm_grid)
            history = ckpt.get("history", [])
            set_rng_states(ckpt.get("rng_states", {}))

    print(f"=== Stage 1: Validation ({n_splits} splits x {max_epochs} epochs) ===", flush=True)
    if start_stage == "stage1":
        for s_idx in range(start_split, n_splits):
            seed = cfg["seed"] + s_idx
            seed_everything(seed)
            fit_df, val_seen_df, val_unseen_df, holdout = make_validation_split(
                train_all,
                n_holdout_pairs=vcfg["n_holdout_pairs"],
                min_holdout_images=vcfg["min_holdout_images"],
                val_fraction=vcfg["val_fraction"],
                seed=seed,
                exclude_diseases=tuple(vcfg.get("exclude_holdout_diseases", ())),
                max_images_per_holdout_pair=vcfg.get("max_images_per_holdout_pair"),
            )
            print(f"\n--- Split {s_idx + 1}/{n_splits} (fit={len(fit_df)}, val_seen={len(val_seen_df)}, val_unseen={len(val_unseen_df)}) ---", flush=True)

            model = PretextMultiTaskNet(
                backbone=mcfg["backbone"],
                pretrained=mcfg["pretrained"],
                heads={h: HEAD_SIZES[h] for h in heads},
                dropout=mcfg.get("head_dropout", 0.3),
                pretext_type=pretext_cfg.get("type", "simsiam"),
                pretext_proj_dim=pretext_cfg.get("proj_dim", 512),
                pretext_hidden_dim=pretext_cfg.get("hidden_dim", 2048),
                pretext_pred_hidden=pretext_cfg.get("pred_hidden", 512),
                drop_path_rate=mcfg.get("drop_path_rate", 0.0),
            ).to(device)

            loss_weighter = LossWeighter(lcfg, heads).to(device)
            opt = build_optimizer(model, loss_weighter, cfg)
            loader_train = make_two_view_loader(fit_df, image_dir, tf_train, cfg, shuffle=True)
            sched = build_scheduler(opt, len(loader_train) // tcfg.get("accum_steps", 1), max_epochs, cfg)
            scaler = torch.amp.GradScaler(device, enabled=amp)

            first_ep = start_epoch if s_idx == start_split else 1
            patience = tcfg.get("early_stopping", {}).get("patience", 0)
            min_epochs = tcfg.get("early_stopping", {}).get("min_epochs", 1)
            best_hm = -1.0
            epochs_no_improve = 0

            for epoch in range(first_ep, max_epochs + 1):
                freeze = epoch <= tcfg.get("freeze_backbone_epochs", 0)
                train_rec = train_pretext_epoch(
                    model=model,
                    loss_weighter=loss_weighter,
                    opt=opt,
                    sched=sched,
                    scaler=scaler,
                    loader=loader_train,
                    heads=heads,
                    device=device,
                    amp=amp,
                    freeze=freeze,
                    accum_steps=tcfg.get("accum_steps", 1),
                    clip_norm=tcfg.get("grad_clip_norm", 0.0),
                    pretext_cfg=pretext_cfg,
                    desc=f"Split {s_idx + 1} Ep {epoch}",
                )

                hms = score_split(model, fit_df, val_seen_df, val_unseen_df, pairs, image_dir, tf_val, cfg, gammas, hidden)
                hm_grid[s_idx, epoch - 1] = hms
                best_gamma_idx = int(np.nanargmax(hms))
                ep_best_hm = hms[best_gamma_idx]
                best_g = gammas[best_gamma_idx]

                log_record({
                    "stage": "val",
                    "split": s_idx,
                    "epoch": epoch,
                    "val_hm": round(float(ep_best_hm), 4),
                    "best_gamma": round(float(best_g), 2),
                    **train_rec,
                })

                if patience > 0 and epoch >= min_epochs:
                    if ep_best_hm > best_hm + 1e-4:
                        best_hm = ep_best_hm
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered on Split {s_idx + 1} at epoch {epoch}", flush=True)
                            break

                if ckpt_mgr.should_save(epoch, epoch == max_epochs):
                    ckpt_mgr.save({
                        "stage": "stage1",
                        "split": s_idx,
                        "epoch": epoch,
                        "hm_grid": hm_grid,
                        "history": history,
                        "rng_states": get_rng_states(),
                    })
            start_epoch = 1

    # Choose epoch and gamma across splits
    mean_hm = np.nanmean(hm_grid, axis=0)
    best_ep_idx, best_g_idx = np.unravel_index(np.nanargmax(mean_hm), mean_hm.shape)
    chosen_epoch = int(best_ep_idx + 1)
    chosen_gamma = float(gammas[best_g_idx])
    chosen_val_hm = float(mean_hm[best_ep_idx, best_g_idx])
    print(f"\nValidation Selection -> Chosen Epoch: {chosen_epoch}, Chosen Gamma: {chosen_gamma:.2f} (mean val HM: {chosen_val_hm:.4f})", flush=True)

    # Stage 2: Final refit on all train data
    print(f"\n=== Stage 2: Refit on all {len(train_all)} train images for {chosen_epoch} epochs ===", flush=True)
    seed_everything(cfg["seed"])
    final_model = PretextMultiTaskNet(
        backbone=mcfg["backbone"],
        pretrained=mcfg["pretrained"],
        heads={h: HEAD_SIZES[h] for h in heads},
        dropout=mcfg.get("head_dropout", 0.3),
        pretext_type=pretext_cfg.get("type", "simsiam"),
        pretext_proj_dim=pretext_cfg.get("proj_dim", 512),
        pretext_hidden_dim=pretext_cfg.get("hidden_dim", 2048),
        pretext_pred_hidden=pretext_cfg.get("pred_hidden", 512),
        drop_path_rate=mcfg.get("drop_path_rate", 0.0),
    ).to(device)

    loss_weighter = LossWeighter(lcfg, heads).to(device)
    opt = build_optimizer(final_model, loss_weighter, cfg)
    loader_all = make_two_view_loader(train_all, image_dir, tf_train, cfg, shuffle=True)
    sched = build_scheduler(opt, len(loader_all) // tcfg.get("accum_steps", 1), chosen_epoch, cfg)
    scaler = torch.amp.GradScaler(device, enabled=amp)

    for epoch in range(1, chosen_epoch + 1):
        freeze = epoch <= tcfg.get("freeze_backbone_epochs", 0)
        train_rec = train_pretext_epoch(
            model=final_model,
            loss_weighter=loss_weighter,
            opt=opt,
            sched=sched,
            scaler=scaler,
            loader=loader_all,
            heads=heads,
            device=device,
            amp=amp,
            freeze=freeze,
            accum_steps=tcfg.get("accum_steps", 1),
            clip_norm=tcfg.get("grad_clip_norm", 0.0),
            pretext_cfg=pretext_cfg,
            desc=f"Refit Ep {epoch}/{chosen_epoch}",
        )
        log_record({"stage": "refit", "epoch": epoch, **train_rec})

    # Save clean MultiHeadNet weights for inference and Streamlit app
    clean_net = MultiHeadNet(mcfg["backbone"], False, {h: HEAD_SIZES[h] for h in heads}, mcfg.get("head_dropout", 0.3))
    clean_net.backbone.load_state_dict(final_model.backbone.state_dict())
    clean_net.heads.load_state_dict(final_model.heads.state_dict())
    torch.save(clean_net.state_dict(), out / f"{NETWORK_NAME}.pt")
    print(f"Saved {NETWORK_NAME}.pt to {out}", flush=True)

    # Benchmark evaluation on test splits
    print("\n=== Benchmark Evaluation on Test Splits ===", flush=True)
    test_loaders = {s: make_single_view_loader(df[df["split"] == s], image_dir, tf_val, cfg, shuffle=False) for s in ("test_seen", "test_unseen")}
    test_logp = {s: predict_single_view(final_model, test_loaders[s], device, amp) for s in test_loaders}

    del final_model, loss_weighter, opt, sched, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores, labels, files = {}, {}, {}
    for s in test_loaders:
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(test_logp[s]["crop"], test_logp[s]["disease"], pairs)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()

    top_k = tuple(cfg.get("evaluation", {}).get("top_k", [1, 5]))
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, chosen_gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(
        version=cfg["version"],
        chosen_epoch=chosen_epoch,
        selected_epoch=chosen_epoch,
        best_epoch=chosen_epoch,
        val_hm=round(chosen_val_hm, 4),
        harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"],
    )
    no_cal.update(
        version=cfg["version"],
        chosen_epoch=chosen_epoch,
        selected_epoch=chosen_epoch,
        best_epoch=chosen_epoch,
        val_hm=round(chosen_val_hm, 4),
        harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"],
    )

    print("\n================ Test Results (Uncalibrated, gamma=0) ================")
    print(format_summary(no_cal))
    if chosen_gamma > 0.0:
        print(f"\n================ Test Results (Calibrated, gamma={chosen_gamma:.2f}) ===========")
        print(format_summary(result))

    print("\n" + "=" * 65)
    print("                    TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Model Version:         {cfg['version']}")
    print(f"  Backbone:              {mcfg['backbone']}")
    print(f"  Best Selected Epoch:   {chosen_epoch}")
    print(f"  Tuned Gamma:           {chosen_gamma}")
    print(f"  Validation Mean HM:    {chosen_val_hm:.2%}")
    print("-" * 65)
    print(f"  Test Seen Top-1:       {no_cal['test_seen']['top1']:.2%}")
    print(f"  Test Unseen Top-1:     {no_cal['test_unseen']['top1']:.2%}")
    print(f"  Test Harmonic Mean:    {no_cal['harmonic_mean_top1']:.2%}")
    print("=" * 65)

    save_test_outputs(out, result if chosen_gamma > 0.0 else no_cal, scores, labels, files, pairs, chosen_gamma)
    # Save both uncalibrated metrics (metrics.json) and gamma-tuned metrics (metrics_gamma.json)
    (out / "metrics_gamma.json").write_text(json.dumps(result, indent=2))
    (out / "metrics.json").write_text(json.dumps(no_cal, indent=2))
    np.savez_compressed(
        out / "head_scores.npz",
        **{f"{s}_{h}_logp": test_logp[s][h] for s in test_logp for h in test_logp[s]}
    )
    np.save(out / "val_hm_grid.npy", hm_grid)
    print(f"\nFinished run successfully. Metrics written to {out / 'metrics.json'}", flush=True)
    if sync_target:
        sync_run_to_drive(out, sync_target)
    ckpt_mgr.cleanup()


if __name__ == "__main__":
    main()
