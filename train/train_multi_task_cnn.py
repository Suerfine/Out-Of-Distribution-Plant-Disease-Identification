"""Tunable Multi-task CNN trainer (spec baseline 2 and extended experiments).

Usage:
    python -m train.train_multi_task_cnn --config config/multitask/nat_multitask_v1.json
    python -m train.train_multi_task_cnn --config config/multitask/nat_multitask_v1.json --set runtime.num_workers=2
    python -m train.train_multi_task_cnn --config config/multitask/nat_multitask_v1.json --sync-dir /content/drive/MyDrive/Nat_AML_DPD_group_assignment/runs
    python -m train.train_multi_task_cnn --config config/multitask/nat_multitask_v1.json --resume

Features:
- Backbones: resnet18/34/50/101, efficientnet_b0..b3, convnext_tiny/small, vit_b_16, and timm backbones (e.g. resnetv2_50x1_bit.goog_in21k_ft_in1k)
- Optimizers: AdamW, SGD (momentum/nesterov)
- Schedulers: Cosine, Step, None, with linear warmup
- Head warm-up: backbone freezing for first N epochs
- Loss weighting: fixed weights or learnable uncertainty weighting (Kendall et al.)
- Training stability: gradient accumulation, gradient clipping
- Validation: n-split cross-validation holding out pseudo-unseen pairs, sweeping gamma
- Early stopping on validation HM
- Atomic checkpointing every N epochs and at phase completions with full resume support
- Live synchronization to a remote/Google Drive directory (--sync-dir)
- Stage 2 refit on all train data, evaluating with the shared benchmark protocol
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
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split

METHOD = "multi_task_cnn"
NETWORK_NAME = "multitask_net"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_overrides(cfg: dict, overrides: list[str]) -> list[str]:
    """Apply key.path=value overrides in-place to cfg with json-parsed types."""
    applied = []
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be format 'key.path=value', got: {item!r}")
        keypath, raw_val = item.split("=", 1)
        keys = keypath.strip().split(".")
        try:
            val = json.loads(raw_val)
        except Exception:
            val = raw_val

        curr = cfg
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                raise KeyError(f"Invalid override path {keypath!r}: '{k}' is not a dict in config")
            curr = curr[k]
        target_k = keys[-1]
        if target_k not in curr:
            raise KeyError(f"Invalid override path {keypath!r}: '{target_k}' not found in config section")
        curr[target_k] = val
        applied.append(f"{keypath}={val!r}")
    return applied


class LossWeighter(nn.Module):
    """Handles fixed loss weighting or learned uncertainty weighting (Kendall et al. 2018)."""

    def __init__(self, loss_cfg: dict, heads: tuple[str, ...] = ("crop", "disease")):
        super().__init__()
        self.mode = loss_cfg.get("weighting", "fixed")
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        self.heads = heads

        if self.mode == "uncertainty":
            init_s = float(loss_cfg.get("init_log_var", 0.0))
            self.log_vars = nn.ParameterDict({
                h: nn.Parameter(torch.tensor(init_s, dtype=torch.float32))
                for h in heads
            })
        elif self.mode == "fixed":
            self.weights = {h: float(loss_cfg.get("weights", {}).get(h, 1.0)) for h in heads}
        else:
            raise ValueError(f"Unknown loss weighting mode {self.mode!r}")

    def forward(self, logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        ce = {}
        for h in self.heads:
            ce[h] = F.cross_entropy(logits[h].float(), targets[h], label_smoothing=self.label_smoothing)

        if self.mode == "uncertainty":
            loss = torch.tensor(0.0, device=logits[self.heads[0]].device)
            stats = {}
            for h in self.heads:
                s = self.log_vars[h]
                precision = torch.exp(-s)
                loss = loss + precision * ce[h] + s
                stats[f"loss_{h}"] = round(ce[h].item(), 4)
                stats[f"s_{h}"] = round(s.item(), 4)
                stats[f"w_{h}"] = round(precision.item(), 4)
            return loss, stats
        else:
            loss = sum(self.weights[h] * ce[h] for h in self.heads)
            stats = {f"loss_{h}": round(ce[h].item(), 4) for h in self.heads}
            return loss, stats


def set_backbone_trainable(model: MultiHeadNet, trainable: bool):
    """Freeze or unfreeze backbone parameters and freeze BatchNorm statistics if not trainable."""
    for p in model.backbone.parameters():
        p.requires_grad = trainable
    if not trainable:
        model.backbone.eval()


def build_optimizer(model: MultiHeadNet, loss_weighter: LossWeighter, cfg: dict) -> torch.optim.Optimizer:
    ocfg = cfg["optimizer"]
    param_groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": ocfg["lr_backbone"]},
        {"params": model.heads.parameters(), "lr": ocfg["lr_head"]},
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


def build_scheduler(opt: torch.optim.Optimizer, steps_per_epoch: int, total_epochs: int, cfg: dict):
    scfg = cfg["scheduler"]
    name = scfg.get("name", "cosine").lower()
    warmup_epochs = scfg.get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(total_epochs * steps_per_epoch, 1)
    min_lr_ratio = scfg.get("min_lr_ratio", 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))
        elif name == "step":
            epoch = step // max(1, steps_per_epoch)
            factor = scfg.get("step_factor", 0.1)
            step_size = scfg.get("step_size_epochs", 7)
            return factor ** (epoch // step_size)
        elif name == "none":
            return 1.0
        else:
            raise ValueError(f"Unknown scheduler {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_loader(df, image_dir, transform, cfg, shuffle: bool):
    rt = cfg["runtime"]
    return DataLoader(
        ImageLabels(df, image_dir, transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=rt["num_workers"],
        pin_memory=True,
        persistent_workers=rt["num_workers"] > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def predict(model: MultiHeadNet, loader: DataLoader, device: str, amp: bool) -> dict[str, np.ndarray]:
    """Log-probabilities for crop and disease heads for all images in loader."""
    model.eval()
    out = {k: [] for k in model.heads}
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        for k, v in logits.items():
            out[k].append(F.log_softmax(v.float(), dim=-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def get_rng_states():
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict):
    if not states:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, n_every: int, keep: bool, sync_ckpt_dir: Path | None = None):
        self.ckpt_dir = ckpt_dir
        self.n_every = n_every
        self.keep = keep
        self.sync_ckpt_dir = sync_ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.ckpt_dir / "last.pt"

    def should_save(self, epoch: int, is_phase_end: bool) -> bool:
        if self.n_every <= 0:
            return False
        return (epoch % self.n_every == 0) or is_phase_end

    def save(self, payload: dict):
        if self.n_every <= 0:
            return
        tmp_path = self.ckpt_dir / "last.pt.tmp"
        torch.save(payload, tmp_path)
        if tmp_path.exists():
            os.replace(tmp_path, self.path)
            if self.sync_ckpt_dir is not None:
                try:
                    self.sync_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.path, self.sync_ckpt_dir / "last.pt")
                except Exception as e:
                    print(f"Warning: Failed to sync checkpoint to Drive: {e}", flush=True)

    def load(self) -> dict | None:
        if self.path.exists():
            return torch.load(self.path, map_location="cpu")
        return None

    def cleanup(self):
        if not self.keep and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        if not self.keep and self.sync_ckpt_dir is not None:
            sync_path = self.sync_ckpt_dir / "last.pt"
            if sync_path.exists():
                try:
                    sync_path.unlink()
                except OSError:
                    pass


def sync_run_to_drive(out_dir: Path, sync_target: Path | None):
    """Mirror run directory files and checkpoints to Google Drive/sync target."""
    if sync_target is None:
        return
    try:
        sync_target.mkdir(parents=True, exist_ok=True)
        for item in out_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, sync_target / item.name)
            elif item.is_dir() and item.name == "checkpoints":
                ckpt_dest = sync_target / "checkpoints"
                ckpt_dest.mkdir(parents=True, exist_ok=True)
                for f in item.iterdir():
                    if f.is_file():
                        shutil.copy2(f, ckpt_dest / f.name)
    except Exception as e:
        print(f"Warning: Failed to sync outputs to {sync_target}: {e}", flush=True)


def train_epoch(model, loss_weighter, opt, sched, scaler, loader, heads, device, amp, freeze, accum_steps, clip_norm, desc: str = ""):
    model.train()
    if freeze:
        set_backbone_trainable(model, False)
    else:
        set_backbone_trainable(model, True)

    tot_loss, correct, n = 0.0, {h: 0 for h in heads}, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    n_batches = len(loader)
    log_interval = max(1, n_batches // 5)  # 5 progress checkpoints per epoch

    for step_idx, (x, y_crop, y_dis) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        targets = {"crop": y_crop.to(device), "disease": y_dis.to(device)}

        with torch.autocast(device, enabled=amp):
            logits = model(x)
            loss, loss_stats = loss_weighter(logits, targets)
            loss_scaled = loss / accum_steps

        scaler.scale(loss_scaled).backward()

        if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == n_batches:
            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scale_before <= scaler.get_scale():
                sched.step()
            opt.zero_grad(set_to_none=True)

        batch_sz = len(x)
        tot_loss += loss.item() * batch_sz
        n += batch_sz
        for h in heads:
            correct[h] += (logits[h].argmax(1) == targets[h]).sum().item()

        if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == n_batches:
            prefix = f"  [{desc}] " if desc else "  "
            acc_str = " ".join(f"{h}_acc={correct[h] / max(1, n):.1%}" for h in heads)
            print(f"{prefix}Step {step_idx + 1}/{n_batches} | running_loss={tot_loss / max(1, n):.4f} | {acc_str}", flush=True)

    sec = round(time.time() - t0)
    train_acc = {f"train_{h}_acc": round(correct[h] / max(1, n), 4) for h in heads}
    rec = {"loss": round(tot_loss / max(1, n), 4), **train_acc, "sec": sec}
    rec.update(loss_stats)
    return rec


def main():
    parser = argparse.ArgumentParser(description="Multi-task CNN training with extended options")
    parser.add_argument("--config", required=True, help="Path to config json")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="Config overrides like runtime.num_workers=2")
    parser.add_argument("--sync-dir", default=None, help="Directory to sync checkpoints and outputs (e.g. Google Drive runs folder)")
    parser.add_argument("--resume", action="store_true", help="Resume training from runs/.../checkpoints/last.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg["method"] == METHOD, f"Config method must be {METHOD!r}, got {cfg.get('method')!r}"

    applied_overrides = []
    if args.overrides:
        applied_overrides = apply_overrides(cfg, args.overrides)
        cfg["_overrides"] = applied_overrides
        print("Applied overrides:\n  " + "\n  ".join(applied_overrides), flush=True)

    runtime = cfg["runtime"]
    tcfg = cfg["training"]
    vcfg = cfg["validation"]
    ev = cfg["evaluation"]
    mcfg = cfg["model"]
    ocfg = cfg["optimizer"]
    lcfg = cfg["loss"]

    device = runtime["device"]
    amp = runtime["amp"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")

    torch.backends.cudnn.benchmark = True

    out = output_dir(cfg)
    # Save the resolved configuration
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Determine sync target if requested
    sync_base = args.sync_dir or runtime.get("sync_dir")
    sync_target = None
    if sync_base:
        sync_base = Path(sync_base)
        rel_out = Path(cfg["output_dir"])
        if len(rel_out.parts) > 1 and rel_out.parts[0] == "runs":
            sync_target = sync_base / Path(*rel_out.parts[1:])
        else:
            sync_target = sync_base / rel_out
        sync_target.mkdir(parents=True, exist_ok=True)
        print(f"Live synchronization enabled -> {sync_target}", flush=True)
        # Sync initial config.json
        try:
            shutil.copy2(out / "config.json", sync_target / "config.json")
        except Exception as e:
            print(f"Warning: Failed to copy initial config to sync target: {e}")

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

    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 3)
    max_epochs = tcfg["max_epochs"]
    heads = ("crop", "disease")

    # State containers
    hm_grid = np.full((n_splits, max_epochs, len(gammas)), np.nan)
    start_stage = "stage1"
    start_split = 0
    start_epoch = 1
    patience = tcfg.get("early_stopping", {}).get("patience", 0)
    min_epochs = tcfg.get("early_stopping", {}).get("min_epochs", 5)

    resumed_ckpt = None
    if args.resume:
        # Check if local checkpoint missing but remote exists
        if not ckpt_mgr.path.exists() and sync_target is not None:
            remote_ckpt = sync_target / "checkpoints" / "last.pt"
            if remote_ckpt.exists():
                print(f"Restoring checkpoint from sync directory: {remote_ckpt}")
                shutil.copy2(remote_ckpt, ckpt_mgr.path)
                remote_hist = sync_target / "history.csv"
                if remote_hist.exists():
                    shutil.copy2(remote_hist, out / "history.csv")

        resumed_ckpt = ckpt_mgr.load()
        if resumed_ckpt is not None:
            print(f"Resuming from checkpoint at stage={resumed_ckpt['stage']} split={resumed_ckpt.get('split', 0)} epoch={resumed_ckpt['epoch']}")
            start_stage = resumed_ckpt["stage"]
            start_split = resumed_ckpt.get("split", 0)
            start_epoch = resumed_ckpt["epoch"] + 1
            hm_grid = resumed_ckpt.get("hm_grid", hm_grid)
            history = resumed_ckpt.get("history", [])
            set_rng_states(resumed_ckpt.get("rng_states", {}))
        else:
            print("No checkpoint found to resume, starting fresh.")

    # ---------------- Stage 1: Validation & Selection ----------------
    if start_stage == "stage1":
        for k in range(start_split, n_splits):
            split_seed = cfg["seed"] + k
            fit_df, val_seen, val_unseen, _ = make_validation_split(
                train_all,
                vcfg["n_holdout_pairs"],
                vcfg["min_holdout_images"],
                vcfg["val_fraction"],
                split_seed,
                tuple(vcfg["exclude_holdout_diseases"]),
                vcfg["max_images_per_holdout_pair"],
            )
            print(f"\n--- Validation Split {k + 1}/{n_splits} (fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}) ---", flush=True)

            val_df = pd.concat([val_seen, val_unseen])
            n_vs = len(val_seen)
            val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
            y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()

            train_loader = make_loader(fit_df, image_dir, tfs[0], cfg, shuffle=True)
            eval_loader = make_loader(val_df, image_dir, tfs[1], cfg, shuffle=False)

            # Build model, loss weighter, optimizer, scheduler
            seed_everything(split_seed)
            model = MultiHeadNet(mcfg["backbone"], mcfg["pretrained"], {h: HEAD_SIZES[h] for h in heads}, mcfg["head_dropout"]).to(device)
            loss_weighter = LossWeighter(lcfg, heads).to(device)
            opt = build_optimizer(model, loss_weighter, cfg)
            sched = build_scheduler(opt, len(train_loader) // tcfg.get("accum_steps", 1), max_epochs, cfg)
            scaler = torch.amp.GradScaler(device, enabled=amp)

            curr_epoch_start = 1
            best_split_hm = -1.0
            patience_count = 0

            # If resuming mid-split
            if resumed_ckpt is not None and resumed_ckpt["stage"] == "stage1" and resumed_ckpt.get("split") == k:
                model.load_state_dict(resumed_ckpt["model_state"])
                loss_weighter.load_state_dict(resumed_ckpt["loss_weighter_state"])
                opt.load_state_dict(resumed_ckpt["opt_state"])
                sched.load_state_dict(resumed_ckpt["sched_state"])
                scaler.load_state_dict(resumed_ckpt["scaler_state"])
                curr_epoch_start = start_epoch
                best_split_hm = resumed_ckpt.get("best_split_hm", -1.0)
                patience_count = resumed_ckpt.get("patience_count", 0)
                resumed_ckpt = None  # Consumed

            for epoch in range(curr_epoch_start, max_epochs + 1):
                freeze = (epoch <= tcfg.get("freeze_backbone_epochs", 0))
                freeze_str = " [backbone frozen]" if freeze else ""
                print(f"\n[Stage 1 | Split {k + 1}/{n_splits}] >>> Currently Training Epoch {epoch}/{max_epochs}{freeze_str}...", flush=True)
                rec = train_epoch(
                    model, loss_weighter, opt, sched, scaler, train_loader, heads,
                    device, amp, freeze, tcfg.get("accum_steps", 1), tcfg.get("grad_clip_norm", 0.0),
                    desc=f"Split {k + 1}/{n_splits} Ep {epoch}/{max_epochs}"
                )

                # Validation scoring
                val_logp = predict(model, eval_loader, device, amp)
                s = compose_pair_scores(val_logp["crop"], val_logp["disease"], pairs)
                s[:, hidden] = -1e9
                scores_vs, scores_vu = s[:n_vs], s[n_vs:]

                hms = [evaluate(scores_vs, y_vs, scores_vu, y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
                hm_grid[k, epoch - 1] = hms
                best_g_idx = int(np.argmax(hms))
                best_hm = float(hms[best_g_idx])
                best_gamma = float(gammas[best_g_idx])
                hm_g0 = float(hms[0])

                rec.update({
                    "stage": f"split{k + 1}",
                    "epoch": epoch,
                    "val_hm_best": round(best_hm, 4),
                    "best_gamma": best_gamma,
                    "val_hm_g0": round(hm_g0, 4),
                    "lr_backbone": opt.param_groups[0]["lr"],
                    "lr_head": opt.param_groups[1]["lr"],
                })
                log_record(rec)
                print(f"[Stage 1 | Split {k + 1}/{n_splits}] Finished Epoch {epoch}/{max_epochs} | loss={rec['loss']} | val_hm={best_hm:.2%} (gamma={best_gamma}) | sec={rec['sec']}s", flush=True)

                # Early stopping check
                is_phase_end = (epoch == max_epochs)
                if best_hm > best_split_hm:
                    best_split_hm = best_hm
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience > 0 and epoch >= min_epochs and patience_count >= patience:
                        print(f"Early stopping triggered on split {k + 1} at epoch {epoch} (no improvement for {patience} epochs)")
                        is_phase_end = True

                # Checkpoint
                if ckpt_mgr.should_save(epoch, is_phase_end):
                    ckpt_mgr.save({
                        "stage": "stage1",
                        "split": k,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "loss_weighter_state": loss_weighter.state_dict(),
                        "opt_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "hm_grid": hm_grid,
                        "history": history,
                        "best_split_hm": best_split_hm,
                        "patience_count": patience_count,
                        "rng_states": get_rng_states(),
                    })

                if is_phase_end:
                    break

            del model, loss_weighter, opt, sched, scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        start_stage = "stage2"
        start_epoch = 1
        sync_run_to_drive(out, sync_target)

    # Determine best epoch and gamma from Stage 1
    valid_mask = ~np.isnan(hm_grid).any(axis=(0, 2))
    valid_epochs = np.where(valid_mask)[0]
    if len(valid_epochs) == 0:
        mean_grid = np.nanmean(hm_grid, axis=0)
    else:
        max_valid_e = valid_epochs[-1] + 1
        mean_grid = np.nanmean(hm_grid[:, :max_valid_e, :], axis=0)

    best_e, best_g = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
    best_epoch = int(best_e) + 1
    gamma = float(gammas[best_g])
    best_val_summary = {
        "epoch": best_epoch,
        "gamma": gamma,
        "val_hm_mean": round(float(mean_grid[best_e, best_g]), 4),
        "val_hm_per_split": [round(float(hm_grid[k, best_e, best_g]), 4) if not np.isnan(hm_grid[k, best_e, best_g]) else None for k in range(n_splits)],
        "val_hm_g0_mean": round(float(mean_grid[best_e, 0]), 4),
    }
    print(f"\n=======================================================")
    print(f"Selected epoch {best_epoch}, gamma {gamma}: mean val HM {best_val_summary['val_hm_mean']:.2%}")
    print(f"=======================================================\n", flush=True)

    # ---------------- Stage 2: Final Refit on all Train Data ----------------
    print(f"--- Stage 2: Refit on all training data for {best_epoch} epochs ---", flush=True)
    full_loader = make_loader(train_all, image_dir, tfs[0], cfg, shuffle=True)
    refit_schedule = tcfg.get("refit_schedule", "compress")
    total_refit_sched_epochs = best_epoch if refit_schedule == "compress" else max_epochs

    seed_everything(cfg["seed"])
    model = MultiHeadNet(mcfg["backbone"], mcfg["pretrained"], {h: HEAD_SIZES[h] for h in heads}, mcfg["head_dropout"]).to(device)
    loss_weighter = LossWeighter(lcfg, heads).to(device)
    opt = build_optimizer(model, loss_weighter, cfg)
    sched = build_scheduler(opt, len(full_loader) // tcfg.get("accum_steps", 1), total_refit_sched_epochs, cfg)
    scaler = torch.amp.GradScaler(device, enabled=amp)

    curr_epoch_start = 1
    if resumed_ckpt is not None and resumed_ckpt["stage"] == "stage2":
        model.load_state_dict(resumed_ckpt["model_state"])
        loss_weighter.load_state_dict(resumed_ckpt["loss_weighter_state"])
        opt.load_state_dict(resumed_ckpt["opt_state"])
        sched.load_state_dict(resumed_ckpt["sched_state"])
        scaler.load_state_dict(resumed_ckpt["scaler_state"])
        curr_epoch_start = start_epoch

    for epoch in range(curr_epoch_start, best_epoch + 1):
        freeze = (epoch <= tcfg.get("freeze_backbone_epochs", 0))
        freeze_str = " [backbone frozen]" if freeze else ""
        print(f"\n[Stage 2 Refit] >>> Currently Training Epoch {epoch}/{best_epoch}{freeze_str}...", flush=True)
        rec = train_epoch(
            model, loss_weighter, opt, sched, scaler, full_loader, heads,
            device, amp, freeze, tcfg.get("accum_steps", 1), tcfg.get("grad_clip_norm", 0.0),
            desc=f"Refit Ep {epoch}/{best_epoch}"
        )
        rec.update({
            "stage": "final/multitask_net",
            "epoch": epoch,
            "lr_backbone": opt.param_groups[0]["lr"],
            "lr_head": opt.param_groups[1]["lr"],
        })
        log_record(rec)
        print(f"[Stage 2 Refit] Finished Epoch {epoch}/{best_epoch} | loss={rec['loss']} | sec={rec['sec']}s", flush=True)

        is_phase_end = (epoch == best_epoch)
        if ckpt_mgr.should_save(epoch, is_phase_end):
            ckpt_mgr.save({
                "stage": "stage2",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss_weighter_state": loss_weighter.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(),
                "scaler_state": scaler.state_dict(),
                "hm_grid": hm_grid,
                "history": history,
                "best_epoch": best_epoch,
                "gamma": gamma,
                "rng_states": get_rng_states(),
            })

    # Save model weights
    torch.save(model.state_dict(), out / f"{NETWORK_NAME}.pt")

    # Evaluate once on test splits
    test_loaders = {s: make_loader(df[df["split"] == s], image_dir, tfs[1], cfg, shuffle=False) for s in ("test_seen", "test_unseen")}
    test_logp = {s: predict(model, test_loaders[s], device, amp) for s in test_loaders}

    del model, loss_weighter, opt, sched, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores, labels, files = {}, {}, {}
    for s in test_loaders:
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(test_logp[s]["crop"], test_logp[s]["disease"], pairs)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()

    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(
        version=cfg["version"],
        selected_epoch=best_epoch,
        best_epoch=best_epoch,
        val=best_val_summary,
        harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"]
    )
    no_cal.update(
        version=cfg["version"],
        selected_epoch=best_epoch,
        best_epoch=best_epoch,
        val=best_val_summary,
        harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"]
    )

    print("\n================ Test Results (Uncalibrated, gamma=0) ================")
    print(format_summary(no_cal))
    print("\n================ Test Results (Calibrated, gamma from val) ===========")
    print(format_summary(result))

    print("\n" + "=" * 65)
    print("                    TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Model Version:         {cfg['version']}")
    print(f"  Backbone:              {mcfg['backbone']}")
    print(f"  Total Stage 1 Epochs:  {max_epochs} (across {n_splits} validation split{'s' if n_splits > 1 else ''})")
    print(f"  Best Selected Epoch:   {best_epoch}")
    print(f"  Tuned Gamma:           {gamma}")
    print(f"  Validation Mean HM:    {best_val_summary['val_hm_mean']:.2%}")
    print(f"  Stage 2 Refit Epochs:  {best_epoch}")
    print("-" * 65)
    print(f"  Test Seen Top-1:       {result['test_seen']['top1']:.2%}")
    print(f"  Test Unseen Top-1:     {result['test_unseen']['top1']:.2%}")
    print(f"  Test Harmonic Mean:    {result['harmonic_mean_top1']:.2%}")
    print("=" * 65)

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    # Save both uncalibrated metrics (metrics.json) and gamma-tuned metrics (metrics_gamma.json)
    (out / "metrics_gamma.json").write_text(json.dumps(result, indent=2))
    (out / "metrics.json").write_text(json.dumps(no_cal, indent=2))
    np.savez_compressed(
        out / "head_scores.npz",
        **{f"{s}_{h}_logp": test_logp[s][h] for s in test_logp for h in test_logp[s]}
    )
    np.save(out / "val_hm_grid.npy", hm_grid)
    ckpt_mgr.cleanup()
    sync_run_to_drive(out, sync_target)
    print(f"\nAll outputs saved successfully to {out}")
    if sync_target is not None:
        print(f"All outputs mirrored to Drive at {sync_target}")


if __name__ == "__main__":
    main()
