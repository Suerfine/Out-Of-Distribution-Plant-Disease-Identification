"""Recoverable single-GPU training and validation for the owned baseline study."""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import CachedImages, IndependentNet, freeze_batchnorm_statistics, optimizer, transforms


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StudyLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("Another controller holds this study's OS lock; do not launch a duplicate.") from error
        return self

    def __exit__(self, *_):
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def keep_awake():
    """Prevent idle sleep during the window; never change permanent power settings."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def loader(frame, cache_dir, transform, config, training=False, generator=None, batch_size=None):
    return DataLoader(CachedImages(frame, cache_dir, transform),
                      batch_size=batch_size or config["training"]["batch_size"], shuffle=training,
                      num_workers=config["runtime"]["num_workers"], pin_memory=config["runtime"]["device"] == "cuda",
                      persistent_workers=False, worker_init_fn=worker_seed, generator=generator, drop_last=False)


def lr_factor(epoch, training):
    warmup, horizon = training["warmup_epochs"], training["max_epochs"]
    if warmup and epoch <= warmup:
        return epoch / warmup
    progress = min(1.0, max(0.0, (epoch - warmup) / max(1, horizon - warmup)))
    floor = training["min_lr_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def accumulation_samples(batch_index, batches, batch_size, sample_count, accumulation):
    start = (batch_index // accumulation) * accumulation
    end = min(start + accumulation, batches)
    return min((end - start) * batch_size, sample_count - start * batch_size)


def move_optimizer(opt, device):
    for state in opt.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and name != "step":
                state[name] = value.to(device)


@torch.no_grad()
def predict(model, frame, cache_dir, config, batch_size=None):
    model.eval()
    _, test = transforms(config)
    output = []
    for x, _, _ in loader(frame, cache_dir, test, config, batch_size=batch_size):
        with torch.autocast(config["runtime"]["device"], enabled=config["runtime"]["amp"]):
            output.append(model(x.to(config["runtime"]["device"], non_blocking=True)).float().cpu())
    logits = torch.cat(output)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite prediction logits")
    return logits


class Actor:
    """Independent model/optimizer/RNG state; offloaded between crop/disease epochs."""
    def __init__(self, target, config, folder, identity, model_factory=IndependentNet):
        self.target, self.config, self.folder, self.identity = target, config, Path(folder), identity
        self.folder.mkdir(parents=True, exist_ok=True)
        saved_path = self.folder / "last.pt"
        seed_everything(config["seed"])
        self.model = model_factory(config, target, pretrained=not saved_path.exists())
        self.opt = optimizer(self.model, config)
        self.scaler = torch.amp.GradScaler(config["runtime"]["device"], enabled=config["runtime"]["amp"])
        self.generator = torch.Generator().manual_seed(config["seed"])
        self.epoch, self.history, self.val_logits = 0, [], None
        self.rng = rng_state()
        if saved_path.exists():
            saved = torch.load(saved_path, map_location="cpu", weights_only=True)
            if saved["fingerprint"] != identity or saved["target"] != target:
                raise ValueError("Recovery checkpoint configuration/target mismatch")
            self.model.load_state_dict(saved["model_state"])
            self.opt.load_state_dict(saved["optimizer_state"])
            self.scaler.load_state_dict(saved["scaler_state"])
            self.generator.set_state(saved["generator_state"])
            self.epoch, self.history = saved["epoch"], saved["history"]
            self.val_logits, self.rng = saved["val_logits"], saved["rng_state"]
            expected = {"epoch": self.epoch, "horizon": config["training"]["max_epochs"],
                        "factor": lr_factor(self.epoch, config["training"]) if self.epoch else 0.0}
            if saved["scheduler_state"] != expected:
                raise ValueError("Recovery LR schedule mismatch")

    def checkpoint(self):
        atomic_checkpoint(self.folder / "last.pt", {
            "fingerprint": self.identity, "target": self.target, "epoch": self.epoch,
            "model_state": self.model.state_dict(), "optimizer_state": self.opt.state_dict(),
            "scaler_state": self.scaler.state_dict(), "generator_state": self.generator.get_state(),
            "rng_state": self.rng, "history": self.history, "val_logits": self.val_logits,
            "scheduler_state": {"epoch": self.epoch, "horizon": self.config["training"]["max_epochs"],
                                "factor": lr_factor(self.epoch, self.config["training"]) if self.epoch else 0.0}})

    def epoch_step(self, frame, val_frame, cache_dir):
        cfg = self.config
        device = cfg["runtime"]["device"]
        self.model.to(device)
        move_optimizer(self.opt, device)
        restore_rng(self.rng)
        self.model.train()
        if cfg["model"]["freeze_bn_stats"]:
            freeze_batchnorm_statistics(self.model)
        epoch = self.epoch + 1
        factor = lr_factor(epoch, cfg["training"])
        for group in self.opt.param_groups:
            group["lr"] = group["initial_lr"] * factor
        train, _ = transforms(cfg)
        data = loader(frame, cache_dir, train, cfg, training=True, generator=self.generator)
        accumulation = cfg["training"]["accumulation_steps"]
        loss_sum, hits, count, steps, skipped = 0.0, 0, 0, 0, 0
        started = time.monotonic()
        self.opt.zero_grad(set_to_none=True)
        for batch, (x, crop, disease) in enumerate(data):
            target = (crop if self.target == "crop" else disease).to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device, enabled=cfg["runtime"]["amp"]):
                logits = self.model(x)
                loss = F.cross_entropy(logits.float(), target, label_smoothing=cfg["training"]["label_smoothing"])
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Non-finite {self.target} training loss at epoch {epoch}")
            denominator = accumulation_samples(batch, len(data), cfg["training"]["batch_size"], len(frame), accumulation)
            self.scaler.scale(loss * (len(x) / denominator)).backward()
            if (batch + 1) % accumulation == 0 or batch + 1 == len(data):
                self.scaler.unscale_(self.opt)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["training"]["gradient_clip"])
                if not bool(torch.isfinite(norm)) and not self.scaler.is_enabled():
                    raise ValueError("Non-finite gradient norm; checkpoint retained at previous complete epoch")
                before_scale = self.scaler.get_scale()
                self.scaler.step(self.opt)
                self.scaler.update()
                skipped += int(self.scaler.get_scale() < before_scale)
                steps += 1
                self.opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(x)
            hits += int((logits.argmax(1) == target).sum())
            count += len(x)
            if (batch + 1) % 200 == 0:
                print(f"  {self.target} epoch {epoch}: batch {batch + 1}/{len(data)}", flush=True)
        if skipped == steps:
            raise ValueError("Every optimizer update overflowed; retained previous complete checkpoint")
        self.val_logits = predict(self.model, val_frame, cache_dir, cfg) if val_frame is not None else None
        record = {"epoch": epoch, "train_loss": loss_sum / count, "train_top1": hits / count,
                  "seconds": time.monotonic() - started, "lr_backbone": cfg["optimizer"]["lr_backbone"] * factor,
                  "lr_head": cfg["optimizer"]["lr_head"] * factor,
                  "optimizer_steps": steps, "amp_skipped_steps": skipped}
        if val_frame is not None:
            targets = torch.tensor(val_frame[f"{self.target}_id"].to_numpy())
            record.update(validation_loss=float(F.cross_entropy(self.val_logits, targets)),
                          validation_top1=float((self.val_logits.argmax(1) == targets).float().mean()))
        self.rng = rng_state()
        self.epoch = epoch
        self.history.append(record)
        self.model.cpu()
        move_optimizer(self.opt, "cpu")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        self.checkpoint()
        atomic_json(self.folder / "history.json", self.history)
        print(f"{self.target} epoch {epoch}: train={record['train_top1']:.2%} "
              f"time={record['seconds']:.0f}s", flush=True)
        return record

    def save_weights(self, path, epoch=None):
        if any(not bool(torch.isfinite(p).all()) for p in self.model.state_dict().values() if p.is_floating_point()):
            raise ValueError("Non-finite model weights")
        atomic_checkpoint(path, {"model_state": self.model.state_dict(), "target": self.target,
                                 "epoch": epoch or self.epoch, "fingerprint": self.identity})


def select_gamma(scores, labels, unseen_mask, allowed_mask, seen_count, gammas):
    """Use the shared evaluator's Top-1 ranking, including its exact-tie behavior."""
    from dpd.evaluate import calibrate, topk_hits

    seen_ids = np.flatnonzero(allowed_mask & ~unseen_mask)
    unseen_ids = np.flatnonzero(allowed_mask & unseen_mask)
    if not len(seen_ids) or not len(unseen_ids) or not 0 < seen_count < len(labels):
        raise ValueError("Validation requires both seen and pseudo-unseen samples/candidates")
    masked = scores.copy()
    masked[:, ~allowed_mask] = -np.inf
    s, u = [], []
    for gamma in gammas:
        # Use the same deterministic pair-id tie rule and split boundaries as evaluate().
        s.append(topk_hits(calibrate(masked[:seen_count], unseen_mask, float(gamma)), labels[:seen_count], 1).mean())
        u.append(topk_hits(calibrate(masked[seen_count:], unseen_mask, float(gamma)), labels[seen_count:], 1).mean())
    s, u = np.asarray(s), np.asarray(u)
    hm = np.divide(2 * s * u, s + u, out=np.zeros_like(s), where=s + u != 0)
    index = int(hm.argmax())
    return {"gamma": float(gammas[index]), "harmonic_mean_top1": float(hm[index]),
            "seen_top1": float(s[index]), "unseen_top1": float(u[index]),
            "gamma0_harmonic_mean_top1": float(hm[0])}


def choose(candidates):
    if not candidates:
        raise ValueError("No completed validation candidates")
    return max(candidates, key=lambda row: row["validation"]["harmonic_mean_top1"])


def plot_history(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    history = read_json(folder / "history.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for target in ("crop", "disease"):
        path = folder / target / "history.json"
        if not path.exists():
            continue
        factor = read_json(path)
        axes[0].plot([r["epoch"] for r in factor], [r["train_top1"] for r in factor], label=target + " fit")
        if "validation_top1" in factor[0]:
            axes[0].plot([r["epoch"] for r in factor], [r["validation_top1"] for r in factor], label=target + " val")
            axes[1].plot([r["epoch"] for r in factor], [r["validation_loss"] for r in factor], label=target + " val CE")
    if history:
        axes[2].plot([r["epoch"] for r in history], [r["harmonic_mean_top1"] for r in history], label="calibrated val H")
        axes[2].plot([r["epoch"] for r in history], [r["gamma0_harmonic_mean_top1"] for r in history], label="gamma-zero val H")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Factor Top-1")
    axes[1].set_ylabel("Ordinary validation CE")
    axes[2].set_ylabel("Composition harmonic mean")
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=130)
    plt.close(fig)


def prune_recovery(folder, root):
    """Only retire this completed job's full optimizer states, never old experiments."""
    folder, root = Path(folder).resolve(), Path(root).resolve()
    if not folder.is_relative_to(root) or not (folder / "complete.json").is_file():
        raise ValueError("Refusing to prune recovery outside a verified completed study job")
    for target in ("crop", "disease"):
        path = folder / target / "last.pt"
        if path.exists():
            path.unlink()
