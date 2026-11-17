"""Fresh full-data refit of the original Tiny recipe, without new selection trials.

The pinned sibling Plant_Disease/baseline engine supplies the original transforms,
optimizer-step/accumulation implementation and LR function. Existing runs are read
only. This controller saves new weights and uses the shared composition evaluator.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LEGACY = PROJECT.parent / "Plant_Disease"
os.environ["HF_HOME"] = str(LEGACY / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "runs/cache/independent_cnn_baseline/matplotlib"))
os.environ.setdefault("CUDA_CACHE_PATH", str(PROJECT / "runs/cache/independent_cnn_baseline/cuda"))
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dpd.data import pair_table
from dpd.evaluate import TIE_BREAK_POLICY, evaluate, ranked_indices, topk_hits
from dpd.report import save_test_outputs
from .engine import StudyLock, Tee, atomic_checkpoint, atomic_json, fingerprint, keep_awake, now, read_json, sha256

DEFAULT_SPEC = PROJECT / "independent_cnn_baseline/configs/tiny_full_refit_seed42.json"
TARGETS = {"crop": 55, "disease": 175}


class StopRefit(Exception):
    pass


def legacy_engine():
    from baseline import engine
    return engine


def validate_spec(spec):
    if (PROJECT / spec["legacy_root"]).resolve() != LEGACY.resolve():
        raise ValueError("Expected the pinned sibling Tiny project")
    if spec["target_epochs"] != {"crop": 17, "disease": 16} or spec["target_seeds"] != {"crop": 42, "disease": 43}:
        raise ValueError("Only the original validation-selected Tiny refits are authorized")
    if spec["gamma"] != 0 or spec["lr_schedule_horizon"] != 30:
        raise ValueError("Tiny gamma zero and original LR horizon must stay frozen")
    root = (PROJECT / spec["output_dir"]).resolve()
    if not root.is_relative_to(PROJECT / "runs/baseline"):
        raise ValueError("New Tiny outputs must stay under this repository's runs/baseline")
    source = (LEGACY / spec["source_run"]).resolve()
    if not source.is_relative_to(LEGACY / "runs"):
        raise ValueError("Unexpected Tiny source directory")
    for name, digest in spec["legacy_code_sha256"].items():
        if sha256(LEGACY / name) != digest:
            raise ValueError(f"Original Tiny implementation changed: {name}")
    for name, digest in spec["source_artifact_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Original Tiny artifact changed: {name}")
    weights = LEGACY / spec["pretrained_file"]
    if sha256(weights) != spec["pretrained_sha256"]:
        raise ValueError("Original pretrained Tiny weights changed")
    config = read_json(source / "config.resolved.json")
    if config["model"] != "convnext_tiny.fb_in22k" or config["device"] != "cuda" or not config["pretrained"]:
        raise ValueError("Expected the original CUDA/IN22K Tiny recipe")
    if config["epochs"] != 30:
        raise ValueError("Original Tiny LR schedule must retain its 30-epoch horizon")
    for target, epochs in spec["target_epochs"].items():
        if read_json(source / target / "summary.json")["epoch"] != epochs:
            raise ValueError("Source validation-selected epoch changed")
    return root, source, config


def audit_data(source, root, config, update):
    """Match original metadata and every image hash; restore all 8000 train rows."""
    provenance = read_json(source / "provenance.json")
    expected_images = {Path(path).name: digest for path, digest in provenance["images"].items()}
    columns = ["file", "crop_id", "disease_id", "pair_id", "crop", "disease", "pair"]
    frames = {}
    image_dir = LEGACY / "DPD_Small_dataset"
    for split, suffix in (("train", "train"), ("test_seen", "seen"), ("test_unseen", "unseen")):
        file = LEGACY / ("DPD_Small_train.csv" if split == "train" else f"DPD_Small_{split}.csv")
        if sha256(file) != provenance["metadata"][suffix]["sha256"]:
            raise ValueError(f"Original metadata changed: {split}")
        frames[split] = pd.read_csv(file, header=None, names=columns)
        frames[split]["split"] = split
    if [len(frames[k]) for k in ("train", "test_seen", "test_unseen")] != [8000, 2000, 1448]:
        raise ValueError("Unexpected official image counts")
    df = pd.concat(frames.values(), ignore_index=True)
    if not df.file.is_unique:
        raise ValueError("Image filenames overlap official splits")
    if len(expected_images) != len(df):
        raise ValueError("Original image identity is incomplete")
    for index, file in enumerate(df.file, 1):
        if sha256(image_dir / file) != expected_images[file]:
            raise ValueError(f"Image differs from original Tiny run: {file}")
        if index % 2000 == 0:
            update(phase="audit", audited_images=index)
            print(f"Audited {index}/{len(df)} original image hashes", flush=True)
    update(phase="audit", audited_images=len(df))
    fit = pd.read_csv(source / "fit.csv")
    val = pd.read_csv(source / "validation.csv")
    if len(fit) != 6746 or len(val) != 1254 or set(fit.filename) & set(val.filename):
        raise ValueError("Original Tiny validation partition changed")
    if set(fit.filename) | set(val.filename) != set(frames["train"].file):
        raise ValueError("Full-data refit must restore the original validation images")
    labels = read_json(source / "labels.json")
    for target, count in TARGETS.items():
        if labels[target] != list(range(count)) or sorted(frames["train"][target + "_id"].unique()) != labels[target]:
            raise ValueError("Unexpected Tiny label mapping")
    df["is_unseen"] = df.pair_id.isin(set(frames["test_unseen"].pair_id))
    pairs = pair_table(df)
    if len(pairs) != 333 or pairs.is_unseen.sum() != 10:
        raise ValueError("Expected all 333 compositions and 10 unseen pairs")
    frames["train"].to_csv(root / "all_train.csv", index=False)
    audit = {"verified_at_utc": now(), "counts": {k: len(v) for k, v in frames.items()},
             "original_fit_images": len(fit), "restored_validation_images": len(val),
             "all_original_image_hashes_match": True, "image_hash_fingerprint": fingerprint(expected_images)}
    atomic_json(root / "audit.json", audit)
    return frames, pairs, image_dir, audit


class OriginalImages(Dataset):
    def __init__(self, frame, image_dir, transform, target=None):
        self.files = [Path(image_dir) / file for file in frame.file]
        self.labels = frame[target + "_id"].to_numpy() if target else np.arange(len(frame))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(self.labels[index])


def create_model(config, target, pretrained):
    engine = legacy_engine()
    options = engine.model_regularization(config)
    if pretrained:
        options["pretrained_cfg_overlay"] = {"file": config["pinned_pretrained_file"], "custom_load": False}
    return engine.timm.create_model(config["model"], pretrained=pretrained,
                                    num_classes=TARGETS[target], **options)


def data_loader(frame, image_dir, transform, config, target=None, training=False, generator=None, batch_size=None):
    return DataLoader(OriginalImages(frame, image_dir, transform, target),
                      batch_size=batch_size or config["batch_size"], shuffle=training,
                      num_workers=config["workers"], pin_memory=config["device"].startswith("cuda"),
                      persistent_workers=False, worker_init_fn=legacy_engine().worker_seed, generator=generator)


def plot_training(folder, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("train_loss", "train_top1")):
        ax.plot([r["epoch"] for r in history], [r[key] for r in history])
        ax.set(xlabel="Full-data refit epoch", ylabel=key)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=160)
    plt.close(fig)


def refit_target(target, frame, image_dir, config, epochs, seed, root, identity, update, boundary):
    folder = root / target
    folder.mkdir(exist_ok=True)
    weights = root / f"{target}_net.pt"
    done = folder / "complete.json"
    if done.exists():
        row = read_json(done)
        if row["fingerprint"] != identity or row["epochs"] != epochs or sha256(weights) != row["sha256"]:
            raise ValueError("Completed Tiny refit identity/hash mismatch")
        return row
    boundary()
    engine = legacy_engine()
    engine.seed_everything(seed)
    recovery = folder / "last.pt"
    saved = torch.load(recovery, map_location="cpu", weights_only=True) if recovery.exists() else None
    if saved and (saved["fingerprint"] != identity or saved["target"] != target):
        raise ValueError("Partial Tiny refit identity mismatch")
    model = create_model(config, target, pretrained=saved is None)
    train_tf, _, data_config = engine.transforms_for(model, config["image_size"])
    atomic_json(folder / "preprocessing.json", {"data_config": data_config, "train_transform": str(train_tf)})
    model.to(config["device"])
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": config["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    # Keep the original 30-epoch horizon, independently of the selected refit length.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: engine.learning_rate_factor(e, config["epochs"], config["warmup_epochs"]))
    device = torch.device(config["device"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    history = []
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history = saved["history"]
        engine.random.setstate(saved["python_rng"])
        state = saved["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    dl = data_loader(frame, image_dir, train_tf, config, target, True, generator)
    for epoch in range(len(history), epochs):
        boundary()
        update(phase="refit", active={"target": target, "epoch": epoch + 1, "epochs": epochs})
        started = time.perf_counter()
        lrs = [group["lr"] for group in optimizer.param_groups]
        stats = engine.run_epoch(model, dl, device, amp, optimizer, scaler,
                                 config["accumulation_steps"], config["label_smoothing"])
        history.append({"epoch": epoch + 1, "train_loss": stats["loss"], "train_top1": stats["top1"],
                        "backbone_lr": lrs[0], "head_lr": lrs[1], "seconds": time.perf_counter() - started})
        scheduler.step()
        state = np.random.get_state()
        atomic_checkpoint(recovery, {"fingerprint": identity, "target": target, "epoch": epoch + 1,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "history": history,
            "python_rng": engine.random.getstate(), "numpy_rng": (state[0], state[1].tolist(), *state[2:]),
            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        atomic_json(folder / "history.json", history)
        print(f"{target} refit epoch {epoch + 1}/{epochs}: train={stats['top1']:.2%}, {history[-1]['seconds']:.0f}s", flush=True)
        update()
    atomic_checkpoint(weights, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    plot_training(folder, history)
    row = {"target": target, "fingerprint": identity, "epochs": epochs, "seed": seed,
           "training_images": len(frame), "lr_schedule_horizon": config["epochs"],
           "training_epoch_seconds": sum(r["seconds"] for r in history), "sha256": sha256(weights), "completed_at_utc": now()}
    atomic_json(done, row)
    del model, optimizer, scheduler, scaler, saved, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


@torch.inference_mode()
def predict_target(target, frames, image_dir, config, root, boundary):
    weights = root / f"{target}_net.pt"
    model = create_model(config, target, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(config["device"]).eval()
    _, transform, _ = legacy_engine().transforms_for(model, config["image_size"])
    cache = root / "evaluation_recovery"
    cache.mkdir(exist_ok=True)
    outputs = {}
    for split in ("test_seen", "test_unseen"):
        path = cache / f"{target}_{split}.pt"
        expected = {"weights_sha256": sha256(weights), "files": frames[split].file.tolist()}
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["identity"] != expected:
                raise ValueError("Cached Tiny logits do not match the frozen checkpoint/data")
            outputs[split] = saved["logits"].numpy()
            continue
        boundary()
        logits = []
        for images, _ in data_loader(frames[split], image_dir, transform, config, target, batch_size=1):
            with torch.autocast(config["device"], enabled=config["amp"], dtype=torch.float16):
                pred = model(images.to(config["device"], non_blocking=True))
            logits.append(pred.float().cpu())
        values = torch.cat(logits)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite Tiny test logits")
        atomic_checkpoint(path, {"identity": expected, "logits": values})
        outputs[split] = values.numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


class TinyRefitRunModel:
    logit_scale = 1.0
    score_help = "log P(crop) + log P(disease), gamma zero"

    def __init__(self, run, pairs):
        self.run, self.pairs = run, pairs
        self.config = run["cfg"]["legacy_recipe"]
        self.gamma = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nets = {}
        for target in TARGETS:
            model = create_model(self.config, target, pretrained=False)
            model.load_state_dict(torch.load(Path(run["dir"]) / f"{target}_net.pt", map_location="cpu", weights_only=True))
            self.nets[target] = model.to(self.device).eval()
        _, self.transform, _ = legacy_engine().transforms_for(model, self.config["image_size"])

    @torch.inference_mode()
    def _forward(self, images):
        logp, features = {}, []
        with torch.autocast(self.device, enabled=self.device == "cuda", dtype=torch.float16):
            for target, model in self.nets.items():
                feature = model.forward_features(images.to(self.device))
                logits = model.forward_head(feature)
                features.append(nn.functional.normalize(model.forward_head(feature, pre_logits=True).float(), dim=-1))
                logp[target] = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        embedding = nn.functional.normalize(torch.cat(features, dim=-1), dim=-1).cpu().numpy()
        from dpd.cnn import compose_pair_scores
        return embedding, compose_pair_scores(logp["crop"], logp["disease"], self.pairs)

    def predict(self, src):
        with Image.open(src) as image:
            images = self.transform(image.convert("RGB")).unsqueeze(0)
        embedding, scores = self._forward(images)
        return embedding[0], scores[0]

    def train_bank(self, train):
        folder = Path(self.run["dir"])
        identity = fingerprint({"files": train.file.tolist(), "weights": {t: sha256(folder / f"{t}_net.pt") for t in TARGETS}})
        path = folder / "train_features.npz"
        if path.exists():
            saved = np.load(path)
            if saved["identity"].item() == identity:
                return saved["features"]
        loader = data_loader(train, LEGACY / "DPD_Small_dataset", self.transform,
                             self.config, batch_size=self.config["batch_size"])
        features = np.concatenate([self._forward(images)[0] for images, _ in loader])
        np.savez_compressed(path, identity=identity, features=features)
        return features


def evaluate_final(frames, pairs, image_dir, config, app_config, root, boundary):
    marker = root / "completion_verified.json"
    if marker.exists():
        for name, digest in read_json(marker)["artifact_sha256"].items():
            if sha256(root / name) != digest:
                raise ValueError("Completed Tiny evaluation artifact changed")
        return read_json(root / "metrics.json")
    outputs = {target: predict_target(target, frames, image_dir, config, root, boundary) for target in TARGETS}
    from dpd.cnn import compose_pair_scores
    scores, labels, files, factors = {}, {}, {}, {}
    for split in ("test_seen", "test_unseen"):
        logp = {t: torch.log_softmax(torch.from_numpy(outputs[t][split]), dim=-1).numpy() for t in TARGETS}
        scores[split] = compose_pair_scores(logp["crop"], logp["disease"], pairs)
        labels[split], files[split] = frames[split].pair_id.to_numpy(), frames[split].file.to_numpy()
        factors[split] = {t: {"top1": float(topk_hits(outputs[t][split], frames[split][t + "_id"].to_numpy(), 1).mean()),
                              "top5": float(topk_hits(outputs[t][split], frames[split][t + "_id"].to_numpy(), 5).mean())} for t in TARGETS}
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma=0.0)
    result.update(version=app_config["version"], selected_epochs={"crop": 17, "disease": 16},
                  final_training_images=8000, raw_factor_metrics=factors, tie_break_policy=TIE_BREAK_POLICY)
    save_test_outputs(root, result, scores, labels, files, pairs, 0.0)
    np.savez_compressed(root / "factor_logits.npz", **{f"{s}_{t}": outputs[t][s] for t in TARGETS for s in scores})
    app = TinyRefitRunModel({"dir": root, "cfg": app_config, "metrics": result}, pairs)
    checks = []
    for split in scores:
        for index in np.linspace(0, len(frames[split]) - 1, 4, dtype=int):
            _, actual = app.predict(image_dir / frames[split].iloc[index].file)
            error = float(np.max(np.abs(actual - scores[split][index])))
            if error > 1e-4:
                raise ValueError(f"Tiny app score parity failed: {error}")
            checks.append({"split": split, "index": int(index), "max_absolute_difference": error})
    del app
    gc.collect()
    torch.cuda.empty_cache()
    matrix = pd.read_csv(root / "confusion_test_seen.csv", index_col=0)
    if matrix.shape != (323, 333) or matrix.to_numpy().sum() != 2000:
        raise ValueError("Tiny seen confusion matrix count/shape mismatch")
    for split, count in (("test_seen", 2000), ("test_unseen", 1448)):
        exported = pd.read_csv(root / f"predictions_{split}.csv")
        if len(exported) != count or not np.isfinite(scores[split]).all():
            raise ValueError("Tiny prediction count/finite score mismatch")
        if not np.array_equal(exported.correct_top1, ranked_indices(scores[split], 1)[:, 0] == labels[split]):
            raise ValueError("Tiny prediction table differs from evaluator")
    atomic_json(root / "app_inference_verification.json", checks)
    artifacts = ["metrics.json", "scores.npz", "factor_logits.npz", "config.json", "selected_source.json",
                 "crop_net.pt", "disease_net.pt", "predictions_test_seen.csv", "predictions_test_unseen.csv",
                 "confusion_test_seen.csv", "app_inference_verification.json", "crop/complete.json", "disease/complete.json"]
    atomic_json(marker, {"verified_at_utc": now(), "tie_break_policy": TIE_BREAK_POLICY,
                         "prediction_counts": {s: len(f) for s, f in files.items()}, "candidate_count": 333,
                         "seen_confusion_shape": [323, 333], "seen_confusion_count": 2000, "app_inference_checks": checks,
                         "artifact_sha256": {name: sha256(root / name) for name in artifacts}})
    return result


def smoke(config, root):
    legacy_engine().seed_everything(123)
    model = create_model(config, "disease", pretrained=True).to("cuda").train()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = nn.functional.cross_entropy(model(torch.randn(8, 3, 224, 224, device="cuda")), torch.zeros(8, dtype=torch.long, device="cuda"))
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("CUDA Tiny synthetic check failed")
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        image = torch.randn(1, 3, 224, 224, device="cuda")
        if not torch.equal(model(image), model.forward_head(model.forward_features(image))):
            raise ValueError("Native Tiny forward-head path differs from model forward")
    atomic_json(root / "cuda_preflight.json", {"passed": True, "device": torch.cuda.get_device_name(0),
                "batch_size": 8, "dataset_images_used": 0, "native_forward_head_parity": True,
                "loss": float(loss.detach()), "pretrained_sha256": sha256(config["pinned_pretrained_file"])})
    del model
    gc.collect()
    torch.cuda.empty_cache()


def study(spec_path=DEFAULT_SPEC, resume=False, prepare_only=False, hours=None, deadline_utc=None, clear_stop=False):
    if hours is not None and (not np.isfinite(hours) or hours <= 0):
        raise ValueError("Window hours must be finite and positive")
    spec = read_json(spec_path)
    root, source, config = validate_spec(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the Tiny full-data refit")
    torch.set_num_threads(4)
    root.mkdir(parents=True, exist_ok=True)
    with StudyLock(root / "runner.lock"), keep_awake():
        previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if previous and previous["state"] not in ("prepared",) and not resume:
            raise FileExistsError("Tiny run already started; verify controller exit, then use --resume")
        if clear_stop and resume:
            (root / "STOP.request").unlink(missing_ok=True)
        state = {**previous, "state": "running", "phase": "audit", "pid": os.getpid(),
                 "target_epochs": spec["target_epochs"], "gamma": 0.0}
        (root / "runner.pid").write_text(str(os.getpid()) + "\n")

        def update(**values):
            state.update(values, updated_at_utc=now())
            atomic_json(root / "status.json", state)

        def boundary():
            if (root / "STOP.request").exists():
                raise StopRefit("paused")
            if state.get("window_deadline_at_utc") and datetime.now(timezone.utc) >= datetime.fromisoformat(state["window_deadline_at_utc"]):
                raise StopRefit("window_complete")

        with (root / "training.log").open("a", encoding="utf-8", buffering=1) as log, \
             redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                update()
                if (root / "STOP.request").exists():
                    raise StopRefit("paused")
                config = {**config, "data_root": str(LEGACY), "output_dir": str(root),
                          "pinned_pretrained_file": str(LEGACY / spec["pretrained_file"])}
                frames, pairs, image_dir, audit = audit_data(source, root, config, update)
                identity = fingerprint({"spec": spec, "config": config, "data": audit["image_hash_fingerprint"], "train": frames["train"].file.tolist()})
                marker = root / "fingerprint.txt"
                if marker.exists() and marker.read_text().strip() != identity:
                    raise ValueError("Tiny full-refit configuration/data identity changed")
                marker.write_text(identity + "\n")
                atomic_json(root / "spec.json", spec)
                app_config = {"version": spec["name"], "method": "cnn_independent", "implementation": "independent_cnn_baseline_tiny_refit",
                              "output_dir": spec["output_dir"], "model": {"backbone": config["model"]}, "legacy_recipe": config}
                atomic_json(root / "config.json", app_config)
                atomic_json(root / "selected_source.json", {"source": str(source), "target_epochs": spec["target_epochs"],
                            "target_seeds": spec["target_seeds"], "gamma": 0.0, "source_artifact_sha256": spec["source_artifact_sha256"]})
                for target in TARGETS:
                    shutil.copy2(source / target / "learning_curves.png", root / f"{target}_selection_learning_curves.png")
                if not (root / "cuda_preflight.json").exists():
                    smoke(config, root)
                if prepare_only:
                    update(state="prepared", phase="prepared", active=None)
                    return
                if deadline_utc:
                    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
                elif resume and hours is None and previous.get("window_deadline_at_utc"):
                    deadline = datetime.fromisoformat(previous["window_deadline_at_utc"])
                else:
                    deadline = datetime.now(timezone.utc) + timedelta(hours=hours or spec["window_hours"])
                    state["window_started_at_utc"] = now()
                update(window_deadline_at_utc=deadline.isoformat(), window_hours=hours or spec["window_hours"])
                rows = {}
                for target in TARGETS:
                    rows[target] = refit_target(target, frames["train"], image_dir, config, spec["target_epochs"][target],
                                                spec["target_seeds"][target], root, identity, update, boundary)
                    update(completed_targets=list(rows))
                update(phase="evaluation", active=None, official_test_evaluation_started_at_utc=now())
                result = evaluate_final(frames, pairs, image_dir, config, app_config, root, boundary)
                # Check original artifacts again after fitting/evaluation; this never alters them.
                validate_spec(spec)
                summary = {"completed_at_utc": now(), "target_refits": rows, "metrics": result,
                           "training_epoch_seconds": sum(r["training_epoch_seconds"] for r in rows.values()),
                           "source_artifacts_unchanged": True, "epochs_and_gamma_frozen_before_test": True}
                atomic_json(root / "study_summary.json", summary)
                original = read_json(source / "metrics.json")
                base = read_json(PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_seed42/uncalibrated_metrics.json")
                lines = ["# ConvNeXt-Tiny original recipe: full-data refit", "", "All 8000 training images; crop 17 epochs / seed 42, disease 16 / seed 43; gamma zero.",
                         "Original Tiny initialization, transforms and optimizer; original 30-epoch LR horizon. No new model/epoch/gamma selection.", "",
                         "| Run | Seen Top-1 | Unseen Top-1 | H Top-1 | Seen Top-5 | Unseen Top-5 | H Top-5 |", "|---|---:|---:|---:|---:|---:|---:|"]
                for name, values in (("Original Tiny", [original["seen"]["composition"]["top1"], original["unseen"]["composition"]["top1"], original["harmonic_mean"]["top1"], original["seen"]["composition"]["top5"], original["unseen"]["composition"]["top5"], original["harmonic_mean"]["top5"]]),
                                     ("Tiny full-data refit", [result["test_seen"]["top1"], result["test_unseen"]["top1"], result["harmonic_mean_top1"], result["test_seen"]["top5"], result["test_unseen"]["top5"], result["harmonic_mean_top5"]]),
                                     ("Base full-data refit", [base["test_seen"]["top1"], base["test_unseen"]["top1"], base["harmonic_mean_top1"], base["test_seen"]["top5"], base["test_unseen"]["top5"], base["harmonic_mean_top5"]])):
                    lines.append("| " + name + " | " + " | ".join(f"{v:.2%}" for v in values) + " |")
                lines.extend(["", "Official tests: 2000 seen / 1448 unseen images, all 333 candidate compositions.",
                              "Raw independent factor metrics and separately labeled unseen-only diagnostics are in metrics.json.",
                              "The original Tiny/full-data comparison changes fitting images only; the Base comparison still differs in pretrained initialization, augmentation and selection protocol.",
                              f"Recorded full-data training epoch time: {summary['training_epoch_seconds']:.3f} seconds.",
                              "[Metrics](metrics.json), [crop weights](crop_net.pt), [disease weights](disease_net.pt), [verification](completion_verified.json)."])
                (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
                update(state="completed", phase="completed", active=None, completed_at_utc=now())
            except StopRefit as stop:
                update(state=str(stop), phase="checkpointed_stop", active=state.get("active"))
            except Exception:
                update(state="failed", error=traceback.format_exc())
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    args = parser.parse_args()
    if args.request_stop:
        root = PROJECT / read_json(args.spec)["output_dir"]
        root.mkdir(parents=True, exist_ok=True)
        (root / "STOP.request").write_text(now())
        return
    study(args.spec, args.resume, args.prepare, args.hours, args.deadline_utc, args.clear_stop)


if __name__ == "__main__":
    main()
