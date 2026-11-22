"""Run the isolated, single-seed three-arm study through final verification."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from objectives import FactorCNN, cross_crop_supcon
from protocol import (ARMS, HERE, EpochBatches, StudyLock, harmonic, joint_scores,
                      load_config, loader, metrics, paired_bootstrap, prepare,
                      restore_rng, rng_state, save_checkpoint, seed_all, sha256,
                      stamp, status, tensor_hash, transforms_for, write_json)


def amp(cfg):
    return torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"])


def to_device(model, cfg):
    return model.to(cfg["device"], memory_format=torch.channels_last if cfg["channels_last"] else torch.contiguous_format)


def image_device(images, cfg):
    return images.to(cfg["device"], non_blocking=True,
                     memory_format=torch.channels_last if cfg["channels_last"] else torch.contiguous_format)


def optimizer_for(model, cfg):
    head_ids = {id(p) for p in model.net.get_classifier().parameters()} | {id(p) for p in model.projector.parameters()}
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg["backbone_lr"], "base_lr": cfg["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": cfg["head_lr"], "base_lr": cfg["head_lr"]}
    ], weight_decay=cfg["weight_decay"])


def lr_factor(epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        return (epoch+1)/cfg["warmup_epochs"]
    progress = (epoch-cfg["warmup_epochs"])/max(1, cfg["epochs"]-cfg["warmup_epochs"]-1)
    return 0.01 + 0.99*0.5*(1+math.cos(math.pi*progress))


@torch.inference_mode()
def predict(model, rows, cfg, ontology, transform):
    model.eval()
    values = []
    for images, _ in loader(rows, cfg, ontology, transform):
        with amp(cfg):
            logits, _ = model(image_device(images, cfg))
        values.append(logits.float().cpu())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite evaluation logits")
    return result


def validation(model, target, prepared, cfg, crop_cache, transform):
    ontology = prepared["ontology"]
    output = {}
    ces = []
    keyname = "crop" if target == "crop" else "disease"
    mapping = {v: i for i, v in enumerate(ontology[keyname])}
    validation_roles = ("seen_validation", "development")
    combined_rows = sum([prepared["roles"][role] for role in validation_roles], [])
    combined_logits = predict(model, combined_rows, cfg, ontology, transform)
    offset = 0
    for role in validation_roles:
        rows = prepared["roles"][role]
        logits = combined_logits[offset:offset+len(rows)]
        offset += len(rows)
        targets = torch.tensor([mapping[r[keyname+"_id"]] for r in rows])
        ces.append(float(F.cross_entropy(logits, targets)))
        if target == "crop":
            output[role] = {"top1": float(logits.argmax(1).eq(targets).double().mean()), "ce": ces[-1]}
        else:
            output[role] = metrics(crop_cache[role], logits, rows, ontology)
    output["mean_ce"] = sum(ces)/2
    if target == "crop":
        output["selection_score"] = sum(output[r]["top1"] for r in ("seen_validation", "development"))/2
    else:
        output["selection_score"] = harmonic(output["seen_validation"]["composition"]["top1"],
                                              output["development"]["composition"]["top1"])
    return output


def training_epoch(model, target, arm, prepared, cfg, optimizer, scaler, transform, run, epoch):
    model.train()
    rows = prepared["roles"]["fit"]
    sampler = EpochBatches(rows, cfg, grouped=arm in ("ce_balanced", "ce_supcon"))
    sampler.epoch = epoch
    indices = sampler.indices()
    data = loader(rows, cfg, prepared["ontology"], transform, sampler=sampler, epoch=epoch)
    optimizer.zero_grad(set_to_none=True)
    totals = {"samples": 0, "ce_sum": 0.0, "contrastive_sum": 0.0, "correct": 0,
              "eligible_anchors": 0, "positive_pairs": 0, "amp_skipped_steps": 0}
    window = 0
    target_column = 0 if target == "crop" else 1
    healthy = prepared["ontology"]["disease"].index(prepared["ontology"]["healthy_disease_id"])
    for batch, (images, labels) in enumerate(data, 1):
        images, labels = image_device(images, cfg), labels.to(cfg["device"], non_blocking=True)
        with amp(cfg):
            logits, projection = model(images)
            ce = F.cross_entropy(logits, labels[:, target_column], label_smoothing=cfg["label_smoothing"], reduction="sum")
        contrastive, detail = cross_crop_supcon(projection, labels[:, 1], labels[:, 0],
                                               cfg["contrastive_temperature"],
                                               healthy if cfg["exclude_healthy_from_contrastive"] else None)
        weight = cfg["contrastive_weight"] if arm == "ce_supcon" else 0.0
        loss = ce + weight*len(images)*contrastive
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {arm} epoch {epoch+1}")
        scaler.scale(loss).backward()
        window += len(images)
        if batch % cfg["accumulation_steps"] == 0 or batch == len(data):
            previous = scaler.get_scale()
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(window)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not cfg["amp"] and not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradients after AMP unscale")
            scaler.step(optimizer)
            scaler.update()
            totals["amp_skipped_steps"] += int(scaler.get_scale() < previous)
            optimizer.zero_grad(set_to_none=True)
            window = 0
        totals["samples"] += len(images)
        totals["ce_sum"] += float(ce.detach())
        totals["contrastive_sum"] += float(contrastive.detach())*len(images)
        totals["correct"] += int(logits.detach().argmax(1).eq(labels[:, target_column]).sum())
        for key in ("eligible_anchors", "positive_pairs"):
            totals[key] += detail[key]
        if batch == 1 or batch % 100 == 0 or batch == len(data):
            status(run, "training", arm=arm, epoch=epoch+1, batch=batch, total_batches=len(data),
                   train_ce=totals["ce_sum"]/totals["samples"])
    counts = np.bincount(indices, minlength=len(rows))
    exposure_path = run / arm / f"exposures_epoch{epoch+1:02}.npz"
    np.savez_compressed(exposure_path, indices=np.array(indices), image_counts=counts)
    return {"samples": totals["samples"], "ce": totals["ce_sum"]/totals["samples"],
            "contrastive": totals["contrastive_sum"]/totals["samples"],
            "top1": totals["correct"]/totals["samples"],
            "eligible_anchors": totals["eligible_anchors"], "positive_pairs": totals["positive_pairs"],
            "unique_images": int(np.count_nonzero(counts)), "exposure_sha256": sha256(exposure_path),
            "amp_skipped_steps": totals["amp_skipped_steps"]}


def train_factor(arm, target, prepared, cfg, run, crop_cache=None):
    directory = run / arm
    directory.mkdir(exist_ok=True)
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True) if (directory / "last.pt").exists() else None
    if checkpoint and checkpoint["fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Checkpoint provenance changed")
    if checkpoint and checkpoint["complete"]:
        status(run, "reusing_completed_model", arm=arm, epochs=checkpoint["epoch"])
        return
    seed_all(cfg["seed"] + (target == "disease"))
    model = FactorCNN(cfg, len(prepared["ontology"][target]), pretrained=checkpoint is None)
    initial_hash = tensor_hash(model.state_dict()) if checkpoint is None else checkpoint["initial_hash"]
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model = to_device(model, cfg)
    train_transform, eval_transform, preprocess = transforms_for(model.net, cfg["image_size"])
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"], init_scale=cfg["amp_initial_scale"])
    history = checkpoint["history"] if checkpoint else []
    best_key = tuple(checkpoint["best_key"]) if checkpoint else (-1.0, -float("inf"))
    stale = checkpoint["stale"] if checkpoint else 0
    best_epoch = checkpoint["best_epoch"] if checkpoint else None
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
    write_json(directory / "initialization.json", {"initial_hash": initial_hash,
               "pretrained_sha256": prepared["pretrained_sha256"], "seed": cfg["seed"]+(target == "disease"),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "preprocessing": preprocess, "train_transform": str(train_transform), "eval_transform": str(eval_transform)})
    start = len(history)
    del checkpoint
    gc.collect()
    for epoch in range(start, cfg["epochs"]):
        begin = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"]*lr_factor(epoch, cfg)
        training = training_epoch(model, target, arm, prepared, cfg, optimizer, scaler, train_transform, run, epoch)
        status(run, "validating", arm=arm, epoch=epoch+1)
        val = validation(model, target, prepared, cfg, crop_cache, eval_transform)
        key = (val["selection_score"], -val["mean_ce"])
        improved = key > best_key
        if improved:
            best_key, stale, best_epoch = key, 0, epoch+1
            save_checkpoint(directory / "best.pt", {"model": model.state_dict(), "arm": arm, "target": target,
                "epoch": epoch+1, "validation": val, "fingerprint": prepared["fingerprint"], "initial_hash": initial_hash})
        else:
            stale += 1
        complete = epoch+1 == cfg["epochs"] or (epoch+1 >= cfg["minimum_epochs"] and stale >= cfg["patience"])
        item = {"epoch": epoch+1, "training": training, "validation": val,
                "seconds": time.perf_counter()-begin, "improved": improved,
                "backbone_lr": optimizer.param_groups[0]["lr"]}
        history.append(item)
        save_checkpoint(directory / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "arm": arm, "target": target,
            "epoch": epoch+1, "history": history, "best_key": list(best_key), "best_epoch": best_epoch,
            "stale": stale, "complete": complete, "initial_hash": initial_hash, "fingerprint": prepared["fingerprint"]})
        write_json(directory / "history.json", history)
        write_json(directory / "summary.json", {"complete": complete, "epochs": epoch+1,
                   "selected_epoch": best_epoch, "selected_score": best_key[0],
                   "seconds": sum(h["seconds"] for h in history), "initial_hash": initial_hash})
        status(run, "epoch_complete", arm=arm, epoch=epoch+1, selected_epoch=best_epoch,
               selection_score=val["selection_score"], train_top1=training["top1"],
               seconds=item["seconds"], complete=complete)
        if complete:
            break
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()


def load_selected(arm, target, prepared, cfg, run):
    saved = torch.load(run / arm / "best.pt", map_location="cpu", weights_only=True)
    if saved["fingerprint"] != prepared["fingerprint"]:
        raise ValueError("Selected model fingerprint mismatch")
    model = FactorCNN(cfg, len(prepared["ontology"][target]), pretrained=False)
    model.load_state_dict(saved["model"])
    del saved
    return to_device(model, cfg)


def cache_crop(prepared, cfg, run, roles):
    model = load_selected("crop", "crop", prepared, cfg, run)
    _, transform, _ = transforms_for(model.net, cfg["image_size"])
    results = {}
    for role in roles:
        path = run / f"crop_{role}.npz"
        if path.exists():
            bank = np.load(path)
            if bank["checkpoint_hash"].item() != sha256(run / "crop" / "best.pt"):
                raise ValueError("Crop cache hash mismatch")
            results[role] = torch.from_numpy(bank["logits"].copy())
        else:
            status(run, "caching_selected_crop", role=role)
            results[role] = predict(model, prepared["roles"][role], cfg, prepared["ontology"], transform)
            np.savez_compressed(path, logits=results[role].numpy(), checkpoint_hash=sha256(run / "crop" / "best.pt"))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def preflight(prepared, cfg, run):
    seed_all(cfg["seed"]+1)
    model = FactorCNN(cfg, len(prepared["ontology"]["disease"]))
    before = tensor_hash(model.net.state_dict())
    model = to_device(model, cfg)
    transform, _, _ = transforms_for(model.net, cfg["image_size"])
    local = {**cfg, "workers": 0}
    sampler = EpochBatches(prepared["roles"]["fit"], cfg, grouped=True)
    data = loader(prepared["roles"]["fit"], local, prepared["ontology"], transform, sampler)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"], init_scale=cfg["amp_initial_scale"])
    healthy = prepared["ontology"]["disease"].index(prepared["ontology"]["healthy_disease_id"])
    torch.cuda.reset_peak_memory_stats()
    for images, labels in data:
        labels = labels.to(cfg["device"])
        with amp(cfg):
            logits, z = model(image_device(images, cfg))
            ce = F.cross_entropy(logits, labels[:, 1], label_smoothing=cfg["label_smoothing"])
        contrast, detail = cross_crop_supcon(z, labels[:, 1], labels[:, 0], cfg["contrastive_temperature"], healthy)
        if detail["eligible_anchors"]:
            break
    scaler.scale(ce+cfg["contrastive_weight"]*contrast).backward()
    scaler.unscale_(optimizer)
    backbone_gradient = float(model.net.stem[0].weight.grad.float().norm())
    projection_gradient = float(model.projector[0].weight.grad.float().norm())
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    after = tensor_hash(model.net.state_dict())
    if not (backbone_gradient > 0 and projection_gradient > 0 and before != after):
        raise RuntimeError("Real CNN preflight failed to update backbone and projection")
    model.eval()
    with torch.inference_mode(), amp(cfg):
        image = image_device(images[:2], cfg)
        wrapper = model(image)[0]
        reference = model.net(image)
        if not torch.equal(wrapper, reference):
            raise RuntimeError("Feature extraction changed the original classifier forward")
    write_json(run / "preflight.json", {"passed": True, "at_utc": stamp(),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "python": platform.python_version(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "backbone_gradient": backbone_gradient,
        "projection_gradient": projection_gradient, "backbone_before": before, "backbone_after": after,
        "eligible_anchors": detail["eligible_anchors"], "ce": float(ce.detach()), "supcon": float(contrast.detach()),
        "classifier_forward_exact": True, "preflight_update_discarded": True})
    status(run, "preflight_passed", peak_allocated_bytes=torch.cuda.max_memory_allocated())
    del model, optimizer, scaler, data
    gc.collect()
    torch.cuda.empty_cache()


def frozen_selection(prepared, run):
    selected = {}
    for arm in ("crop", *ARMS):
        last = torch.load(run / arm / "last.pt", map_location="cpu", weights_only=True)
        if not last["complete"]:
            raise ValueError("Cannot freeze an incomplete training")
        best = torch.load(run / arm / "best.pt", map_location="cpu", weights_only=True)
        selected[arm] = {"epoch": best["epoch"], "sha256": sha256(run / arm / "best.pt"),
                         "selection_score": best["validation"]["selection_score"],
                         "initial_hash": best["initial_hash"]}
        del last, best
    value = {"frozen_utc": stamp(), "fingerprint": prepared["fingerprint"], "selected": selected,
             "primary_contrast": "ce_supcon minus ce_balanced", "official_scores_used": False,
             "audit_scores_used": False}
    path = run / "selection_frozen_before_audit_and_test.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["selected"] != selected or old["fingerprint"] != prepared["fingerprint"]:
            raise ValueError("Previously frozen selection changed")
    else:
        write_json(path, value)


def evaluate(prepared, cfg, run, crop_cache):
    roles = ("seen_validation", "development", "audit", "official_seen", "official_unseen")
    crop_cache.update(cache_crop(prepared, cfg, run, [r for r in roles if r not in crop_cache]))
    report = {"protocol": "All 333 candidates; unchanged independent log-probability product ranking",
              "candidate_count": 333, "metric_units": "fractions", "arms": {}}
    for arm in ARMS:
        model = load_selected(arm, "disease", prepared, cfg, run)
        _, transform, _ = transforms_for(model.net, cfg["image_size"])
        values = {}
        for role in roles:
            rows = prepared["roles"][role]
            path = run / arm / f"logits_{role}.npz"
            if path.exists():
                bank = np.load(path)
                if bank["checkpoint_hash"].item() != sha256(run / arm / "best.pt"):
                    raise ValueError("Existing disease logits belong to a different checkpoint")
                disease = torch.from_numpy(bank["disease"].copy())
            else:
                status(run, "final_inference", arm=arm, role=role)
                disease = predict(model, rows, cfg, prepared["ontology"], transform)
                np.savez_compressed(path, crop=crop_cache[role].numpy(), disease=disease.numpy(),
                    filenames=[r["filename"] for r in rows], crop_targets=[r["crop_id"] for r in rows],
                    disease_targets=[r["disease_id"] for r in rows], composition_targets=[r["composition_id"] for r in rows],
                    checkpoint_hash=sha256(run / arm / "best.pt"))
            values[role] = metrics(crop_cache[role], disease, rows, prepared["ontology"])
            candidates = prepared["ontology"]["candidates"]
            ranking = joint_scores(crop_cache[role], disease, prepared["ontology"]).argsort(dim=1, descending=True, stable=True)[:, :5]
            with (run / arm / f"predictions_{role}.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "true_composition_id", "predicted_composition_id", "top5_ids"])
                for row, ranks in zip(rows, ranking.tolist()):
                    ids = [candidates[i]["composition_id"] for i in ranks]
                    writer.writerow([row["filename"], row["composition_id"], ids[0], json.dumps(ids)])
        values["official_H"] = {key: harmonic(values["official_seen"]["composition"][key], values["official_unseen"]["composition"][key])
                                  for key in ("top1", "top5", "macro_top1")}
        values["audit_H"] = harmonic(values["seen_validation"]["composition"]["top1"], values["audit"]["composition"]["top1"])
        # Some official-seen pairs are deliberately withheld in this new study.
        retained = {r["composition_id"] for r in prepared["roles"]["fit"]}
        bank = np.load(run / arm / "logits_official_seen.npz")
        for name, condition in (("official_seen_retained_compositions", True), ("official_seen_heldout_compositions", False)):
            idx = [i for i, r in enumerate(prepared["roles"]["official_seen"]) if (r["composition_id"] in retained) == condition]
            if idx:
                values[name] = metrics(torch.from_numpy(bank["crop"][idx]), torch.from_numpy(bank["disease"][idx]),
                                      [prepared["roles"]["official_seen"][i] for i in idx], prepared["ontology"])
        report["arms"][arm] = values
        write_json(run / "metrics.json", report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return report


def correct_bank(run, arm, role, prepared):
    bank = np.load(run / arm / f"logits_{role}.npz")
    scores = joint_scores(torch.from_numpy(bank["crop"]), torch.from_numpy(bank["disease"]), prepared["ontology"])
    ids = np.array([c["composition_id"] for c in prepared["ontology"]["candidates"]])
    return ids[scores.argmax(1).numpy()] == bank["composition_targets"]


def analyze(prepared, cfg, run, report):
    analysis = {}
    for a, b in (("ce_supcon", "ce_balanced"), ("ce_balanced", "ce_random"), ("ce_supcon", "ce_random")):
        comparison = {}
        for method in ("image", "composition"):
            draws = {}
            for role in ("official_seen", "official_unseen", "seen_validation", "audit"):
                ca, cb = correct_bank(run, a, role, prepared), correct_bank(run, b, role, prepared)
                draws[role] = paired_bootstrap(ca, cb, prepared["roles"][role], cfg, method == "composition")
            def interval(value):
                return np.percentile(value*100, [2.5, 97.5]).tolist()
            sa, sb = draws["official_seen"]
            ua, ub = draws["official_unseen"]
            ha = np.divide(2*sa*ua, sa+ua, out=np.zeros_like(sa), where=sa+ua > 0)
            hb = np.divide(2*sb*ub, sb+ub, out=np.zeros_like(sb), where=sb+ub > 0)
            aa, ab = draws["audit"]
            comparison[method] = {"unseen_delta_95_pp": interval(ua-ub), "official_H_delta_95_pp": interval(ha-hb),
                                  "audit_delta_95_pp": interval(aa-ab)}
        comparison["point_delta_pp"] = {
            "seen": 100*(report["arms"][a]["official_seen"]["composition"]["top1"]-report["arms"][b]["official_seen"]["composition"]["top1"]),
            "unseen": 100*(report["arms"][a]["official_unseen"]["composition"]["top1"]-report["arms"][b]["official_unseen"]["composition"]["top1"]),
            "H": 100*(report["arms"][a]["official_H"]["top1"]-report["arms"][b]["official_H"]["top1"]),
            "audit": 100*(report["arms"][a]["audit"]["composition"]["top1"]-report["arms"][b]["audit"]["composition"]["top1"])}
        analysis[f"{a}_minus_{b}"] = comparison
    write_json(run / "paired_analysis.json", analysis)
    return analysis


def plots(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
    for row, arm in enumerate(ARMS):
        history = json.loads((run / arm / "history.json").read_text())
        epochs = [h["epoch"] for h in history]
        axes[row, 0].plot(epochs, [h["training"]["ce"] for h in history], label="Training CE (smoothed)")
        axes[row, 0].plot(epochs, [h["validation"]["mean_ce"] for h in history], label="Validation CE")
        for role, label in (("seen_validation", "Seen validation"), ("development", "Held-out combinations")):
            axes[row, 1].plot(epochs, [100*h["validation"][role]["disease"]["top1"] for h in history], label=label)
        axes[row, 2].plot(epochs, [100*h["validation"]["selection_score"] for h in history], color="tab:purple", label="Composition H")
        for col in range(3):
            axes[row, col].set_title(arm+": "+("Loss", "Disease accuracy (%)", "Selection score (%)")[col])
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].grid(alpha=0.2)
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=160)
    plt.close(fig)


def report_text(prepared, cfg, run, report, analysis):
    lines = ["# Cross-crop contrastive disease training: first trial", "",
        "Three matched disease arms, one independent shared crop CNN, one full training seed.",
        "All 333 candidates; no calibration. Numbers are percentages.", "",
        "| Arm | Official seen Top-1 | Official unseen Top-1 | H | Unseen disease Top-1 | Audit Top-1 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, result in report["arms"].items():
        lines.append(f"| {arm} | {100*result['official_seen']['composition']['top1']:.2f} | "
            f"{100*result['official_unseen']['composition']['top1']:.2f} | {100*result['official_H']['top1']:.2f} | "
            f"{100*result['official_unseen']['disease']['top1']:.2f} | {100*result['audit']['composition']['top1']:.2f} |")
    primary = analysis["ce_supcon_minus_ce_balanced"]
    point = primary["point_delta_pp"]
    lines += ["", "## Primary comparison: contrastive loss versus sampler-matched CE", "",
        f"Unseen composition delta: {point['unseen']:+.2f} percentage points; "
        f"95% whole-composition bootstrap interval {primary['composition']['unseen_delta_95_pp']} pp.",
        f"H delta: {point['H']:+.2f} pp; cluster interval {primary['composition']['official_H_delta_95_pp']} pp.",
        f"Untouched audit composition delta: {point['audit']:+.2f} pp; "
        f"cluster interval {primary['composition']['audit_delta_95_pp']} pp.", "",
        "## Training and selection", "",
        "| Model | Epochs completed | Selected epoch | Epoch time (minutes) |",
        "|---|---:|---:|---:|"]
    for arm in ("crop", *ARMS):
        summary = json.loads((run / arm / "summary.json").read_text())
        lines.append(f"| {arm} | {summary['epochs']} | {summary['selected_epoch']} | {summary['seconds']/60:.1f} |")
    lines += ["", "## Limits and interpretation", "",
        "The loss weight (0.1), temperature (0.1), sampling rule and 20-epoch cap were fixed before training.",
        "All selected model hashes were frozen before any audit or official inference. All three arms are reported.",
        "The primary comparison isolates the added contrastive objective; the grouped CE versus random CE comparison measures sampling.",
        "The old independent baseline scored 71.05% seen / 12.43% unseen / 21.16% H. Its training partition and epoch budget differ; that comparison is descriptive.",
        "The 16 new held-out pairs are absent from both CNNs' fitting and seen validation. Some occur in the supplied official seen CSV; metrics.json also separates retained and withheld official-seen compositions.",
        "Intervals condition on one full initialization seed. There are only eight audit and ten official unseen compositions. Official tests are familiar from earlier studies, so evidence remains exploratory.",
        "Only cross-crop same-label pairs provide positive anchors; single-crop diseases still train by CE. Healthy labels are excluded as positive anchors. Shared symptom labels need not have identical biology or appearance across crops.",
        "Gradient accumulation does not increase the eight-image contrastive comparison set. Stronger conclusions require independent CNN seeds and fresh composition holdouts.",
        "Training CE uses label smoothing; validation CE does not. Their absolute magnitudes are not directly comparable.", "",
        "Artifacts: prepared.json, role_summary.json, metrics.json, paired_analysis.json, training_curves.png, "
        "selection_frozen_before_audit_and_test.json, per-arm histories, checkpoints, exposures, predictions and raw logits."]
    (run / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = HERE / "runs" / cfg["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("This authorized experiment requires CUDA")
    with StudyLock(run / "study.lock"):
        try:
            prepared = prepare(cfg, run)
            if args.prepare_only:
                return
            if shutil.disk_usage(HERE).free < 4*1024**3:
                raise RuntimeError("Need at least 4 GiB free for atomic full-training checkpoints")
            if not (run / "preflight.json").exists():
                preflight(prepared, cfg, run)
            if args.preflight_only:
                return
            train_factor("crop", "crop", prepared, cfg, run)
            crop_cache = cache_crop(prepared, cfg, run, ("seen_validation", "development"))
            for arm in ARMS:
                train_factor(arm, "disease", prepared, cfg, run, crop_cache)
            frozen_selection(prepared, run)
            report = evaluate(prepared, cfg, run, crop_cache)
            analysis = analyze(prepared, cfg, run, report)
            plots(run)
            report_text(prepared, cfg, run, report, analysis)
            from verify import verify_run
            verification = verify_run(run)
            status(run, "complete", verification_checks=verification["checks"], results=str(run / "results.md"))
        except BaseException as error:
            status(run, "failed", error=repr(error), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
