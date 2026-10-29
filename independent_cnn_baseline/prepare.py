"""Pin data, partition, pretrained revisions and memory settings before fitting."""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .engine import atomic_json, fingerprint, now, read_json, seed_everything, sha256
from .models import IndependentNet, PROJECT, freeze_batchnorm_statistics, optimizer


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("timm", "torchvision", "numpy", "pandas", "pillow", "huggingface_hub", "safetensors")}}
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return result


def require_cuda(spec):
    if spec["runtime"]["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This approved study requires CUDA. CPU fallback is disabled.")


def audit_data(spec, root):
    from dpd.data import COLUMNS, pair_table
    from dpd.splits import make_validation_split

    data_root = (PROJECT / spec["data_root"]).resolve()
    sources = {s: data_root / name for s, name in {
        "train": "DPD_Small_train.csv", "test_seen": "DPD_Small_test_seen.csv",
        "test_unseen": "DPD_Small_test_unseen.csv"}.items()}
    frames = []
    for split, source in sources.items():
        frame = pd.read_csv(source, header=None, names=COLUMNS)
        frame["split"] = split
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if df["file"].str.casefold().duplicated().any():
        raise ValueError("Duplicate file path across official metadata")
    if df["file"].map(lambda f: Path(f).stem.casefold()).duplicated().any():
        raise ValueError("Cache filename stem collision")
    for id_col, name_col in (("crop_id", "crop"), ("disease_id", "disease"), ("pair_id", "pair")):
        if df.groupby(id_col)[name_col].nunique().max() != 1:
            raise ValueError(f"Conflicting {name_col} names for one ID")
    if df.groupby("pair_id")[["crop_id", "disease_id"]].nunique().to_numpy().max() != 1:
        raise ValueError("Conflicting composition factors")
    unseen = set(df.loc[df.split == "test_unseen", "pair_id"])
    df["is_unseen"] = df["pair_id"].isin(unseen)
    train = df[df.split == "train"].copy()
    if set(train.pair_id) & unseen:
        raise ValueError("An official unseen composition occurs in training")
    if set(df.crop_id) != set(train.crop_id) or set(df.disease_id) != set(train.disease_id):
        raise ValueError("An official test factor has no training examples")
    counts = df.groupby("split").size().to_dict()
    if counts != {"train": 8000, "test_seen": 2000, "test_unseen": 1448}:
        raise ValueError(f"Unexpected dataset counts: {counts}")
    pairs = pair_table(df)
    if len(pairs) != 333 or len(unseen) != 10 or set(train.crop_id) != set(range(55)) or set(train.disease_id) != set(range(175)):
        raise ValueError("Unexpected DPD-Small label ontology")
    metadata = {s: {"path": str(p), "sha256": sha256(p)} for s, p in sources.items()}
    provenance_file = root / "data_provenance.json"
    existing = read_json(provenance_file) if provenance_file.exists() else None

    def inspect(filename):
        path = data_root / "DPD_Small_dataset" / filename
        if path.name != filename:
            raise ValueError("Expected flat image filenames")
        with Image.open(path) as image:
            image.convert("RGB").load()
        return filename, sha256(path)

    print("Auditing all 11,448 original image files and metadata...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        image_hashes = dict(pool.map(inspect, df.file))
    by_split = {s: {image_hashes[f] for f in df.loc[df.split == s, "file"]} for s in sources}
    for a, b in (("train", "test_seen"), ("train", "test_unseen"), ("test_seen", "test_unseen")):
        if by_split[a] & by_split[b]:
            raise ValueError(f"Exact image-content leakage between {a} and {b}")
    provenance = {"metadata": metadata, "image_hashes": image_hashes}
    if existing is not None and existing != provenance:
        raise ValueError("Dataset identity changed; refusing to resume")
    atomic_json(provenance_file, provenance)
    v = spec["validation"]
    if v["n_splits"] != 1:
        raise ValueError("This approved study uses exactly one seed-42 validation split")
    fit, seen, pseudo, holdout = make_validation_split(
        train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"], spec["seed"],
        tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
    parts = {"fit": fit, "validation_seen": seen, "validation_unseen": pseudo, "all_train": train}
    if set(fit.crop_id) != set(train.crop_id) or set(fit.disease_id) != set(train.disease_id):
        raise ValueError("A factor disappeared from fitting after validation splitting")
    if set(fit.pair_id) & set(holdout) or not set(seen.pair_id).issubset(set(fit.pair_id)):
        raise ValueError("Invalid seen/pseudo-unseen validation composition partition")
    used = set(fit.file) | set(seen.file) | set(pseudo.file)
    excluded = train[~train.file.isin(used)]
    if not set(excluded.pair_id).issubset(set(holdout)):
        raise ValueError("Selection exclusions must come only from capped held-out pairs")
    for a, b in (("fit", "validation_seen"), ("fit", "validation_unseen"), ("validation_seen", "validation_unseen")):
        if {image_hashes[f] for f in parts[a].file} & {image_hashes[f] for f in parts[b].file}:
            raise ValueError(f"Exact image-content leakage between {a} and {b}")
    for name, frame in {**parts, "selection_excluded": excluded}.items():
        path = root / f"{name}.csv"
        serialized = frame.to_csv(index=False, lineterminator="\n")
        if path.exists() and path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Saved {name} partition changed")
        if not path.exists():
            path.write_text(serialized, encoding="utf-8", newline="")
    report = {"official_counts": counts, "partition_counts": {k: len(v) for k, v in parts.items()},
              "pseudo_unseen_pair_ids": holdout, "split_seed": spec["seed"], "n_splits": 1,
              "selection_excluded_count": len(excluded),
              "selection_exclusion_reason": "Held-out pair images above the fixed 40-image validation cap; included in final full-data refit",
              "images_verified": True, "exact_hash_leakage_checked": True,
              "data_identity": fingerprint(provenance)}
    atomic_json(root / "audit.json", report)
    atomic_json(root / "pairs.json", pairs.to_dict(orient="records"))
    print(f"Saved validation partition: {report['partition_counts']}", flush=True)
    return df, pairs, parts, report


def base_config(spec, model, prepared_model):
    config = {k: copy.deepcopy(spec[k]) for k in ("seed", "input", "augmentation", "optimizer", "training", "runtime")}
    config.update(method="cnn_independent", implementation="independent_cnn_baseline", display_name="Independent CNN Baseline")
    config["model"] = {**copy.deepcopy(model), "pretrained_file": prepared_model["pretrained_file"],
                       "pretrained_sha256": prepared_model["sha256"], "pretrained_revision": prepared_model["revision"],
                       "head_dropout": spec["training"]["head_dropout"]}
    config["input"].update(prepared_model["preprocessing"])
    config["training"].update(batch_size=prepared_model["batch_size"],
                              accumulation_steps=spec["training"]["effective_batch_size"] // prepared_model["batch_size"])
    return config


def _smoke(config, batch_size):
    model = opt = x = y = logits = loss = None
    try:
        seed_everything(config["seed"])
        model = IndependentNet(config, "disease", pretrained=True).cuda()
        model.train()
        if config["model"]["freeze_bn_stats"]:
            freeze_batchnorm_statistics(model)
        opt = optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.randn(batch_size, 3, config["input"]["crop_size"], config["input"]["crop_size"], device="cuda")
        y = torch.arange(batch_size, device="cuda") % 175
        torch.cuda.reset_peak_memory_stats()
        timings, updates = [], 0
        for step in range(5):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.float(), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite CUDA smoke-test loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            updates += int(scaler.get_scale() >= before)
            torch.cuda.synchronize()
            if step >= 2:
                timings.append(time.monotonic() - started)
        if not updates or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("CUDA optimizer smoke check did not produce finite updated weights")
        return {"batch_size": batch_size, "median_step_seconds": float(np.median(timings)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "successful_optimizer_updates": updates,
                "parameters": sum(p.numel() for p in model.parameters()), "normalization_modules_frozen":
                sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not m.training for m in model.modules())}
    finally:
        model = opt = x = y = logits = loss = None
        gc.collect()
        torch.cuda.empty_cache()


def prepare_models(spec, root):
    import timm
    from huggingface_hub import HfApi, hf_hub_download

    prepared_file = root / "prepared_models.json"
    records = read_json(prepared_file) if prepared_file.exists() else {}
    for entry in spec["models"]:
        name = entry["name"]
        if name in records:
            saved = records[name]
            if saved["backbone"] != entry["backbone"] or sha256(PROJECT / saved["pretrained_file"]) != saved["sha256"]:
                raise ValueError(f"Pinned initialization changed for {name}")
            continue
        repo_id = "timm/" + entry["backbone"]
        pin_file = root / f"initialization_{name}.json"
        if pin_file.exists():
            pin = read_json(pin_file)
            revision = pin["revision"]
        else:
            revision = HfApi().model_info(repo_id).sha
            atomic_json(pin_file, {"repo_id": repo_id, "revision": revision})
        print(f"Caching pinned pretrained model {repo_id}@{revision}", flush=True)
        path = Path(hf_hub_download(repo_id, "model.safetensors", revision=revision))
        native = timm.get_pretrained_cfg(entry["backbone"]).to_dict()
        saved = {"backbone": entry["backbone"], "revision": revision, "sha256": sha256(path),
                 "pretrained_file": os.path.relpath(path, PROJECT).replace("\\", "/"),
                 "native_input_size": native["input_size"],
                 "preprocessing": {k: native[k] for k in ("mean", "std", "interpolation")},
                 "batch_size": spec["training"]["initial_batch_size"]}
        config = base_config(spec, entry, saved)
        while True:
            try:
                print(f"CUDA forward/backward/optimizer check: {name}, physical batch {saved['batch_size']}", flush=True)
                smoke = _smoke(config, saved["batch_size"])
                break
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if saved["batch_size"] == 1:
                    raise
                saved["batch_size"] //= 2
        saved.update(smoke=smoke, prepared_at_utc=now())
        records[name] = saved
        atomic_json(prepared_file, records)
        print(f"Prepared {name}: batch={saved['batch_size']}, peak={smoke['peak_allocated_bytes']/2**30:.2f} GiB, "
              f"synthetic step={smoke['median_step_seconds']:.3f}s", flush=True)
    return records
