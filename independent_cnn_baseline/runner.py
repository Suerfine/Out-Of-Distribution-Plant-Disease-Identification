"""Run from the repository root: python -m independent_cnn_baseline.runner --help."""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT / "runs/cache/independent_cnn_baseline/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "runs/cache/independent_cnn_baseline/matplotlib"))
os.environ.setdefault("CUDA_CACHE_PATH", str(PROJECT / "runs/cache/independent_cnn_baseline/cuda"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd
import torch

from .engine import (Actor, StudyLock, Tee, atomic_checkpoint, atomic_json, choose, fingerprint,
                     keep_awake, now, plot_history, predict, prune_recovery, read_json,
                     select_gamma, sha256)
from .models import IndependentNet
from .prepare import audit_data, base_config, environment, prepare_models, require_cuda


class WindowEnd(Exception):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise WindowEnd()


def window_timing(previous, duration, deadline_utc=None):
    """Recovery may reuse the original deadline; a new window needs a new launch."""
    current = datetime.now(timezone.utc)
    if deadline_utc is None:
        end = current + timedelta(hours=duration)
        fields = {"window_started_at_utc": current.isoformat(), "window_hours": duration}
    else:
        end = datetime.fromisoformat(deadline_utc)
        if end.tzinfo is None or not previous.get("window_started_at_utc"):
            raise ValueError("Recovery deadline requires a timezone and an existing window")
        start = datetime.fromisoformat(previous["window_started_at_utc"])
        if end != start + timedelta(hours=previous["window_hours"]):
            raise ValueError("Recovery deadline must equal the original window deadline")
        fields = {"window_started_at_utc": previous["window_started_at_utc"],
                  "window_hours": previous["window_hours"], "resumed_at_utc": current.isoformat()}
    fields["window_deadline_at_utc"] = end.astimezone(timezone.utc).isoformat()
    return time.monotonic() + max(0, (end - current).total_seconds()), fields


def validate_spec(spec):
    expected = {"resnetv2_50x1_bit.goog_in21k_ft_in1k", "convnext_base.fb_in22k_ft_in1k", "tf_efficientnetv2_l.in21k_ft_in1k"}
    if len(spec["models"]) != 3 or {m["backbone"] for m in spec["models"]} != expected:
        raise ValueError("Expected exactly the three approved pretrained backbones")
    if spec["seed"] != 42 or spec["validation"]["n_splits"] != 1:
        raise ValueError("Expected the approved single seed-42 validation partition")
    if len({m["name"] for m in spec["models"]}) != 3 or any(not m["name"].replace("_", "").isalnum() for m in spec["models"]):
        raise ValueError("Unsafe or duplicate model names")
    root = (PROJECT / spec["output_dir"]).resolve()
    if not root.is_relative_to(PROJECT / "runs/baseline") or not root.name.startswith("independent_cnn_baseline_"):
        raise ValueError("Study output must be a new independent_cnn_baseline folder under runs/baseline")
    train = spec["training"]
    if train["effective_batch_size"] % train["initial_batch_size"] or train["initial_batch_size"] not in (1, 2, 4):
        raise ValueError("Invalid approved physical/effective batch")
    if not 0 <= train["warmup_epochs"] < train["max_epochs"] or train["patience"] < 1:
        raise ValueError("Invalid schedule")
    if any(not math.isfinite(x) or x <= 0 for x in spec["search"]["learning_rates"]):
        raise ValueError("Invalid LR search")
    if len(spec["search"]["learning_rates"]) != 3 or len(spec["search"]["alternative_head_dropout"]) != 2:
        raise ValueError("Expected R0-R5: three LRs, one WD and two independent dropout alternatives")
    gamma = np.arange(*spec["validation"]["gamma_grid"])
    if not len(gamma) or gamma[0] != 0 or not np.isfinite(gamma).all() or np.any(np.diff(gamma) <= 0):
        raise ValueError("Gamma grid must start at zero and increase")
    return root


def candidate_complete(folder, identity):
    path = Path(folder) / "complete.json"
    if not path.exists():
        return None
    row = read_json(path)
    if row["fingerprint"] != identity:
        raise ValueError("Completed job fingerprint mismatch")
    for target, digest in row["checkpoint_sha256"].items():
        if sha256(Path(folder) / target / "best.pt") != digest:
            raise ValueError("Completed job checkpoint changed")
    history = read_json(Path(folder) / "history.json")
    selected = max(history, key=lambda r: r["harmonic_mean_top1"])
    if selected["epoch"] != row["selected_epoch"] or len(history) != row["completed_epochs"]:
        raise ValueError("Completed job selection/history mismatch")
    return row


def run_pair(config, folder, data_identity, frames, pairs, cache_dir, spec, deadline, update,
             model_factory=IndependentNet):
    from dpd.cnn import compose_pair_scores
    from dpd.evaluate import evaluate

    folder = Path(folder)
    val = pd.concat([frames["validation_seen"], frames["validation_unseen"]], ignore_index=True)
    identity = fingerprint({"config": config, "data": data_identity,
                            "fit": frames["fit"].file.tolist(), "validation": val.file.tolist()})
    complete = candidate_complete(folder, identity)
    if complete:
        print(f"Keeping completed {config['version']}", flush=True)
        return complete
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists() and fingerprint(read_json(config_path)) != fingerprint(config):
        raise ValueError("Existing trial config changed")
    atomic_json(config_path, config)
    history = read_json(folder / "history.json") if (folder / "history.json").exists() else []
    best = read_json(folder / "best.json") if (folder / "best.json").exists() else None
    actors = {t: Actor(t, config, folder / t, identity, model_factory) for t in ("crop", "disease")}
    n_seen = len(frames["validation_seen"])
    labels = val.pair_id.to_numpy()
    pseudo_ids = set(frames["validation_unseen"].pair_id)
    val_pairs = pairs.assign(is_unseen=pairs.pair_id.isin(pseudo_ids) | pairs.is_unseen)
    allowed = ~pairs.is_unseen.to_numpy()
    gammas = np.round(np.arange(*spec["validation"]["gamma_grid"]), 6)
    start_epoch = len(history) + 1
    try:
        for epoch in range(start_epoch, config["training"]["max_epochs"] + 1):
            if best and epoch - 1 - best["epoch"] >= config["training"]["patience"]:
                break
            for target, actor in actors.items():
                if actor.epoch < epoch:
                    check_deadline(deadline)
                    update(phase="selection", active={"trial": config["version"], "target": target, "epoch": epoch})
                    actor.epoch_step(frames["fit"], val, cache_dir)
                if actor.epoch != epoch:
                    raise ValueError("Mismatched paired recovery epochs")
            logp = {t: torch.log_softmax(a.val_logits, 1).numpy() for t, a in actors.items()}
            scores = compose_pair_scores(logp["crop"], logp["disease"], pairs)
            row = select_gamma(scores, labels, val_pairs.is_unseen.to_numpy(), allowed, n_seen, gammas)
            hidden_scores = scores.copy()
            hidden_scores[:, ~allowed] = -np.inf
            metrics = evaluate(hidden_scores[:n_seen], labels[:n_seen], hidden_scores[n_seen:], labels[n_seen:],
                               val_pairs, row["gamma"])
            if abs(metrics["harmonic_mean_top1"] - row["harmonic_mean_top1"]) > 1e-12:
                raise ValueError("Gamma selection disagrees with the shared evaluator")
            row.update(epoch=epoch, seen_top5=metrics["test_seen"]["top5"],
                       unseen_top5=metrics["test_unseen"]["top5"], harmonic_mean_top5=metrics["harmonic_mean_top5"])
            if best is None or row["harmonic_mean_top1"] > best["validation"]["harmonic_mean_top1"]:
                for target, actor in actors.items():
                    actor.save_weights(folder / target / "best.pt")
                best = {"epoch": epoch, "validation": row}
                atomic_json(folder / "best.json", best)
                np.savez_compressed(folder / "best_validation_logits.npz", **{t: a.val_logits.numpy() for t, a in actors.items()})
            history.append(row)
            atomic_json(folder / "history.json", history)
            print(f"{config['version']} paired epoch {epoch}: val H={row['harmonic_mean_top1']:.2%}, "
                  f"gamma={row['gamma']:g}, gamma-zero H={row['gamma0_harmonic_mean_top1']:.2%}", flush=True)
        if best is None:
            raise ValueError("No paired validation epoch completed")
        checkpoints = {t: sha256(folder / t / "best.pt") for t in actors}
        result = {"name": config["version"], "model_id": config["model"]["name"], "config": config,
                  "folder": str(folder.relative_to(PROJECT)), "fingerprint": identity,
                  "selected_epoch": best["epoch"], "validation": best["validation"],
                  "completed_epochs": len(history), "checkpoint_sha256": checkpoints,
                  "training_epoch_seconds": sum(r["seconds"] for a in actors.values() for r in a.history),
                  "completed_at_utc": now()}
        for target in actors:
            saved = torch.load(folder / target / "best.pt", map_location="cpu", weights_only=True)
            if saved["epoch"] != best["epoch"] or saved["fingerprint"] != identity:
                raise ValueError("Paired best checkpoint transaction is incomplete")
        plot_history(folder)
        atomic_json(folder / "complete.json", result)
        prune_recovery(folder, PROJECT / spec["output_dir"])
        return result
    finally:
        actors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def refit(config, chosen, folder, data_identity, train, cache_dir, deadline, update,
          model_factory=IndependentNet):
    folder = Path(folder)
    identity = fingerprint({"config": config, "data": data_identity, "train": train.file.tolist(),
                            "epochs": chosen["selected_epoch"], "phase": "refit"})
    marker = folder / "refit_complete.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["fingerprint"] != identity or any(sha256(folder / f"{t}_net.pt") != v for t, v in saved["checkpoint_sha256"].items()):
            raise ValueError("Completed refit changed")
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "config.json").exists() and fingerprint(read_json(folder / "config.json")) != fingerprint(config):
        raise ValueError("Existing final refit configuration changed")
    atomic_json(folder / "config.json", config)
    atomic_json(folder / "selected_source.json", chosen)
    recovery = folder / "refit_recovery"
    for target in ("crop", "disease"):
        output = folder / f"{target}_net.pt"
        done = folder / f"{target}_refit_complete.json"
        if done.exists():
            mark = read_json(done)
            if mark["fingerprint"] != identity or sha256(output) != mark["sha256"]:
                raise ValueError("Completed target refit changed")
            continue
        check_deadline(deadline)
        actor = Actor(target, config, recovery / target, identity, model_factory)
        while actor.epoch < chosen["selected_epoch"]:
            check_deadline(deadline)
            update(phase="refit", active={"model": config["model"]["name"], "target": target, "epoch": actor.epoch + 1})
            actor.epoch_step(train, None, cache_dir)
        atomic_checkpoint(output, actor.model.state_dict())
        atomic_json(folder / f"{target}_refit_history.json", actor.history)
        atomic_json(done, {"fingerprint": identity, "epochs": actor.epoch, "sha256": sha256(output)})
        # The final weights and completion marker make this target independently recoverable.
        last = (recovery / target / "last.pt").resolve()
        if not last.is_relative_to(folder.resolve()):
            raise ValueError("Unsafe recovery path")
        last.unlink(missing_ok=True)
        del actor
        gc.collect()
    result = {"fingerprint": identity, "selected_epoch": chosen["selected_epoch"], "completed_at_utc": now(),
              "checkpoint_sha256": {t: sha256(folder / f"{t}_net.pt") for t in ("crop", "disease")}}
    atomic_json(marker, result)
    shutil.copyfile(PROJECT / chosen["folder"] / "learning_curves.png", folder / "selection_learning_curves.png")
    return result


def evaluate_final(config, chosen, folder, df, pairs, cache_dir, deadline, update):
    from dpd.cnn import compose_pair_scores
    from dpd.evaluate import TIE_BREAK_POLICY, evaluate
    from dpd.report import save_test_outputs
    from .inference import IndependentCnnRunModel

    folder = Path(folder)
    completion = folder / "completion_verified.json"
    if completion.exists():
        checked = read_json(completion)
        for name, digest in checked["artifact_sha256"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Completed official evaluation changed")
        return read_json(folder / "metrics.json")
    splits = {s: df[df.split == s] for s in ("test_seen", "test_unseen")}
    cache = folder / "evaluation_recovery"
    cache.mkdir(exist_ok=True)
    outputs = {s: {} for s in splits}
    for target in ("crop", "disease"):
        model = None
        for split, frame in splits.items():
            path = cache / f"{target}_{split}.pt"
            expected = {"checkpoint_sha256": sha256(folder / f"{target}_net.pt"), "files": frame.file.tolist()}
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["identity"] != expected:
                    raise ValueError("Saved evaluation logits do not match selected weights/data")
                outputs[split][target] = saved["logits"]
                continue
            check_deadline(deadline)
            update(phase="evaluation", active={"model": config["model"]["name"], "target": target, "split": split})
            if model is None:
                model = IndependentNet(config, target, pretrained=False)
                model.load_state_dict(torch.load(folder / f"{target}_net.pt", map_location="cpu", weights_only=True))
                model.to("cuda")
            logits = predict(model, frame, cache_dir, config, batch_size=config["runtime"]["inference_batch_size"])
            atomic_checkpoint(path, {"identity": expected, "logits": logits})
            outputs[split][target] = logits
        del model
        gc.collect()
        torch.cuda.empty_cache()
    scores, labels, files = {}, {}, {}
    for split, frame in splits.items():
        logp = {t: torch.log_softmax(x, 1).numpy() for t, x in outputs[split].items()}
        scores[split] = compose_pair_scores(logp["crop"], logp["disease"], pairs)
        labels[split], files[split] = frame.pair_id.to_numpy(), frame.file.to_numpy()
    gamma = chosen["validation"]["gamma"]
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma)
    uncalibrated = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0)
    result.update(version=config["version"], selected_epoch=chosen["selected_epoch"], val=chosen["validation"],
                  harmonic_mean_top1_gamma0=uncalibrated["harmonic_mean_top1"],
                  raw_factor_metrics={s: {t: {"top1": float((x.argmax(1) == torch.tensor(splits[s][t + '_id'].to_numpy())).float().mean())}
                                         for t, x in out.items()} for s, out in outputs.items()})
    staging = folder / "evaluation_staging"
    staging.mkdir(exist_ok=True)
    save_test_outputs(staging, result, scores, labels, files, pairs, gamma)
    atomic_json(staging / "uncalibrated_metrics.json", uncalibrated)
    np.savez_compressed(staging / "factor_logits.npz", **{f"{s}_{t}": x.numpy() for s, out in outputs.items() for t, x in out.items()})
    for split, expected_count in (("test_seen", 2000), ("test_unseen", 1448)):
        if len(pd.read_csv(staging / f"predictions_{split}.csv")) != expected_count or not np.isfinite(scores[split]).all():
            raise ValueError("Official prediction count/non-finite score mismatch")
    matrix = pd.read_csv(staging / "confusion_test_seen.csv", index_col=0)
    if matrix.shape != (323, 333) or int(matrix.to_numpy().sum()) != 2000:
        raise ValueError("Seen confusion matrix mismatch")
    run = {"dir": folder, "cfg": config, "metrics": result}
    app = IndependentCnnRunModel(run, pairs)
    parity = []
    data_root = Path(os.environ["DPD_ROOT"])
    for split, frame in splits.items():
        for index in np.linspace(0, len(frame) - 1, 4, dtype=int):
            embedding, actual = app.predict(data_root / "DPD_Small_dataset" / frame.iloc[index].file)
            error = float(np.abs(actual - scores[split][index]).max())
            if not np.isfinite(embedding).all() or error > 1e-4 or actual.argmax() != scores[split][index].argmax():
                raise ValueError(f"Shared-app inference mismatch: {split}/{index}, max difference {error}")
            parity.append({"split": split, "row": int(index), "max_score_difference": error})
    del app
    atomic_json(staging / "app_inference_verification.json", parity)
    # Publish metrics last: the dashboard cannot discover a half-written run.
    names = [p.name for p in staging.iterdir() if p.name != "metrics.json"] + ["metrics.json"]
    for name in names:
        (staging / name).replace(folder / name)
    artifact_names = names + ["config.json", "crop_net.pt", "disease_net.pt", "selected_source.json"]
    verification = {"verified_at_utc": now(), "prediction_counts": {s: len(f) for s, f in splits.items()},
                    "candidate_count": 333, "seen_confusion_shape": [323, 333], "seen_confusion_count": 2000,
                    "app_inference_checks": parity, "finite_scores": True, "tie_break_policy": TIE_BREAK_POLICY,
                    "artifact_sha256": {name: sha256(folder / name) for name in artifact_names}}
    atomic_json(completion, verification)
    atomic_json(folder / "results.json", {"calibrated": result, "uncalibrated": uncalibrated})
    return result


def study(spec_path, resume=False, hours=None, prepare_only=False, deadline_utc=None):
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    root = validate_spec(spec)
    os.environ["DPD_ROOT"] = str((PROJECT / spec["data_root"]).resolve())
    require_cuda(spec)
    if deadline_utc is not None and not resume:
        raise ValueError("Reusing a window deadline requires --resume")
    torch.set_num_threads(spec["runtime"]["torch_threads"])
    root.mkdir(parents=True, exist_ok=True)
    with StudyLock(root / "runner.lock"), keep_awake():
        marker = root / "study_fingerprint.txt"
        identity = fingerprint(spec)
        if marker.exists() and marker.read_text().strip() != identity:
            raise ValueError("Approved study settings changed; use a new output directory")
        marker.write_text(identity + "\n", encoding="utf-8")
        previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if previous.get("state") == "completed":
            print(f"Study already completed: {root / 'study_summary.json'}")
            return
        if prepare_only and (previous.get("window_started_at_utc") or previous.get("state") not in (None, "prepared", "failed")):
            raise FileExistsError("Preparation cannot replace a started study's status. Use --resume.")
        if previous.get("state") not in (None, "prepared") and not resume and not prepare_only:
            raise FileExistsError("Study already started. Confirm its controller exited, then use --resume.")
        state = {**previous, "state": "running", "pid": os.getpid(), "active": None,
                 "started_at_utc": previous.get("started_at_utc", now()), "completed_trials": previous.get("completed_trials", []),
                 "total_selection_trials": 18, "total_selection_target_trainings": 36, "total_final_target_refits": 6}
        state.pop("error", None)
        (root / "runner.pid").write_text(str(os.getpid()), encoding="utf-8")

        def update(**values):
            state.update(values, updated_at_utc=now())
            atomic_json(root / "status.json", state)

        with (root / "training.log").open("a", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                try:
                    print(f"\nIndependent CNN Baseline controller {now()}, PID {os.getpid()}", flush=True)
                    update(phase="preflight")
                    atomic_json(root / "study_config.json", spec)
                    atomic_json(root / "environment.json", environment())
                    source_files = list((PROJECT / "independent_cnn_baseline").glob("*.py"))
                    atomic_json(root / "code_provenance.json", {"recorded_at_utc": now(), "files":
                                {str(p.relative_to(PROJECT)): sha256(p) for p in source_files}})
                    prepared = prepare_models(spec, root)
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    df, pairs, frames, audit = audit_data(spec, root)
                    from dpd.resize_cache import ensure_resized
                    cache_dir = ensure_resized(df.file, spec["input"]["resize_cache"])
                    if prepare_only:
                        update(state="prepared", phase="prepared", active=None)
                        print("Preflight completed; no dataset training has started.", flush=True)
                        return
                    duration = spec["window_hours"] if hours is None else hours
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("Window hours must be positive and finite")
                    deadline, timing = window_timing(previous, duration, deadline_utc)
                    update(**timing, phase="selection")
                    pools = {m["name"]: [] for m in spec["models"]}

                    def run(model, index, overrides):
                        check_deadline(deadline)
                        cfg = base_config(spec, model, prepared[model["name"]])
                        cfg["optimizer"].update(overrides.get("optimizer", {}))
                        cfg["model"].update(overrides.get("model", {}))
                        cfg["version"] = f"independent_cnn_baseline_{model['name']}_r{index}_seed42"
                        folder = root / "trials" / cfg["version"]
                        cfg["output_dir"] = str(folder.relative_to(PROJECT)).replace("\\", "/")
                        generated = PROJECT / "independent_cnn_baseline/configs" / spec["name"] / (cfg["version"] + ".json")
                        if generated.exists() and fingerprint(read_json(generated)) != fingerprint(cfg):
                            raise ValueError("Versioned generated recipe changed")
                        atomic_json(generated, cfg)
                        result = run_pair(cfg, folder, audit["data_identity"], frames, pairs, cache_dir, spec, deadline, update)
                        pools[model["name"]].append(result)
                        state["completed_trials"] = [r for r in state["completed_trials"] if r["name"] != result["name"]] + [result]
                        update(completed_selection_target_trainings=2 * len(state["completed_trials"]))

                    for index, lr in enumerate(spec["search"]["learning_rates"]):
                        for model in spec["models"]:
                            run(model, index, {"optimizer": {"lr_backbone": lr, "lr_head": 10 * lr}})
                    selected_lr = {name: choose(pool) for name, pool in pools.items()}
                    update(selected_learning_rates=selected_lr)
                    for model in spec["models"]:
                        selected = selected_lr[model["name"]]["config"]["optimizer"]
                        run(model, 3, {"optimizer": {"lr_backbone": selected["lr_backbone"], "lr_head": selected["lr_head"],
                                                    "weight_decay": spec["search"]["alternative_weight_decay"]}})
                    selected_base = {name: choose([selected_lr[name], pool[-1]]) for name, pool in pools.items()}
                    update(selected_regularization_base=selected_base)
                    for index, dropout in enumerate(spec["search"]["alternative_head_dropout"], start=4):
                        for model in spec["models"]:
                            selected = selected_base[model["name"]]["config"]["optimizer"]
                            run(model, index, {"optimizer": copy.deepcopy(selected), "model": {"head_dropout": dropout}})
                    selected = {name: choose(pool) for name, pool in pools.items()}
                    frozen = root / "selected_models.json"
                    if frozen.exists() and read_json(frozen) != selected:
                        raise ValueError("Previously frozen final selections changed")
                    atomic_json(frozen, selected)
                    update(phase="refit", selected_models=selected, active=None)
                    final_configs = {}
                    for model in spec["models"]:
                        chosen = selected[model["name"]]
                        cfg = copy.deepcopy(chosen["config"])
                        cfg["version"] = f"independent_cnn_baseline_{model['name']}_seed42"
                        cfg["output_dir"] = "runs/baseline/" + cfg["version"]
                        final_configs[model["name"]] = cfg
                        refit(cfg, chosen, PROJECT / cfg["output_dir"], audit["data_identity"], frames["all_train"], cache_dir, deadline, update)
                    results = {}
                    for model in spec["models"]:
                        name = model["name"]
                        cfg = final_configs[name]
                        results[name] = evaluate_final(cfg, selected[name], PROJECT / cfg["output_dir"], df, pairs, cache_dir, deadline, update)
                    summary = {"completed_at_utc": now(), "study_fingerprint": identity,
                               "selection_trials": state["completed_trials"], "selected_models": selected,
                               "results": results, "audit": audit, "official_tests_used_for_selection": False,
                               "selection_training_epoch_seconds": sum(r["training_epoch_seconds"] for r in state["completed_trials"])}
                    atomic_json(root / "study_summary.json", summary)
                    lines = ["# Independent CNN Baseline study", "", "All selections used the single saved seed-42 validation partition.",
                             "", "| Backbone | Seen Top-1 | Unseen Top-1 | H Top-1 | H Top-5 | Gamma | Epoch |",
                             "|---|---:|---:|---:|---:|---:|---:|"]
                    for name, result in results.items():
                        lines.append(f"| {name} | {result['test_seen']['top1']:.2%} | {result['test_unseen']['top1']:.2%} | "
                                     f"{result['harmonic_mean_top1']:.2%} | {result['harmonic_mean_top5']:.2%} | "
                                     f"{result['gamma']:g} | {result['selected_epoch']} |")
                    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    update(state="completed", phase="completed", active=None, finished_at_utc=now(), summary_path=str(root / "study_summary.json"))
                    print(f"Completed and verified: {root / 'results.md'}", flush=True)
                except WindowEnd:
                    update(state="window_complete", phase="checkpointed", active=None, window_finished_at_utc=now())
                    print("Ten-hour window reached. Work is checkpointed and incomplete; resume this study with --resume.", flush=True)
                except BaseException:
                    update(state="failed", error=traceback.format_exc())
                    traceback.print_exc()
                    raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="independent_cnn_baseline/configs/study_seed42.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Pin weights, smoke-test CUDA and audit/cache data without fitting")
    parser.add_argument("--hours", type=float, help="Window length; defaults to the approved spec's ten hours")
    parser.add_argument("--deadline-utc", help="Recovery only: reuse exactly the previous window's UTC deadline")
    args = parser.parse_args()
    study(args.spec, args.resume, args.hours, args.prepare, args.deadline_utc)


if __name__ == "__main__":
    main()
