"""Fresh validation-only Optuna study; gamma zero governs every selection decision."""
from __future__ import annotations

import argparse
import contextlib
import gc
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sqlalchemy.pool import NullPool

from .engine import (Actor, StudyLock, Tee, atomic_json, fingerprint, keep_awake,
                     now, read_json, sha256)
from .models import IndependentNet
from .optuna_search import (SPACE, SearchStop, boundary, immutable_json, report_history,
                            request_trial, retire_recovery, sampler_for)
from .prepare import _smoke, environment, require_cuda
from .runner import window_timing
from .uncalibrated_objective import OBJECTIVE_NAME, validation_uncalibrated

PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "independent_cnn_baseline_convnext_base_optuna_gamma0_seed42"
OBJECTIVE = {"name": OBJECTIVE_NAME, "gamma": 0.0, "direction": "maximize",
             "adaptive_decisions": "gamma_zero_only", "version": 1}
R0 = {"lr_backbone": 5e-5, "weight_decay": .05, "head_dropout": .2}
TARGETS = ("crop", "disease")


def open_study(root, identity):
    storage = optuna.storages.RDBStorage("sqlite:///" + (Path(root) / "study.sqlite3").as_posix(),
        engine_kwargs={"connect_args": {"timeout": 60}, "poolclass": NullPool})
    study = optuna.create_study(study_name="convnext_base_gamma0_seed42", storage=storage,
        direction="maximize", load_if_exists=True, sampler=sampler_for(0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=6,
                                         interval_steps=3, n_min_trials=1))
    for key, value in {"study_identity": identity, "objective": OBJECTIVE}.items():
        saved = study.user_attrs.get(key)
        if saved is not None and saved != value:
            raise ValueError("SQLite study/objective identity changed")
        study.set_user_attr(key, value)
    if not study.trials:
        study.enqueue_trial(R0, user_attrs={"fresh_R0": True, "objective": OBJECTIVE_NAME})
    first = study.trials[0]
    if not first.user_attrs.get("fresh_R0") or first.user_attrs.get("incumbent"):
        raise ValueError("First trial must be a fresh R0, not an imported calibrated observation")
    return study


def trial_config(base, params, number, root):
    from copy import deepcopy
    cfg = deepcopy(base)
    cfg["optimizer"].update(lr_backbone=params["lr_backbone"], lr_head=10 * params["lr_backbone"],
                            weight_decay=params["weight_decay"])
    cfg["training"]["head_dropout"] = params["head_dropout"]
    cfg["model"]["head_dropout"] = params["head_dropout"]
    cfg["selection_objective"] = dict(OBJECTIVE)
    cfg["version"] = f"convnext_base_optuna_gamma0_trial_{number:03d}_seed42"
    cfg["output_dir"] = (Path(root) / "trials" / f"trial_{number:03d}").relative_to(PROJECT).as_posix()
    return cfg


def plot_history(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    history = read_json(folder / "history.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for target in TARGETS:
        h = read_json(folder / target / "history.json")
        epochs = [r["epoch"] for r in h]
        axes[0].plot(epochs, [r["train_top1"] for r in h], label=target + " fit")
        axes[0].plot(epochs, [r["validation_top1"] for r in h], label=target + " val")
        axes[1].plot(epochs, [r["validation_loss"] for r in h], label=target + " val CE")
    for metric, label in (("harmonic_mean_top1", "gamma-zero validation H1"),
                          ("harmonic_mean_top5", "gamma-zero validation H5")):
        axes[2].plot([r["epoch"] for r in history], [r[metric] for r in history], label=label)
    for ax in axes:
        ax.set_xlabel("Paired epoch")
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Raw independent factor Top-1")
    axes[1].set_ylabel("Validation CE")
    axes[2].set_ylabel("Uncalibrated composition harmonic mean")
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=130)
    plt.close(fig)


def verify_terminal(folder, identity, frames, pairs):
    folder = Path(folder)
    result = read_json(folder / "terminal.json")
    if result["identity"] != identity or result["objective"] != OBJECTIVE or result["outcome"] not in ("COMPLETE", "PRUNED"):
        raise ValueError("Terminal trial/objective identity changed")
    for name, digest in result["artifact_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f"Terminal artifact changed: {name}")
    best = result["best"]
    for target in TARGETS:
        saved = torch.load(folder / best["weights"][target], map_location="cpu", weights_only=True)
        if saved["fingerprint"] != identity or saved["epoch"] != best["epoch"] or saved["target"] != target:
            raise ValueError("Selected gamma-zero paired checkpoint mismatch")
        if any(not bool(torch.isfinite(v).all()) for v in saved["model_state"].values() if v.is_floating_point()):
            raise ValueError("Non-finite selected model weights")
    with np.load(folder / best["logits"]) as logits:
        actual = validation_uncalibrated(logits["crop"], logits["disease"], frames, pairs)
    if actual != {key: best["validation"][key] for key in actual}:
        raise ValueError("Gamma-zero checkpoint logits/objective disagree")
    pair = read_json(folder / "pair_state.json")
    if pair["best"] != best or len(pair["history"]) != result["completed_paired_epochs"]:
        raise ValueError("Terminal paired history mismatch")
    if any(r["objective_name"] != OBJECTIVE_NAME or r["gamma"] != 0. for r in pair["history"]):
        raise ValueError("Nonzero gamma or wrong objective in history")
    if max(pair["history"], key=lambda r: r["harmonic_mean_top1"])["epoch"] != best["epoch"]:
        raise ValueError("Gamma-zero epoch selection/tie rule mismatch")
    return result


def run_trial(trial, cfg, folder, study_identity, frames, pairs, cache, root,
              deadline, update, model_factory=IndependentNet):
    if cfg.get("selection_objective") != OBJECTIVE:
        raise ValueError("Every trial must declare the gamma-zero objective")
    folder = Path(folder)
    identity = fingerprint({"config": cfg, "study": study_identity})
    immutable_json(folder / "config.json", cfg)
    immutable_json(folder / "identity.json", {"identity": identity, "study_identity": study_identity,
        "trial_number": trial.number, "params": trial.params, "objective": OBJECTIVE})
    if (folder / "terminal.json").exists():
        return verify_terminal(folder, identity, frames, pairs)
    pair = read_json(folder / "pair_state.json") if (folder / "pair_state.json").exists() else {"history": [], "best": None}
    history, best = pair["history"], pair["best"]
    if any(r["gamma"] != 0. or r["objective_name"] != OBJECTIVE_NAME for r in history):
        raise ValueError("Cannot resume history from another objective")
    atomic_json(folder / "history.json", history)
    report_history(trial, history)
    actors = {}
    try:
        for target in TARGETS:
            actors[target] = Actor(target, cfg, folder / target, identity, model_factory)
            atomic_json(folder / target / "history.json", actors[target].history)
        if any(a.epoch < len(history) or a.epoch > len(history) + 1 for a in actors.values()):
            raise ValueError("Paired checkpoint/history epoch mismatch")
        val = pd.concat([frames["validation_seen"], frames["validation_unseen"]], ignore_index=True)
        while True:
            if history:
                epoch = history[-1]["epoch"]
                if epoch == cfg["training"]["max_epochs"]:
                    outcome, reason = "COMPLETE", "max_epochs"
                    break
                if epoch - best["epoch"] >= cfg["training"]["patience"]:
                    outcome, reason = "COMPLETE", "early_stopping_gamma0"
                    break
                if epoch >= 6 and (epoch - 6) % 3 == 0 and trial.should_prune():
                    outcome, reason = "PRUNED", "median_pruner_gamma0"
                    break
            epoch = len(history) + 1
            for target, actor in actors.items():
                if actor.epoch < epoch:
                    boundary(root, deadline)
                    update(phase="selection", active={"trial": trial.number, "target": target,
                                                       "epoch": epoch, "params": trial.params, "gamma": 0.0})
                    actor.epoch_step(frames["fit"], val, cache)
                if actor.epoch != epoch:
                    raise ValueError("Mismatched paired recovery epochs")
            row = validation_uncalibrated(actors["crop"].val_logits, actors["disease"].val_logits, frames, pairs)
            row["epoch"] = epoch
            if best is None or row["harmonic_mean_top1"] > best["validation"]["harmonic_mean_top1"]:
                weights = {t: f"{t}/best_epoch_{epoch:02d}.pt" for t in TARGETS}
                for target, actor in actors.items():
                    actor.save_weights(folder / weights[target])
                logits_name = f"validation_epoch_{epoch:02d}.npz"
                np.savez_compressed(folder / logits_name, **{t: a.val_logits.numpy() for t, a in actors.items()})
                best = {"epoch": epoch, "validation": row, "weights": weights, "logits": logits_name}
            history.append(row)
            atomic_json(folder / "pair_state.json", {"history": history, "best": best})
            atomic_json(folder / "history.json", history)
            report_history(trial, history)
            print(f"Trial {trial.number} pair {epoch}: uncalibrated validation H={row['harmonic_mean_top1']:.4%}, "
                  f"seen={row['seen_top1']:.2%}, pseudo-unseen={row['unseen_top1']:.2%}, gamma=0", flush=True)
            retained = set(best["weights"].values()) | {best["logits"]}
            for old in [*folder.glob("*/best_epoch_*.pt"), *folder.glob("validation_epoch_*.npz")]:
                if old.relative_to(folder).as_posix() not in retained:
                    old.unlink()
        plot_history(folder)
        names = ["config.json", "identity.json", "pair_state.json", "history.json", "learning_curves.png",
                 "crop/history.json", "disease/history.json", *best["weights"].values(), best["logits"]]
        result = {"trial_number": trial.number, "identity": identity, "objective": OBJECTIVE,
            "params": trial.params, "outcome": outcome, "reason": reason, "best": best,
            "completed_at_utc": now(), "completed_paired_epochs": len(history),
            "training_epoch_seconds": sum(r["seconds"] for a in actors.values() for r in a.history),
            "artifact_sha256": {name: sha256(folder / name) for name in names}}
        atomic_json(folder / "terminal.json", result)
        checked = verify_terminal(folder, identity, frames, pairs)
        atomic_json(folder / "verification.json", {"verified_at_utc": now(), "objective": OBJECTIVE,
                                                   "terminal_sha256": sha256(folder / "terminal.json")})
        return checked
    finally:
        actors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_spec(spec):
    if spec["objective"] != OBJECTIVE or spec["seed"] != 42 or spec["search_space"] != SPACE:
        raise ValueError("Wrong gamma-zero objective, seed or search space")
    if spec["window_hours"] != 10 or spec["max_trials"] != 8 or spec["history_import"] != "none":
        raise ValueError("Unapproved budget or history import")
    if spec["baseline_trial"] != "fresh_R0_first":
        raise ValueError("The reference must be trained freshly under gamma zero")
    root = (PROJECT / spec["output_dir"]).resolve()
    if root != PROJECT / "runs/baseline" / RUN_NAME:
        raise ValueError("Gamma-zero study must use its separate output folder")
    return root


def audit_inputs(spec, root, smoke=False):
    from dpd.data import COLUMNS
    require_cuda(spec["base_config"])
    if optuna.__version__ != spec["optuna_version"]:
        raise ValueError("Pinned Optuna version changed")
    source = PROJECT / spec["source_root"]
    for name, digest in spec["source_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Pinned source changed: {name}")
    source_config = read_json(source / spec["reference_folder"] / "complete.json")["config"]
    if source_config != spec["base_config"]:
        raise ValueError("Frozen R0 training controls changed")
    if sha256(PROJECT / source_config["model"]["pretrained_file"]) != source_config["model"]["pretrained_sha256"]:
        raise ValueError("Pretrained initialization changed")
    cache_pin = PROJECT / spec["cache_identity_file"]
    if sha256(cache_pin) != spec["cache_identity_sha256"]:
        raise ValueError("Audited cache identity changed")
    old_identity = read_json(cache_pin)
    # Use only its input identities; import no calibrated objective/trial observations.
    for name, digest in old_identity["code_sha256"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"Pinned training/preprocessing helper changed: {name}")
    frames = {name: pd.read_csv(source / f"{name}.csv") for name in
              ("fit", "validation_seen", "validation_unseen", "selection_excluded", "all_train")}
    if {name: len(frame) for name, frame in frames.items()} != {
            "fit": 5798, "validation_seen": 862, "validation_unseen": 533, "selection_excluded": 807, "all_train": 8000}:
        raise ValueError("Saved partition counts changed")
    data_root = (PROJECT / spec["data_root"]).resolve()
    provenance = read_json(source / "data_provenance.json")
    if sha256(data_root / "DPD_Small_train.csv") != provenance["metadata"]["train"]["sha256"]:
        raise ValueError("Official training metadata changed")
    train = pd.read_csv(data_root / "DPD_Small_train.csv", header=None, names=COLUMNS)
    if train[COLUMNS].to_dict("records") != frames["all_train"][COLUMNS].to_dict("records"):
        raise ValueError("Official training order/labels changed")
    seen_files = set()
    for name in ("fit", "validation_seen", "validation_unseen", "selection_excluded"):
        files = set(frames[name].file)
        if files & seen_files:
            raise ValueError("Saved partitions overlap")
        seen_files |= files
    if seen_files != set(train.file) or set(frames["fit"].pair_id) & set(frames["validation_unseen"].pair_id):
        raise ValueError("Saved partition membership changed")
    pairs = pd.DataFrame(read_json(source / "pairs.json"))
    if pairs.pair_id.tolist() != list(range(333)) or int(pairs.is_unseen.sum()) != 10:
        raise ValueError("Composition ontology changed")
    if set(train.pair_id) & set(pairs.loc[pairs.is_unseen, "pair_id"]):
        raise ValueError("Real-unseen training leakage")
    cache = PROJECT / "runs/cache/resized_256"
    print("Auditing 8,000 training originals and 7,193 pinned cache images for gamma-zero selection...", flush=True)

    def check_file(filename):
        if sha256(data_root / "DPD_Small_dataset" / filename) != provenance["image_hashes"][filename]:
            raise ValueError(f"Original training image changed: {filename}")
        digest = old_identity["cache_sha256"][filename]
        if digest is not None and sha256(cache / (Path(filename).stem + ".jpg")) != digest:
            raise ValueError(f"Preprocessing cache changed: {filename}")
        return filename, digest

    with ThreadPoolExecutor(max_workers=4) as pool:
        cache_hashes = dict(pool.map(check_file, train.file))
    selected_files = set(pd.concat([frames[k] for k in ("fit", "validation_seen", "validation_unseen")]).file)
    if {f for f, h in cache_hashes.items() if h is not None} != selected_files:
        raise ValueError("Pinned selection cache membership changed")
    names = ["independent_cnn_baseline/optuna_gamma0.py", "independent_cnn_baseline/uncalibrated_objective.py",
             "independent_cnn_baseline/run_optuna_gamma0.ps1", *old_identity["code_sha256"].keys()]
    env = environment()
    env["optuna"] = optuna.__version__
    identity_data = {"spec": spec, "cache_sha256": cache_hashes,
        "code_sha256": {name: sha256(PROJECT / name) for name in names},
        "packages": {**env["packages"], "python": env["python"], "torch": env["torch"], "optuna": env["optuna"]}}
    immutable_json(root / "identity.json", identity_data)
    identity = fingerprint(identity_data)
    atomic_json(root / "environment.json", env)
    atomic_json(root / "audit.json", {"identity": identity, "audited_at_utc": now(), "objective": OBJECTIVE,
        "training_images_verified": 8000, "selection_cache_images_verified": 7193,
        "saved_partition_reused": True, "historical_trial_observations_imported": 0,
        "official_test_inference": False, "code_sha256": identity_data["code_sha256"]})
    if smoke:
        atomic_json(root / "cuda_preflight.json", _smoke(source_config, 4))
    return identity, frames, pairs, cache


def snapshot(study, root, state):
    trials = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name, "params": trial.params,
               "value": trial.value, "fresh_R0": bool(trial.user_attrs.get("fresh_R0")),
               "paired_epochs": len(trial.intermediate_values)}
        path = Path(root) / "trials" / f"trial_{trial.number:03d}" / "terminal.json"
        if path.exists():
            terminal = read_json(path)
            if terminal["objective"] != OBJECTIVE or terminal["best"]["validation"]["gamma"] != 0.:
                raise ValueError("Snapshot cannot mix objectives")
            row.update(selected_epoch=terminal["best"]["epoch"], gamma=0.0,
                validation=terminal["best"]["validation"], training_epoch_seconds=terminal["training_epoch_seconds"],
                terminal=path.relative_to(root).as_posix())
        trials.append(row)
    completed = [row for row in trials if row["state"] == "COMPLETE"]
    best = max(completed, key=lambda row: row["value"]) if completed else None
    summary = {"state": state, "updated_at_utc": now(), "objective": OBJECTIVE, "trials": trials,
        "best_eligible": best, "historical_observations_imported": 0, "official_test_inference": False,
        "automatic_final_refit": False, "search_complete": state == "selection_complete"}
    atomic_json(Path(root) / "study_summary.json", summary)
    if state == "selection_complete":
        if best is None:
            raise ValueError("No completed gamma-zero recipe can be selected")
        immutable_json(Path(root) / "selected_recipe.json", {"objective": OBJECTIVE, **best})
    lines = ["# ConvNeXt-Base gamma-zero validation selection", "", f"State: {state}.", "",
        "Gamma is fixed at zero for checkpoints, early stopping, pruning, TPE and recipe ranking.",
        "Fixed split: fit 5,798 / seen validation 862 / pseudo-unseen validation 533; 323 candidates.",
        "No official tests or final refits run here. No calibrated observations are imported.", "",
        "| Trial | State | Uncalibrated validation H1 | Paired epochs |", "|---|---|---:|---:|"]
    for row in trials:
        value = f"{row['value']:.4%}" if row["value"] is not None else "pending"
        lines.append(f"| {row['number']}{' (fresh R0)' if row['fresh_R0'] else ''} | {row['state']} | {value} | {row['paired_epochs']} |")
    lines += ["", f"Best completed: trial {best['number']} / H0 {best['value']:.4%}." if best else "No new completed recipe yet.",
        "Only verified COMPLETE trials can win. Pruned or interrupted trials are ineligible."]
    (Path(root) / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(spec_path, resume=False, prepare=False, hours=None, deadline_utc=None, clear_stop=False):
    spec = read_json(spec_path)
    root = validate_spec(spec)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec["base_config"]["runtime"]["torch_threads"])
    lock_roots = [root, PROJECT / spec["source_root"],
        PROJECT / "runs/baseline/independent_cnn_baseline_r0_finalization_seed42",
        PROJECT / "runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42",
        PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42"]
    with contextlib.ExitStack() as stack:
        for folder in lock_roots:
            stack.enter_context(StudyLock(folder / "runner.lock"))
        state_path = root / "status.json"
        state = read_json(state_path) if state_path.exists() else {}
        if state.get("state") == "selection_complete":
            print("Gamma-zero selection is already complete; no new training launched.", flush=True)
            return
        if state.get("window_started_at_utc") and not resume:
            raise ValueError("Existing gamma-zero window requires --resume")
        if clear_stop:
            if not resume:
                raise ValueError("--clear-stop requires explicit --resume")
            (root / "STOP.request").unlink(missing_ok=True)
        if (root / "STOP.request").exists() or (state.get("state") == "paused" and not clear_stop):
            print("User pause preserved; explicit resume/clear-stop is required.", flush=True)
            return
        if resume and hours is not None:
            raise ValueError("Recovery must retain its saved deadline, not open a new budget")
        if hours is not None and (not np.isfinite(hours) or not 0 < hours <= 10):
            raise ValueError("Window exceeds approved ten-hour ceiling")
        if deadline_utc and not resume:
            raise ValueError("Recovery deadline requires --resume")
        immutable_json(root / "study_config.json", spec)
        (root / "runner.pid").write_text(str(os.getpid()), encoding="ascii")
        log = stack.enter_context((root / "training.log").open("a", encoding="utf-8", buffering=1))
        stack.enter_context(contextlib.redirect_stdout(Tee(sys.stdout, log)))
        stack.enter_context(contextlib.redirect_stderr(Tee(sys.stderr, log)))
        stack.enter_context(keep_awake())

        def update(**fields):
            state.update(fields, pid=os.getpid(), updated_at_utc=now(), objective=OBJECTIVE)
            atomic_json(state_path, state)

        study = None
        try:
            update(state="auditing", phase="audit")
            identity, frames, pairs, cache = audit_inputs(spec, root, smoke=not resume)
            if prepare:
                update(state="prepared", phase="prepared", audit_identity=identity)
                return
            saved_deadline = deadline_utc or (state.get("window_deadline_at_utc") if resume else None)
            if resume and not saved_deadline:
                raise ValueError("Recovery requires an existing window deadline")
            deadline, fields = window_timing(state, hours or 10, saved_deadline)
            update(state="running", **fields, audit_identity=identity, error=None, max_trials=8,
                   historical_observations_imported=0, official_test_inference=False, automatic_final_refit=False)
            study = open_study(root, identity)
            snapshot(study, root, "running")
            while True:
                boundary(root, deadline)
                trial = request_trial(study, 8)
                if trial is None:
                    snapshot(study, root, "selection_complete")
                    update(state="selection_complete", phase="completed", active=None, finished_at_utc=now())
                    return
                cfg = trial_config(spec["base_config"], trial.params, trial.number, root)
                folder = PROJECT / cfg["output_dir"]
                if shutil.disk_usage(root).free < 8 * 2**30:
                    raise RuntimeError("Less than 8 GiB free; preserve checkpoints and free space before recovery")
                print(f"Starting/resuming gamma-zero trial {trial.number}: {trial.params}", flush=True)
                result = run_trial(trial, cfg, folder, identity, frames, pairs, cache, root, deadline, update)
                terminal_state = optuna.trial.TrialState[result["outcome"]]
                value = result["best"]["validation"]["harmonic_mean_top1"] if terminal_state == optuna.trial.TrialState.COMPLETE else None
                study.tell(trial, value, state=terminal_state)
                retire_recovery(folder, root)
                snapshot(study, root, "running")
        except SearchStop as stopped:
            if study is not None:
                snapshot(study, root, stopped.state)
            update(state=stopped.state, phase="checkpointed", stopped_at_utc=now())
            print(f"{stopped.state}: gamma-zero work checkpointed; no new window or final refit.", flush=True)
        except BaseException:
            update(state="failed", error=traceback.format_exc())
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="independent_cnn_baseline/configs/optuna_convnext_base_gamma0_seed42.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    run(args.spec, args.resume, args.prepare, args.hours, args.deadline_utc, args.clear_stop)


if __name__ == "__main__":
    main()
