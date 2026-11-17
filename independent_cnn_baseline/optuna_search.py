"""Bounded, validation-only ConvNeXt-Base search. Never refit or infer official tests."""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import io
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HOME", str(PROJECT / "runs/cache/independent_cnn_baseline/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "runs/cache/independent_cnn_baseline/matplotlib"))
os.environ.setdefault("CUDA_CACHE_PATH", str(PROJECT / "runs/cache/independent_cnn_baseline/cuda"))

import numpy as np
import optuna
import pandas as pd
import torch
from sqlalchemy.pool import NullPool

from .engine import (Actor, StudyLock, Tee, atomic_json, fingerprint, keep_awake,
                     now, plot_history, read_json, select_gamma, sha256)
from .models import IndependentNet
from .prepare import _smoke, environment, require_cuda
from .runner import window_timing

TARGETS = ("crop", "disease")
SPACE = {"lr_backbone": [1e-5, 1.5e-4], "weight_decay": [.005, .1],
         "head_dropout": [0., .1, .2, .3, .4]}


class SearchStop(Exception):
    def __init__(self, state):
        self.state = state


def boundary(root, deadline):
    if (Path(root) / "STOP.request").exists():
        raise SearchStop("paused")
    if time.monotonic() >= deadline:
        raise SearchStop("window_complete")


def immutable_json(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable identity changed: {path}")
    else:
        atomic_json(path, value)


def distributions():
    return {"lr_backbone": optuna.distributions.FloatDistribution(*SPACE["lr_backbone"], log=True),
            "weight_decay": optuna.distributions.FloatDistribution(*SPACE["weight_decay"], log=True),
            "head_dropout": optuna.distributions.CategoricalDistribution(SPACE["head_dropout"])}


def sampler_for(number):
    # Each parameter has a reproducible stream even if a process exits between
    # SQLite suggestion commits. Actual model/loader seeds remain 42 for both heads.
    return ParameterSeededTPE(number)


class ParameterSeededTPE(optuna.samplers.TPESampler):
    """Default independent TPE, with a deterministic seed per trial/parameter."""
    def __init__(self, number):
        super().__init__(seed=42, n_startup_trials=3, multivariate=False, constant_liar=False)
        self.number = number

    def sample_independent(self, study, trial, param_name, param_distribution):
        index = list(SPACE).index(param_name)
        seed = int(np.random.SeedSequence([42, trial.number, index]).generate_state(1)[0])
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials(deepcopy=False))
        sampler = (optuna.samplers.RandomSampler(seed=seed) if completed < 3 else
                   optuna.samplers.TPESampler(seed=seed, n_startup_trials=3,
                                              multivariate=False, constant_liar=False))
        return sampler.sample_independent(study, trial, param_name, param_distribution)


def open_study(root, identity):
    storage = optuna.storages.RDBStorage("sqlite:///" + (Path(root) / "study.sqlite3").as_posix(),
                                        engine_kwargs={"connect_args": {"timeout": 60}, "poolclass": NullPool})
    study = optuna.create_study(study_name="convnext_base_seed42", storage=storage,
                                direction="maximize", load_if_exists=True, sampler=sampler_for(0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2,
                                    n_warmup_steps=6, interval_steps=3, n_min_trials=1))
    saved = study.user_attrs.get("study_identity")
    if saved is not None and saved != identity:
        raise ValueError("SQLite study identity mismatch")
    study.set_user_attr("study_identity", identity)
    return study


def request_trial(study, max_new):
    existing = study.get_trials(deepcopy=False)
    running = [t for t in existing if t.state == optuna.trial.TrialState.RUNNING]
    if len(running) > 1:
        raise ValueError("Multiple RUNNING trials; refusing a duplicate controller")
    if running:
        frozen = running[0]
        study.sampler = sampler_for(frozen.number)
        trial = optuna.trial.Trial(study, frozen._trial_id)
        # Reconstruct missing suggestions with exactly the same per-parameter RNG.
        for name, dist in distributions().items():
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                trial.suggest_categorical(name, dist.choices)
            else:
                trial.suggest_float(name, dist.low, dist.high, log=dist.log)
        return trial
    if sum(not t.user_attrs.get("incumbent", False) for t in existing) >= max_new:
        return None
    study.sampler = sampler_for(len(existing))
    return study.ask(fixed_distributions=distributions())


def trial_config(base, params, number, root):
    cfg = copy.deepcopy(base)
    cfg["optimizer"].update(lr_backbone=params["lr_backbone"], lr_head=10 * params["lr_backbone"],
                            weight_decay=params["weight_decay"])
    cfg["training"]["head_dropout"] = params["head_dropout"]
    cfg["model"]["head_dropout"] = params["head_dropout"]
    cfg["version"] = f"convnext_base_optuna_trial_{number:03d}_seed42"
    cfg["output_dir"] = str((Path(root) / "trials" / f"trial_{number:03d}").relative_to(PROJECT)).replace("\\", "/")
    return cfg


def validation(crop, disease, frames, pairs, gammas):
    from dpd.cnn import compose_pair_scores
    from dpd.evaluate import evaluate
    val = pd.concat([frames["validation_seen"], frames["validation_unseen"]], ignore_index=True)
    pseudo = set(frames["validation_unseen"].pair_id)
    val_pairs = pairs.assign(is_unseen=pairs.pair_id.isin(pseudo) | pairs.is_unseen)
    scores = compose_pair_scores(torch.log_softmax(torch.as_tensor(crop), 1).numpy(),
                                 torch.log_softmax(torch.as_tensor(disease), 1).numpy(), pairs)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation scores")
    allowed = ~pairs.is_unseen.to_numpy()
    seen = len(frames["validation_seen"])
    labels = val.pair_id.to_numpy()
    row = select_gamma(scores, labels, val_pairs.is_unseen.to_numpy(), allowed, seen, gammas)
    scores[:, ~allowed] = -np.inf
    result = evaluate(scores[:seen], labels[:seen], scores[seen:], labels[seen:], val_pairs, row["gamma"])
    if abs(result["harmonic_mean_top1"] - row["harmonic_mean_top1"]) > 1e-12:
        raise ValueError("Validation objective disagrees with shared evaluator")
    row.update(seen_top5=result["test_seen"]["top5"], unseen_top5=result["test_unseen"]["top5"],
               harmonic_mean_top5=result["harmonic_mean_top5"])
    return row


def verify_terminal(folder, identity, frames, pairs, gammas):
    folder = Path(folder)
    result = read_json(folder / "terminal.json")
    if result["identity"] != identity or result["outcome"] not in ("COMPLETE", "PRUNED"):
        raise ValueError("Terminal trial identity/state changed")
    for name, digest in result["artifact_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:
            raise ValueError(f"Terminal trial artifact changed: {name}")
    best = result["best"]
    for target in TARGETS:
        saved = torch.load(folder / best["weights"][target], map_location="cpu", weights_only=True)
        if saved["fingerprint"] != identity or saved["epoch"] != best["epoch"] or saved["target"] != target:
            raise ValueError("Selected paired checkpoint mismatch")
        if any(not bool(torch.isfinite(v).all()) for v in saved["model_state"].values() if v.is_floating_point()):
            raise ValueError("Non-finite selected weights")
    with np.load(folder / best["logits"]) as logits:
        actual = validation(logits["crop"], logits["disease"], frames, pairs, gammas)
    if any(abs(actual[k] - best["validation"][k]) > 1e-12 for k in actual):
        raise ValueError("Saved best validation logits/objective mismatch")
    pair_state = read_json(folder / "pair_state.json")
    if pair_state["best"] != best or len(pair_state["history"]) != result["completed_paired_epochs"]:
        raise ValueError("Terminal history mismatch")
    chosen = max(pair_state["history"], key=lambda r: r["harmonic_mean_top1"])
    if chosen["epoch"] != best["epoch"]:
        raise ValueError("Best epoch/tie rule mismatch")
    return result


def report_history(trial, history):
    stored = trial.study.trials[trial.number].intermediate_values
    best = 0.
    for row in history:
        best = max(best, row["harmonic_mean_top1"])
        if row["epoch"] in stored:
            if stored[row["epoch"]] != best:
                raise ValueError("SQLite intermediate objective changed")
        else:
            trial.report(best, row["epoch"])


def run_trial(trial, cfg, folder, study_identity, frames, pairs, cache, root,
              deadline, update, model_factory=IndependentNet):
    folder = Path(folder)
    identity = fingerprint({"config": cfg, "study": study_identity})
    immutable_json(folder / "config.json", cfg)
    immutable_json(folder / "identity.json", {"identity": identity, "study_identity": study_identity,
                                              "trial_number": trial.number, "params": trial.params})
    gammas = np.round(np.arange(0, 10, .1), 6)
    if (folder / "terminal.json").exists():
        return verify_terminal(folder, identity, frames, pairs, gammas)
    pair_state = read_json(folder / "pair_state.json") if (folder / "pair_state.json").exists() else {"history": [], "best": None}
    history, best = pair_state["history"], pair_state["best"]
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
        outcome, reason = None, None
        while True:
            if history:
                epoch = history[-1]["epoch"]
                if epoch == cfg["training"]["max_epochs"]:
                    outcome, reason = "COMPLETE", "max_epochs"
                elif epoch - best["epoch"] >= cfg["training"]["patience"]:
                    outcome, reason = "COMPLETE", "early_stopping"
                elif epoch >= 6 and (epoch - 6) % 3 == 0 and trial.should_prune():
                    outcome, reason = "PRUNED", "median_pruner"
                if outcome:
                    break
            epoch = len(history) + 1
            for target, actor in actors.items():
                if actor.epoch < epoch:
                    boundary(root, deadline)
                    update(phase="selection", active={"trial": trial.number, "target": target, "epoch": epoch,
                                                       "params": trial.params})
                    actor.epoch_step(frames["fit"], val, cache)
                if actor.epoch != epoch:
                    raise ValueError("Mismatched target checkpoint epochs")
            row = validation(actors["crop"].val_logits, actors["disease"].val_logits, frames, pairs, gammas)
            row["epoch"] = epoch
            if best is None or row["harmonic_mean_top1"] > best["validation"]["harmonic_mean_top1"]:
                # Pair pointer commits only after both distinct epoch files and logits exist.
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
            print(f"Trial {trial.number} pair {epoch}: validation H={row['harmonic_mean_top1']:.4%}, "
                  f"gamma={row['gamma']:g}, gamma-zero H={row['gamma0_harmonic_mean_top1']:.4%}", flush=True)
            # Delete superseded selection-only weights after committing the new pointer.
            retained = set(best["weights"].values()) | {best["logits"]}
            for old in [*folder.glob("*/best_epoch_*.pt"), *folder.glob("validation_epoch_*.npz")]:
                if old.relative_to(folder).as_posix() not in retained:
                    old.unlink()
        plot_history(folder)
        artifact_names = ["config.json", "identity.json", "pair_state.json", "history.json", "learning_curves.png",
                          "crop/history.json", "disease/history.json", *best["weights"].values(), best["logits"]]
        result = {"trial_number": trial.number, "identity": identity, "params": trial.params,
                  "outcome": outcome, "reason": reason, "completed_at_utc": now(), "best": best,
                  "completed_paired_epochs": len(history),
                  "training_epoch_seconds": sum(r["seconds"] for a in actors.values() for r in a.history),
                  "artifact_sha256": {name: sha256(folder / name) for name in artifact_names}}
        atomic_json(folder / "terminal.json", result)
        checked = verify_terminal(folder, identity, frames, pairs, gammas)
        atomic_json(folder / "verification.json", {"verified_at_utc": now(), "terminal_sha256": sha256(folder / "terminal.json")})
        return checked
    finally:
        actors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def retire_recovery(folder, root):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    if not folder.is_relative_to(root / "trials") or not (folder / "verification.json").exists():
        raise ValueError("Recovery retention boundary violation")
    for target in TARGETS:
        (folder / target / "last.pt").unlink(missing_ok=True)


def audit(spec, root, smoke=False):
    """Read original selection inputs; never change historical artifacts or resplit."""
    from dpd.data import COLUMNS
    from dpd.resize_cache import _downscale
    from PIL import Image
    if optuna.__version__ != spec["optuna_version"] or spec["search_space"] != SPACE:
        raise ValueError("Optuna version/search-space mismatch")
    if spec["seed"] != 42 or spec["window_hours"] != 10 or spec["max_new_trials"] != 8:
        raise ValueError("Unapproved seed/budget")
    require_cuda(spec["base_config"])
    source = PROJECT / spec["source_root"]
    for name, digest in spec["source_sha256"].items():
        if sha256(source / name) != digest:
            raise ValueError(f"Pinned source changed: {name}")
    base = spec["base_config"]
    if base != read_json(source / spec["incumbent_folder"] / "complete.json")["config"]:
        raise ValueError("Frozen Base source config changed")
    if sha256(PROJECT / base["model"]["pretrained_file"]) != base["model"]["pretrained_sha256"]:
        raise ValueError("Pinned pretrained initialization changed")
    parts = {name: pd.read_csv(source / f"{name}.csv") for name in
             ("fit", "validation_seen", "validation_unseen", "selection_excluded", "all_train")}
    if {name: len(frame) for name, frame in parts.items()} != {
            "fit": 5798, "validation_seen": 862, "validation_unseen": 533, "selection_excluded": 807, "all_train": 8000}:
        raise ValueError("Frozen partition counts changed")
    data_root = (PROJECT / spec["data_root"]).resolve()
    provenance = read_json(source / "data_provenance.json")
    if sha256(data_root / "DPD_Small_train.csv") != provenance["metadata"]["train"]["sha256"]:
        raise ValueError("Official training metadata changed")
    train = pd.read_csv(data_root / "DPD_Small_train.csv", header=None, names=COLUMNS)
    if train[COLUMNS].to_dict("records") != parts["all_train"][COLUMNS].to_dict("records"):
        raise ValueError("Official all-train order/labels changed")
    used = set()
    for name in ("fit", "validation_seen", "validation_unseen", "selection_excluded"):
        files = set(parts[name].file)
        if files & used:
            raise ValueError("Overlapping saved partition files")
        used |= files
    if used != set(train.file) or set(parts["fit"].pair_id) & set(parts["validation_unseen"].pair_id):
        raise ValueError("Frozen partition membership mismatch")
    pairs = pd.DataFrame(read_json(source / "pairs.json"))
    if pairs.pair_id.tolist() != list(range(333)) or int(pairs.is_unseen.sum()) != 10:
        raise ValueError("Ontology changed")
    if set(train.pair_id) & set(pairs.loc[pairs.is_unseen, "pair_id"]):
        raise ValueError("Real-unseen training leakage")
    cache = PROJECT / "runs/cache/resized_256"
    selected_files = set(pd.concat([parts[k] for k in ("fit", "validation_seen", "validation_unseen")]).file)

    def check_file(filename):
        original = data_root / "DPD_Small_dataset" / filename
        if sha256(original) != provenance["image_hashes"][filename]:
            raise ValueError(f"Original training image changed: {filename}")
        if filename not in selected_files:
            return filename, None
        cached = cache / (Path(filename).stem + ".jpg")
        if not cached.exists():
            raise ValueError(f"Original Base cache missing: {cached}")
        expected = io.BytesIO()
        _downscale(original, 256).save(expected, format="JPEG", quality=95)
        with Image.open(io.BytesIO(expected.getvalue())) as a, Image.open(cached) as b:
            if not np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))):
                raise ValueError(f"Cached preprocessing differs from pinned pipeline: {filename}")
        return filename, sha256(cached)

    print("Auditing 8,000 original training images and 7,193 selection cache images...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(check_file, train.file))
    code_names = ["independent_cnn_baseline/optuna_search.py", "independent_cnn_baseline/engine.py",
                  "independent_cnn_baseline/models.py", "independent_cnn_baseline/prepare.py",
                  "independent_cnn_baseline/runner.py", "dpd/evaluate.py", "dpd/cnn.py", "dpd/resize_cache.py"]
    env = environment()
    env["optuna"] = optuna.__version__
    identity_data = {"spec": spec, "cache_sha256": hashes,
                     "code_sha256": {p: sha256(PROJECT / p) for p in code_names},
                     "packages": {**env["packages"], "torch": env["torch"], "python": env["python"], "optuna": optuna.__version__}}
    immutable_json(root / "identity.json", identity_data)
    atomic_json(root / "environment.json", env)
    identity = fingerprint(identity_data)
    atomic_json(root / "audit.json", {"identity": identity, "audited_at_utc": now(), "training_images_verified": 8000,
                "selection_cache_images_verified": len(selected_files), "saved_partition_reused": True,
                "official_test_inference": False, "code_sha256": identity_data["code_sha256"]})
    if smoke:
        atomic_json(root / "cuda_preflight.json", _smoke(base, 4))
    reference = read_json(source / spec["incumbent_folder"] / "complete.json")
    with np.load(source / spec["incumbent_folder"] / "best_validation_logits.npz") as logits:
        actual = validation(logits["crop"], logits["disease"], parts, pairs, np.round(np.arange(0, 10, .1), 6))
    if actual["gamma"] != 6.2 or abs(actual["harmonic_mean_top1"] - .5973571322087159) > 1e-12:
        raise ValueError("R0 validation reference changed")
    for target in TARGETS:
        path = source / spec["incumbent_folder"] / target / "best.pt"
        if sha256(path) != reference["checkpoint_sha256"][target]:
            raise ValueError("R0 selected checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["epoch"] != 15 or saved["fingerprint"] != reference["fingerprint"]:
            raise ValueError("R0 selected epoch/identity changed")
    atomic_json(root / "incumbent_verified.json", {"source": spec["incumbent_folder"], "validation": actual,
                                                "selected_epoch": 15, "verified_at_utc": now()})
    return identity, parts, pairs, cache, reference


def register_incumbent(study, reference):
    if study.trials:
        if not study.trials[0].user_attrs.get("incumbent") or study.trials[0].value != reference["validation"]["harmonic_mean_top1"]:
            raise ValueError("SQLite incumbent changed")
        return
    study.add_trial(optuna.trial.create_trial(params={"lr_backbone": 5e-5, "weight_decay": .05, "head_dropout": .2},
                    distributions=distributions(), value=reference["validation"]["harmonic_mean_top1"],
                    # No old intermediate curve is imported into pruning.
                    user_attrs={"incumbent": True, "selected_epoch": 15, "gamma": 6.2,
                                "source": reference["folder"]}))


def snapshot(study, root, reference, state):
    trials = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name, "params": trial.params,
               "value": trial.value, "incumbent": bool(trial.user_attrs.get("incumbent")),
               "paired_epochs": len(trial.intermediate_values)}
        marker = root / "trials" / f"trial_{trial.number:03d}" / "terminal.json"
        if marker.exists():
            terminal = read_json(marker)
            row.update(selected_epoch=terminal["best"]["epoch"], gamma=terminal["best"]["validation"]["gamma"],
                       training_epoch_seconds=terminal["training_epoch_seconds"], terminal=str(marker.relative_to(root)))
        trials.append(row)
    # max() retains the first on exact ties; only COMPLETE trials are eligible.
    winner = max((r for r in trials if r["state"] == "COMPLETE"), key=lambda r: r["value"])
    summary = {"state": state, "updated_at_utc": now(), "trials": trials, "best_eligible": winner,
               "incumbent_validation": reference["validation"], "official_test_inference": False,
               "automatic_final_refit": False, "search_complete": state == "selection_complete"}
    atomic_json(root / "study_summary.json", summary)
    if state == "selection_complete":
        immutable_json(root / "selected_recipe.json", winner)
    lines = ["# ConvNeXt-Base Optuna validation search", "", f"State: {state}.", "",
             "Fixed seed-42 split: 5,798 fitting / 862 seen validation / 533 pseudo-unseen validation.",
             "No official test inference or full-data refitting is performed by this controller.", "",
             "| Trial | State | Validation H | Paired epochs |", "|---|---|---:|---:|"]
    for row in trials:
        value = f"{row['value']:.4%}" if row["value"] is not None else "pending"
        lines.append(f"| {row['number']}{' (R0)' if row['incumbent'] else ''} | {row['state']} | {value} | {row['paired_epochs']} |")
    lines += ["", f"Best eligible: trial {winner['number']}, validation H {winner['value']:.4%}.",
              "Only verified completed trials and R0 can win. Pruned or interrupted trials are ineligible.",
              "Official test sets have been viewed previously in project development; this is iterative benchmark work."]
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(spec_path, resume=False, prepare=False, hours=None, deadline_utc=None, clear_stop=False):
    spec = read_json(spec_path)
    root = (PROJECT / spec["output_dir"]).resolve()
    if root != PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42":
        raise ValueError("Unexpected Optuna output folder")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec["base_config"]["runtime"]["torch_threads"])
    lock_roots = [root, PROJECT / spec["source_root"],
                  PROJECT / "runs/baseline/independent_cnn_baseline_r0_finalization_seed42",
                  PROJECT / "runs/baseline/independent_cnn_baseline_convnext_tiny_full_refit_seed42"]
    with contextlib.ExitStack() as stack:
        for lock_root in lock_roots:
            stack.enter_context(StudyLock(lock_root / "runner.lock"))
        state_file = root / "status.json"
        state = read_json(state_file) if state_file.exists() else {}
        if state.get("state") == "selection_complete":
            print("This bounded search is already complete; no new training launched.", flush=True)
            return
        if state.get("window_started_at_utc") and not resume:
            raise ValueError("Existing search requires --resume; no duplicate invocation")
        if clear_stop:
            if not resume:
                raise ValueError("--clear-stop requires explicit --resume")
            (root / "STOP.request").unlink(missing_ok=True)
        if (root / "STOP.request").exists() or (state.get("state") == "paused" and not clear_stop):
            print("User pause is preserved; explicit --resume --clear-stop required.", flush=True)
            return
        if resume and hours is not None:
            raise ValueError("Recovery must reuse saved deadline, not open a fresh budget")
        if hours is not None and (not np.isfinite(hours) or not 0 < hours <= spec["window_hours"]):
            raise ValueError("Window exceeds the approved ten-hour ceiling")
        if deadline_utc and not resume:
            raise ValueError("--deadline-utc is recovery-only")
        (root / "runner.pid").write_text(str(os.getpid()), encoding="ascii")
        immutable_json(root / "study_config.json", spec)
        log = stack.enter_context((root / "training.log").open("a", encoding="utf-8", buffering=1))
        stack.enter_context(contextlib.redirect_stdout(Tee(sys.stdout, log)))
        stack.enter_context(contextlib.redirect_stderr(Tee(sys.stderr, log)))
        stack.enter_context(keep_awake())

        def update(**fields):
            state.update(fields, pid=os.getpid(), updated_at_utc=now())
            atomic_json(state_file, state)

        study = None
        try:
            update(state="auditing", phase="audit")
            identity, frames, pairs, cache, reference = audit(spec, root, smoke=not resume)
            if prepare:
                update(state="prepared", phase="prepared", audit_identity=identity)
                return
            saved_deadline = deadline_utc or (state.get("window_deadline_at_utc") if resume else None)
            if resume and not saved_deadline:
                raise ValueError("Resume requires an existing saved window")
            deadline, fields = window_timing(state, hours or spec["window_hours"], saved_deadline)
            update(state="running", **fields, error=None, audit_identity=identity, official_test_inference=False,
                   max_new_trials=spec["max_new_trials"], automatic_final_refit=False)
            study = open_study(root, identity)
            register_incumbent(study, reference)
            snapshot(study, root, reference, "running")
            while True:
                boundary(root, deadline)
                trial = request_trial(study, spec["max_new_trials"])
                if trial is None:
                    snapshot(study, root, reference, "selection_complete")
                    update(state="selection_complete", phase="completed", active=None, finished_at_utc=now())
                    return
                cfg = trial_config(spec["base_config"], trial.params, trial.number, root)
                folder = PROJECT / cfg["output_dir"]
                print(f"Starting/resuming trial {trial.number}: {trial.params}", flush=True)
                if shutil.disk_usage(root).free < 8 * 2**30:
                    raise RuntimeError("Less than 8 GiB free; checkpoints preserved. Free space before recovery.")
                result = run_trial(trial, cfg, folder, identity, frames, pairs, cache, root, deadline, update)
                terminal_state = optuna.trial.TrialState[result["outcome"]]
                study.tell(trial, result["best"]["validation"]["harmonic_mean_top1"]
                           if terminal_state == optuna.trial.TrialState.COMPLETE else None, state=terminal_state)
                retire_recovery(folder, root)
                snapshot(study, root, reference, "running")
        except SearchStop as stopped:
            if study is not None:
                snapshot(study, root, reference, stopped.state)
            update(state=stopped.state, phase="checkpointed", stopped_at_utc=now())
            print(f"{stopped.state}: checkpoints retained; no new window or final refit launched.", flush=True)
        except BaseException:
            update(state="failed", error=traceback.format_exc())
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="independent_cnn_baseline/configs/optuna_convnext_base_seed42.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    run(args.spec, args.resume, args.prepare, args.hours, args.deadline_utc, args.clear_stop)


if __name__ == "__main__":
    main()
