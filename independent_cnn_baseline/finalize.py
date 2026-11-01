"""Refit the three frozen R0 recipes and evaluate; never launch selection trials."""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import sys
import time
import traceback
from pathlib import Path

from . import runner
from .engine import StudyLock, Tee, atomic_json, fingerprint, keep_awake, now, read_json, sha256
from .prepare import audit_data, environment, require_cuda

PROJECT = runner.PROJECT
EXPECTED = {
    "bit50": ("resnetv2_50x1_bit.goog_in21k_ft_in1k", 3, 6.6),
    "convnext_base": ("convnext_base.fb_in22k_ft_in1k", 15, 6.2),
    "efficientnetv2_l": ("tf_efficientnetv2_l.in21k_ft_in1k", 3, 6.6),
}


class PauseRequested(Exception):
    pass


def paths(spec):
    root = (PROJECT / spec["output_dir"]).resolve()
    source = (PROJECT / spec["source_study"]).resolve()
    base = (PROJECT / "runs/baseline").resolve()
    if not root.is_relative_to(base) or not root.name.startswith("independent_cnn_baseline_r0_"):
        raise ValueError("Refit-only outputs must use their own independent_cnn_baseline_r0 folder")
    if root == source or not source.is_relative_to(base):
        raise ValueError("The selection study must remain separate from finalization outputs")
    if spec["selection_policy"] != "completed_r0_only" or spec["seed"] != 42:
        raise ValueError("Expected the fixed seed-42 R0-only finalization plan")
    if len(spec["models"]) != 3 or {m["name"] for m in spec["models"]} != set(EXPECTED):
        raise ValueError("Expected exactly the three completed R0 models")
    return root, source


def verify_sources(spec, source, frames, audit):
    """Verify saved selections against the original data/partition and pinned hashes."""
    original_path = PROJECT / spec["source_spec"]
    if sha256(original_path) != spec["source_spec_sha256"]:
        raise ValueError("Original study recipe changed after the R0-only plan was pinned")
    original = read_json(original_path)
    if (source / "study_fingerprint.txt").read_text().strip() != fingerprint(original):
        raise ValueError("Original study fingerprint mismatch")
    source_audit = read_json(source / "audit.json")
    if audit["data_identity"] != source_audit["data_identity"]:
        raise ValueError("Dataset identity differs from the completed selection study")
    if audit["pseudo_unseen_pair_ids"] != source_audit["pseudo_unseen_pair_ids"]:
        raise ValueError("Validation composition partition changed")
    selected = {}
    for item in spec["models"]:
        name = item["name"]
        backbone, epoch, gamma = EXPECTED[name]
        expected_name = f"independent_cnn_baseline_{name}_r0_seed42"
        if item["trial"] != expected_name or item["selected_epoch"] != epoch or item["gamma"] != gamma:
            raise ValueError("The refit-only plan may only use the already selected R0 epochs/gammas")
        folder = source / "trials" / expected_name
        if sha256(folder / "complete.json") != item["complete_sha256"]:
            raise ValueError(f"Pinned R0 completion record changed: {name}")
        row = read_json(folder / "complete.json")
        config = row["config"]
        val_files = frames["validation_seen"].file.tolist() + frames["validation_unseen"].file.tolist()
        identity = fingerprint({"config": config, "data": audit["data_identity"],
                                "fit": frames["fit"].file.tolist(), "validation": val_files})
        checked = runner.candidate_complete(folder, identity)
        if checked is None or row["selected_epoch"] != epoch or row["validation"]["gamma"] != gamma:
            raise ValueError(f"R0 selection verification failed: {name}")
        if config["model"]["backbone"] != backbone or config["seed"] != 42:
            raise ValueError("Selected backbone/seed differs from the approved R0 model")
        if sha256(PROJECT / config["model"]["pretrained_file"]) != config["model"]["pretrained_sha256"]:
            raise ValueError("Pinned original pretrained weights changed")
        selected[name] = row
    return original, selected


def final_configs(selected):
    configs = {}
    for name, chosen in selected.items():
        config = copy.deepcopy(chosen["config"])
        config["version"] = f"independent_cnn_baseline_{name}_seed42"
        config["output_dir"] = "runs/baseline/" + config["version"]
        configs[name] = config
    return configs


def finalize(spec_path, resume=False, prepare_only=False, hours=None, deadline_utc=None):
    spec = read_json(spec_path)
    root, source = paths(spec)
    original_path = PROJECT / spec["source_spec"]
    if sha256(original_path) != spec["source_spec_sha256"]:
        raise ValueError("Pinned source recipe changed")
    original = read_json(original_path)
    os.environ["DPD_ROOT"] = str((PROJECT / original["data_root"]).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not prepare_only:
        require_cuda(original)
    if deadline_utc is not None and not resume:
        raise ValueError("Reusing a window deadline requires --resume")
    runner.torch.set_num_threads(original["runtime"]["torch_threads"])
    root.mkdir(parents=True, exist_ok=True)
    # Holding both locks prevents the old tuning controller and this finalizer
    # from producing the same final model folders concurrently.
    with StudyLock(source / "runner.lock"), StudyLock(root / "runner.lock"), keep_awake():
        previous = read_json(root / "status.json") if (root / "status.json").exists() else {}
        marker = root / "study_fingerprint.txt"
        identity = fingerprint(spec)
        if marker.exists() and marker.read_text().strip() != identity:
            raise ValueError("Refit-only plan changed; refusing to overwrite its outputs")
        if previous.get("state") == "completed":
            print(f"Already completed: {root / 'results.md'}", flush=True)
            return
        if prepare_only and previous.get("window_started_at_utc"):
            raise ValueError("Use --resume for a finalization that has already started")
        if previous.get("state") not in (None, "prepared") and not resume and not prepare_only:
            raise ValueError("Finalization already started; verify exit and use --resume")
        marker.write_text(identity + "\n", encoding="utf-8")
        stop = root / "STOP.request"
        if stop.exists():
            if not resume:
                raise ValueError("A user stop request is pending; explicit --resume is required")
            stop.replace(root / f"STOP.consumed_{time.time_ns()}.json")
        state = {**previous, "state": "preparing" if prepare_only else "running", "phase": "verification",
                 "pid": os.getpid(), "active": None, "selection_target_trainings_planned": 0,
                 "total_final_target_refits": 6, "total_final_target_epochs": 42}
        state.pop("error", None)

        def update(**values):
            if values.get("phase") in ("refit", "evaluation") and stop.exists():
                raise PauseRequested()
            state.update(values, updated_at_utc=now())
            atomic_json(root / "status.json", state)

        (root / "runner.pid").write_text(str(os.getpid()), encoding="utf-8")
        with (root / "training.log").open("a", encoding="utf-8", buffering=1) as log, \
             contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"R0-only finalization {now()}, PID {os.getpid()}; prepare_only={prepare_only}", flush=True)
                update()
                atomic_json(root / "finalization_config.json", spec)
                atomic_json(root / "environment.json", environment())
                atomic_json(root / "code_provenance.json", {str(p.relative_to(PROJECT)): sha256(p)
                            for p in (PROJECT / "independent_cnn_baseline").glob("*.py")})
                # Recheck all originals and reproduce the saved selection partition.
                df, pairs, frames, audit = audit_data(original, root)
                for name in ("fit", "validation_seen", "validation_unseen", "all_train", "selection_excluded"):
                    if sha256(root / f"{name}.csv") != sha256(source / f"{name}.csv"):
                        raise ValueError(f"Saved {name} partition differs from the source study")
                _, selected = verify_sources(spec, source, frames, audit)
                frozen = root / "selected_models.json"
                if frozen.exists() and read_json(frozen) != selected:
                    raise ValueError("Previously frozen R0 selections changed")
                atomic_json(frozen, selected)
                configs = final_configs(selected)
                atomic_json(root / "final_model_configs.json", configs)
                if prepare_only:
                    update(state="prepared", phase="ready", active=None,
                           prepared_at_utc=now(), official_test_evaluation_started=False)
                    print("Prepared: three verified R0 sources; six final targets / 42 target epochs. No training started.", flush=True)
                    return
                from dpd.resize_cache import ensure_resized
                cache = ensure_resized(df.file, original["input"]["resize_cache"])
                duration = spec["window_hours"] if hours is None else hours
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Window hours must be positive and finite")
                deadline, timing = runner.window_timing(previous, duration, deadline_utc)
                update(**timing)
                for name, config in configs.items():
                    runner.refit(config, selected[name], PROJECT / config["output_dir"], audit["data_identity"],
                                 frames["all_train"], cache, deadline, update)
                # All six refits must finish before any official test evaluation.
                results = {}
                for name, config in configs.items():
                    results[name] = runner.evaluate_final(config, selected[name], PROJECT / config["output_dir"],
                                                         df, pairs, cache, deadline, update)
                summary = {"completed_at_utc": now(), "scope": "three fixed R0 baselines; no additional tuning",
                           "selected_models": selected, "results": results, "audit": audit,
                           "official_tests_used_for_selection": False,
                           "final_training_epoch_seconds": sum(sum(r["seconds"] for r in read_json(
                               PROJECT / c["output_dir"] / f"{target}_refit_history.json"))
                               for c in configs.values() for target in ("crop", "disease"))}
                atomic_json(root / "study_summary.json", summary)
                lines = ["# Independent CNN Baseline: three fixed R0 recipes", "",
                         "Official tests use all 333 candidates. Gamma and epoch were frozen on validation.", "",
                         "| Backbone | Gamma | Epoch | Seen Top-1 | Unseen Top-1 | H Top-1 | H Top-5 | Gamma-zero H Top-1 |",
                         "|---|---:|---:|---:|---:|---:|---:|---:|"]
                for name, r in results.items():
                    lines.append(f"| {name} | {r['gamma']:g} | {r['selected_epoch']} | {r['test_seen']['top1']:.2%} | "
                                 f"{r['test_unseen']['top1']:.2%} | {r['harmonic_mean_top1']:.2%} | "
                                 f"{r['harmonic_mean_top5']:.2%} | {r['harmonic_mean_top1_gamma0']:.2%} |")
                (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
                update(state="completed", phase="completed", active=None, completed_at_utc=now())
            except PauseRequested:
                update(state="paused", phase="checkpointed", pause_reason="User stop request",
                       pause_verified_at_utc=now())
                print("User stop request reached: completed epoch is checkpointed; use --resume when ready.", flush=True)
            except runner.WindowEnd:
                update(state="window_complete", phase="checkpointed", window_finished_at_utc=now())
                print("Window ended; all completed target checkpoints are preserved.", flush=True)
            except BaseException:
                update(state="failed", error=traceback.format_exc())
                traceback.print_exc()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="independent_cnn_baseline/configs/refit_r0_seed42.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--request-stop", action="store_true", help="Checkpoint at the next target-epoch boundary, then pause")
    args = parser.parse_args()
    if args.request_stop:
        root, _ = paths(read_json(args.spec))
        atomic_json(root / "STOP.request", {"requested_at_utc": now(), "reason": "User requested a checkpointed pause"})
        print(f"Stop requested; the active target epoch will checkpoint before exit: {root}")
        return
    finalize(args.spec, args.resume, args.prepare, args.hours, args.deadline_utc)


if __name__ == "__main__":
    main()
