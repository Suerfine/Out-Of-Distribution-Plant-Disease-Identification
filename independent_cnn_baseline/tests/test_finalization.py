"""Refit-only scope, frozen-source integrity and cooperative-pause tests."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from independent_cnn_baseline import finalize as finalizer
from independent_cnn_baseline.engine import atomic_json, fingerprint, read_json, sha256

PROJECT = Path(__file__).resolve().parents[2]


def fixture(root):
    source = root / "runs/baseline/independent_cnn_baseline_study_seed42"
    source.mkdir(parents=True)
    original = read_json(PROJECT / "independent_cnn_baseline/configs/study_seed42.json")
    original_path = root / "original.json"
    atomic_json(original_path, original)
    (source / "study_fingerprint.txt").write_text(fingerprint(original))
    atomic_json(source / "status.json", {"state": "paused"})
    weights = root / "cached_original.safetensors"
    weights.write_bytes(b"test-only weight identity; never instantiated")
    frame = pd.DataFrame({"file": ["a", "b", "c", "d"], "pair_id": [0, 1, 0, 1],
                          "crop_id": [0, 1, 0, 1], "disease_id": [0, 1, 0, 1]})
    frames = {"fit": frame.iloc[:2], "validation_seen": frame.iloc[2:3],
              "validation_unseen": frame.iloc[3:], "all_train": frame}
    audit = {"data_identity": "fixture", "pseudo_unseen_pair_ids": [1]}
    atomic_json(source / "audit.json", audit)
    for name, part in {**frames, "selection_excluded": frame.iloc[:0]}.items():
        (source / f"{name}.csv").write_text(part.to_csv(index=False, lineterminator="\n"), newline="")
    spec = {"name": "independent_cnn_baseline_r0_finalization_seed42",
            "output_dir": "runs/baseline/independent_cnn_baseline_r0_finalization_seed42",
            "source_study": str(source.relative_to(root)), "source_spec": "original.json",
            "source_spec_sha256": sha256(original_path), "seed": 42,
            "selection_policy": "completed_r0_only", "window_hours": 10, "models": []}
    for name, (backbone, epoch, gamma) in finalizer.EXPECTED.items():
        version = f"independent_cnn_baseline_{name}_r0_seed42"
        folder = source / "trials" / version
        cfg = {k: copy.deepcopy(original[k]) for k in ("seed", "input", "augmentation", "optimizer", "training", "runtime")}
        cfg.update(version=version, model={"name": name, "backbone": backbone,
                   "pretrained_file": weights.name, "pretrained_sha256": sha256(weights)})
        identity = fingerprint({"config": cfg, "data": audit["data_identity"],
                                "fit": frames["fit"].file.tolist(), "validation": ["c", "d"]})
        hashes = {}
        for target in ("crop", "disease"):
            path = folder / target / "best.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((name + target).encode())
            hashes[target] = sha256(path)
        atomic_json(folder / "history.json", [{"epoch": e, "harmonic_mean_top1": e / 100} for e in range(1, epoch + 1)])
        row = {"name": version, "model_id": name, "config": cfg, "folder": str(folder.relative_to(root)),
               "fingerprint": identity, "selected_epoch": epoch, "completed_epochs": epoch,
               "validation": {"gamma": gamma, "harmonic_mean_top1": epoch / 100}, "checkpoint_sha256": hashes}
        atomic_json(folder / "complete.json", row)
        spec["models"].append({"name": name, "trial": version, "selected_epoch": epoch,
                               "gamma": gamma, "complete_sha256": sha256(folder / "complete.json")})
    spec_path = root / "refit.json"
    atomic_json(spec_path, spec)

    def audit_mock(_original, destination):
        for name in ("fit", "validation_seen", "validation_unseen", "all_train", "selection_excluded"):
            (destination / f"{name}.csv").write_bytes((source / f"{name}.csv").read_bytes())
        return frame, None, frames, audit

    return spec, spec_path, source, frames, audit, audit_mock


class FinalizationTests(unittest.TestCase):
    def test_preparation_freezes_only_three_r0s_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, path, source, _, _, audit = fixture(root)
            with patch.object(finalizer, "PROJECT", root), patch.object(finalizer, "environment", return_value={}), \
                 patch.object(finalizer, "audit_data", side_effect=audit), \
                 patch.object(finalizer.runner, "refit") as refit, patch.object(finalizer.runner, "evaluate_final") as evaluate, \
                 redirect_stdout(io.StringIO()):
                finalizer.finalize(path, prepare_only=True)
            refit.assert_not_called()
            evaluate.assert_not_called()
            destination = root / spec["output_dir"]
            selected = read_json(destination / "selected_models.json")
            self.assertEqual([r["selected_epoch"] for r in selected.values()], [3, 15, 3])
            self.assertEqual(read_json(destination / "status.json")["state"], "prepared")
            self.assertEqual(read_json(source / "status.json"), {"state": "paused"})

    def test_checkpoint_tampering_and_partial_r1_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, _, source, frames, audit, _ = fixture(root)
            with patch.object(finalizer, "PROJECT", root):
                finalizer.verify_sources(spec, source, frames, audit)
                changed = copy.deepcopy(spec)
                changed["models"][0]["trial"] = "independent_cnn_baseline_bit50_r1_seed42"
                with self.assertRaisesRegex(ValueError, "already selected R0"):
                    finalizer.verify_sources(changed, source, frames, audit)
                (source / "trials" / spec["models"][0]["trial"] / "crop/best.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    finalizer.verify_sources(spec, source, frames, audit)

    def test_all_refits_precede_tests_and_no_selection_training_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, path, source, _, _, audit = fixture(root)
            destination = root / spec["output_dir"]
            events = []

            def refit(config, chosen, folder, *_):
                self.assertEqual(len(read_json(destination / "selected_models.json")), 3)
                events.append(("refit", config["model"]["name"]))
                folder.mkdir(parents=True, exist_ok=True)
                for target in ("crop", "disease"):
                    atomic_json(folder / f"{target}_refit_history.json", [{"seconds": 1}])

            def evaluate(config, chosen, *_):
                self.assertEqual(sum(e[0] == "refit" for e in events), 3)
                events.append(("test", config["model"]["name"]))
                return {"gamma": chosen["validation"]["gamma"], "selected_epoch": chosen["selected_epoch"],
                        "test_seen": {"top1": .7}, "test_unseen": {"top1": .2},
                        "harmonic_mean_top1": .31, "harmonic_mean_top5": .6, "harmonic_mean_top1_gamma0": .1}

            with patch.object(finalizer, "PROJECT", root), patch.object(finalizer, "environment", return_value={}), \
                 patch.object(finalizer, "require_cuda"), patch.object(finalizer, "audit_data", side_effect=audit), \
                 patch("dpd.resize_cache.ensure_resized", return_value=root), \
                 patch.object(finalizer.runner, "refit", side_effect=refit), \
                 patch.object(finalizer.runner, "evaluate_final", side_effect=evaluate), \
                 patch.object(finalizer.runner, "run_pair") as selection, redirect_stdout(io.StringIO()):
                finalizer.finalize(path)
            self.assertEqual([e[0] for e in events], ["refit"]*3 + ["test"]*3)
            selection.assert_not_called()
            self.assertEqual(read_json(destination / "status.json")["state"], "completed")
            self.assertEqual(read_json(source / "status.json")["state"], "paused")

    def test_stop_request_pauses_before_next_epoch_and_skips_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, path, _, _, _, audit = fixture(root)
            destination = root / spec["output_dir"]

            def refit(_config, _chosen, _folder, _identity, _train, _cache, _deadline, update):
                atomic_json(destination / "STOP.request", {"reason": "fixture user stop"})
                update(phase="refit", active={"epoch": 2})

            with patch.object(finalizer, "PROJECT", root), patch.object(finalizer, "environment", return_value={}), \
                 patch.object(finalizer, "require_cuda"), patch.object(finalizer, "audit_data", side_effect=audit), \
                 patch("dpd.resize_cache.ensure_resized", return_value=root), \
                 patch.object(finalizer.runner, "refit", side_effect=refit), \
                 patch.object(finalizer.runner, "evaluate_final") as evaluate, redirect_stdout(io.StringIO()):
                finalizer.finalize(path)
            self.assertEqual(read_json(destination / "status.json")["state"], "paused")
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
