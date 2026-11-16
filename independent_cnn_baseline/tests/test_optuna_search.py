"""Generated-image and persistent-DB tests; no DPD fitting or downloads."""
import copy
import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import torch

from independent_cnn_baseline.engine import read_json
from independent_cnn_baseline.optuna_search import (SearchStop, boundary, distributions,
    immutable_json, open_study, register_incumbent, request_trial, run_trial,
    snapshot, trial_config, verify_terminal)
from independent_cnn_baseline.tests.test_study import TinyNet, fixture

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2)


def reference():
    return {"validation": {"harmonic_mean_top1": .5}, "folder": "original"}


class SearchTests(unittest.TestCase):
    def test_persistent_identity_cap_and_interrupted_parameter_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = open_study(root, "identity")
            register_incumbent(study, reference())
            trial = request_trial(study, 1)
            params = dict(trial.params)
            again = open_study(root, "identity")
            resumed = request_trial(again, 1)
            self.assertEqual(resumed.number, trial.number)
            self.assertEqual(resumed.params, params)
            again.tell(resumed, .6)
            self.assertIsNone(request_trial(again, 1))
            with self.assertRaisesRegex(ValueError, "identity"):
                open_study(root, "different")
            # A crash in ask(), with only the first suggestion committed, recovers
            # the remaining parameter draws exactly.
            one = optuna.create_study(direction="maximize")
            two = optuna.create_study(direction="maximize")
            register_incumbent(one, reference())
            register_incumbent(two, reference())
            expected = request_trial(one, 2).params
            from independent_cnn_baseline.optuna_search import sampler_for
            two.sampler = sampler_for(1)
            partial = two.ask()
            partial.suggest_float("lr_backbone", 1e-5, 1.5e-4, log=True)
            recovered = request_trial(two, 2)
            self.assertEqual(expected, recovered.params)

    def test_source_controls_and_original_lr_horizon_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, _, _ = fixture(temp)
            cfg["training"].update(max_epochs=30, warmup_epochs=3)
            original = copy.deepcopy(cfg)
            with patch("independent_cnn_baseline.optuna_search.PROJECT", Path(temp)):
                tuned = trial_config(cfg, {"lr_backbone": 1e-5, "weight_decay": .01, "head_dropout": .4}, 1, Path(temp))
            self.assertEqual(cfg, original)
            self.assertEqual(tuned["training"]["max_epochs"], 30)
            self.assertEqual(tuned["optimizer"]["lr_head"], 1e-4)
            self.assertEqual(tuned["seed"], 42)
            self.assertEqual(tuned["input"], original["input"])
            self.assertEqual(tuned["augmentation"], original["augmentation"])
            self.assertEqual(tuned["model"]["head_dropout"], .4)
            with self.assertRaisesRegex(ValueError, "Immutable"):
                immutable_json(Path(temp) / "config.json", {"a": 1})
                immutable_json(Path(temp) / "config.json", {"a": 2})

    def test_user_pause_and_deadline_do_not_finish_a_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(SearchStop) as cap:
                boundary(root, time.monotonic() - 1)
            self.assertEqual(cap.exception.state, "window_complete")
            (root / "STOP.request").write_text("stop")
            with self.assertRaises(SearchStop) as paused:
                boundary(root, float("inf"))
            self.assertEqual(paused.exception.state, "paused")
            self.assertTrue((root / "STOP.request").exists())

    def test_paired_epoch_recovery_matches_uninterrupted_training_exactly(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg, frames, pairs, cache = fixture(temp)
            study = open_study(root, "identity")
            trial = request_trial(study, 1)
            folder = root / "resumed"
            update = lambda **kw: None
            with patch("independent_cnn_baseline.optuna_search.boundary", side_effect=[None, SearchStop("window_complete")]):
                with self.assertRaises(SearchStop):
                    run_trial(trial, cfg, folder, "identity", frames, pairs, cache, root, float("inf"), update, TinyNet)
            self.assertTrue((folder / "crop/last.pt").exists())
            self.assertEqual(study.trials[0].state, optuna.trial.TrialState.RUNNING)
            self.assertEqual(study.trials[0].intermediate_values, {})
            result = run_trial(trial, cfg, folder, "identity", frames, pairs, cache, root, float("inf"), update, TinyNet)
            # A separate fresh objective uses the same data and actual model seed.
            fresh_study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
            fresh = fresh_study.ask()
            expected_folder = root / "fresh"
            expected = run_trial(fresh, cfg, expected_folder, "identity", frames, pairs, cache, root, float("inf"), update, TinyNet)
            self.assertEqual(result["best"]["validation"], expected["best"]["validation"])
            for target in ("crop", "disease"):
                a = torch.load(folder / target / "last.pt", weights_only=True)
                b = torch.load(expected_folder / target / "last.pt", weights_only=True)
                for key in a["model_state"]:
                    torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
            self.assertEqual(list(study.trials[0].intermediate_values), [1, 2])
            # A committed terminal trial can be reconciled after a crash before tell().
            self.assertEqual(run_trial(trial, cfg, folder, "identity", frames, pairs, cache, root, 0, update, TinyNet), result)
            (folder / "history.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_terminal(folder, result["identity"], frames, pairs, np.array([0., .1]))

    def test_best_pair_transaction_repeats_without_training_on_recovery(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cfg, frames, pairs, cache = fixture(temp)
            cfg["training"]["max_epochs"] = 1
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            folder = root / "trial"
            # Both last.pt files exist but pair_state was not committed.
            with patch("independent_cnn_baseline.optuna_search.validation", side_effect=RuntimeError("interrupted pair commit")):
                with self.assertRaisesRegex(RuntimeError, "pair commit"):
                    run_trial(trial, cfg, folder, "id", frames, pairs, cache, root, float("inf"), lambda **kw: None, TinyNet)
            with patch("independent_cnn_baseline.engine.Actor.epoch_step", side_effect=AssertionError("must not train twice")):
                result = run_trial(trial, cfg, folder, "id", frames, pairs, cache, root, float("inf"), lambda **kw: None, TinyNet)
            self.assertEqual(result["completed_paired_epochs"], 1)

    def test_median_grace_missing_r0_steps_and_eligible_winner_ties(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = open_study(root, "id")
            register_incumbent(study, reference())
            self.assertEqual(study.trials[0].intermediate_values, {})
            first = request_trial(study, 8)
            for epoch in range(1, 10):
                first.report(.6, epoch)
            study.tell(first, .6)
            second = request_trial(study, 8)
            second.report(.1, 5)
            self.assertFalse(second.should_prune())
            second.report(.1, 6)
            self.assertTrue(second.should_prune())
            study.tell(second, state=optuna.trial.TrialState.PRUNED)
            tie = request_trial(study, 8)
            study.tell(tie, .6)
            running = request_trial(study, 8)
            running.report(.99, 1)
            summary = snapshot(study, root, reference(), "window_complete")
            self.assertEqual(summary["best_eligible"]["number"], first.number)
            self.assertFalse(summary["search_complete"])
            self.assertFalse(summary["automatic_final_refit"])
            self.assertFalse(summary["official_test_inference"])

    def test_controller_stops_at_eight_new_trials_and_never_refits_or_tests(self):
        import json
        from independent_cnn_baseline.optuna_search import run
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            project = Path(temp)
            cfg, frames, pairs, cache = fixture(project)
            cfg["runtime"]["torch_threads"] = 2
            relative = "runs/baseline/independent_cnn_baseline_convnext_base_optuna_seed42"
            spec = {"output_dir": relative, "source_root": "source", "base_config": cfg,
                    "window_hours": 10, "max_new_trials": 8}
            path = project / "spec.json"
            path.write_text(json.dumps(spec))
            calls = []
            def fake_trial(trial, *args):
                calls.append(trial.number)
                return {"outcome": "COMPLETE", "best": {"validation": {"harmonic_mean_top1": .4}}}
            with patch("independent_cnn_baseline.optuna_search.PROJECT", project), \
                 patch("independent_cnn_baseline.optuna_search.audit", return_value=("id", frames, pairs, cache, reference())), \
                 patch("independent_cnn_baseline.optuna_search.run_trial", side_effect=fake_trial), \
                 patch("independent_cnn_baseline.optuna_search.retire_recovery"), \
                 patch("independent_cnn_baseline.runner.refit", side_effect=AssertionError("unauthorized refit")), \
                 patch("independent_cnn_baseline.runner.evaluate_final", side_effect=AssertionError("unauthorized official inference")):
                run(path)
            self.assertEqual(calls, list(range(1, 9)))
            self.assertEqual(read_json(project / relative / "status.json")["state"], "selection_complete")
            self.assertEqual(read_json(project / relative / "selected_recipe.json")["number"], 0)


if __name__ == "__main__":
    unittest.main()
