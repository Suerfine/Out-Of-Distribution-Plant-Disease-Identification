"""Generated-image tests; these do not train on DPD or download pretrained weights."""
import copy
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from independent_cnn_baseline.engine import (Actor, StudyLock, accumulation_samples, choose,
                                              fingerprint, loader, lr_factor, read_json, select_gamma)
from independent_cnn_baseline.models import transforms
from independent_cnn_baseline.runner import WindowEnd, run_pair, study, window_timing
from dpd.evaluate import evaluate

torch.set_num_threads(2)
PROJECT = Path(__file__).resolve().parents[2]


class TinyNet(nn.Module):
    def __init__(self, config, target, pretrained=False):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(config["model"]["head_dropout"]), nn.Linear(4, 55 if target == "crop" else 175))

    def forward(self, x):
        return self.head(self.backbone(x))


def fixture(folder):
    folder = Path(folder)
    cache = folder / "images"
    cache.mkdir()
    rows = []
    for i in range(11):
        pixels = np.random.default_rng(i).integers(0, 256, (14, 16, 3), dtype=np.uint8)
        name = f"image{i}.jpg"
        Image.fromarray(pixels).save(cache / name)
        pair = [0, 2, 3][i % 3] if i < 7 else (0 if i < 9 else 1)
        rows.append({"file": name, "crop_id": pair // 2, "disease_id": pair % 2, "pair_id": pair})
    frame = pd.DataFrame(rows)
    frames = {"fit": frame.iloc[:7], "validation_seen": frame.iloc[7:9], "validation_unseen": frame.iloc[9:], "all_train": frame}
    pairs = pd.DataFrame({"pair_id": range(4), "crop_id": [0, 0, 1, 1], "disease_id": [0, 1, 0, 1],
                          "pair": ["ax", "ay", "bx", "by"], "is_unseen": [False] * 4})
    cfg = {"seed": 42, "version": "tiny", "model": {"name": "tiny", "head_dropout": 0.2, "freeze_bn_stats": False},
           "input": {"crop_size": 8, "resize": 10, "resize_cache": 10, "interpolation": "bilinear", "mean": [0.5] * 3, "std": [0.5] * 3},
           "augmentation": {"random_resized_crop_scale": [0.5, 1.0], "horizontal_flip_p": 0.5,
                            "vertical_flip_p": 0.5, "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0}},
           "optimizer": {"lr_backbone": 1e-3, "lr_head": 1e-2, "weight_decay": 0.05, "betas": [0.9, 0.999], "eps": 1e-8},
           "training": {"batch_size": 3, "accumulation_steps": 2, "max_epochs": 2, "warmup_epochs": 1,
                        "min_lr_ratio": 0.01, "patience": 8, "label_smoothing": 0.1, "gradient_clip": 1},
           "runtime": {"device": "cpu", "amp": False, "num_workers": 0, "inference_batch_size": 1}}
    return cfg, frames, pairs, cache


class ProtocolTests(unittest.TestCase):
    def test_gamma_search_matches_shared_evaluator_with_ties_and_hidden_classes(self):
        rng = np.random.default_rng(1)
        pairs = pd.DataFrame({"pair_id": range(7), "crop_id": range(7), "disease_id": range(7),
                              "pair": list("abcdefg"), "is_unseen": [False] * 3 + [True] * 4})
        labels = np.array([0, 1, 2, 0, 3, 4, 5, 3])
        scores = rng.normal(size=(8, 7))
        scores[0, 0] = scores[0, 3] = 10
        scores[:, 6] = 100  # Hidden real-unseen candidate must never affect validation.
        allowed = np.array([True] * 6 + [False])
        gammas = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
        selected = select_gamma(scores, labels, pairs.is_unseen.to_numpy(), allowed, 4, gammas)
        masked = scores.copy()
        masked[:, ~allowed] = -np.inf
        actual = [evaluate(masked[:4], labels[:4], masked[4:], labels[4:], pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
        self.assertEqual(selected["gamma"], gammas[np.argmax(actual)])
        self.assertEqual(selected["harmonic_mean_top1"], max(actual))

    def test_accumulation_partial_groups_match_sample_mean_gradient(self):
        x = torch.arange(1.0, 8.0)
        expected = torch.tensor(1.0, requires_grad=True)
        actual = torch.tensor(1.0, requires_grad=True)
        for group in (x[:6], x[6:]):
            ((expected * group).square().mean()).backward()
        for i, batch in enumerate(x.split(3)):
            denominator = accumulation_samples(i, 3, 3, 7, 2)
            ((actual * batch).square().mean() * len(batch) / denominator).backward()
        torch.testing.assert_close(actual.grad, expected.grad)

    def test_calibration_matches_shared_ranker_for_tied_333_class_scores(self):
        rng = np.random.default_rng(31)
        scores = rng.normal(size=(8, 333)).astype(np.float32)
        scores[:4, 0] = 20
        scores[4:, 307] = scores[4:, 309] = 20
        scores[:, 332] = 100
        labels = np.array([0]*4 + [309]*4)
        unseen = np.zeros(333, dtype=bool)
        unseen[[307, 309, 332]] = True
        allowed = np.ones(333, dtype=bool)
        allowed[332] = False
        pairs = pd.DataFrame({"pair_id": range(333), "crop_id": range(333), "disease_id": range(333),
                              "pair": [str(i) for i in range(333)], "is_unseen": unseen})
        gammas = np.round(np.arange(0, 10, .1), 6)
        actual = select_gamma(scores, labels, unseen, allowed, 4, gammas)
        scores[:, ~allowed] = -np.inf
        reference = [evaluate(scores[:4], labels[:4], scores[4:], labels[4:], pairs, float(g), (1,)) for g in gammas]
        selected = max(reference, key=lambda r: r["harmonic_mean_top1"])
        self.assertEqual(actual["gamma"], selected["gamma"])
        self.assertEqual(actual["harmonic_mean_top1"], selected["harmonic_mean_top1"])
        self.assertEqual(actual["unseen_top1"], selected["test_unseen"]["top1"])

    def test_recovery_preserves_original_deadline_and_rejects_extension(self):
        previous = {"window_started_at_utc": "2026-10-27T17:09:12.367334+00:00", "window_hours": 10}
        _, fields = window_timing(previous, 10, "2026-10-28T03:09:12.367334+00:00")
        self.assertEqual(fields["window_started_at_utc"], previous["window_started_at_utc"])
        self.assertEqual(fields["window_deadline_at_utc"], "2026-10-28T03:09:12.367334+00:00")
        with self.assertRaisesRegex(ValueError, "original window deadline"):
            window_timing(previous, 10, "2026-10-28T04:09:12.367334+00:00")

    def test_checkpoint_resume_preserves_next_epoch_exactly(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            cfg, frames, _, cache = fixture(temp)
            actor = Actor("disease", cfg, Path(temp) / "state", "identity", TinyNet)
            actor.epoch_step(frames["fit"], frames["validation_seen"], cache)
            checkpoint = (actor.folder / "last.pt").read_bytes()
            actor.epoch_step(frames["fit"], frames["validation_seen"], cache)
            expected = {k: v.clone() for k, v in actor.model.state_dict().items()}
            (actor.folder / "last.pt").write_bytes(checkpoint)
            resumed = Actor("disease", cfg, actor.folder, "identity", TinyNet)
            resumed.epoch_step(frames["fit"], frames["validation_seen"], cache)
            for key, value in expected.items():
                torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
            for key in ("train_loss", "train_top1", "validation_loss", "validation_top1"):
                self.assertEqual(actor.history[-1][key], resumed.history[-1][key])

    def test_paired_mid_epoch_boundary_resume_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch("independent_cnn_baseline.runner.PROJECT", Path(temp)):
            cfg, frames, pairs, cache = fixture(temp)
            folder = Path(temp) / "trial"
            spec = {"output_dir": temp, "validation": {"gamma_grid": [0, 2, 0.25]}}
            with patch("independent_cnn_baseline.runner.check_deadline", side_effect=[None, WindowEnd()]):
                with self.assertRaises(WindowEnd):
                    run_pair(cfg, folder, "data", frames, pairs, cache, spec, float("inf"), lambda **kw: None, TinyNet)
            self.assertTrue((folder / "crop/last.pt").exists())
            result = run_pair(cfg, folder, "data", frames, pairs, cache, spec, float("inf"), lambda **kw: None, TinyNet)
            self.assertEqual(result["completed_epochs"], 2)
            self.assertFalse((folder / "crop/last.pt").exists())
            self.assertFalse((folder / "disease/last.pt").exists())
            with patch.object(TinyNet, "__init__", side_effect=AssertionError("must skip completed training")):
                self.assertEqual(run_pair(cfg, folder, "data", frames, pairs, cache, spec, float("inf"), lambda **kw: None, TinyNet), result)

    def test_windows_worker_loader_and_model_specific_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, frames, _, cache = fixture(temp)
            cfg["runtime"]["num_workers"] = 2
            batch = next(iter(loader(frames["fit"], cache, transforms(cfg)[1], cfg)))
            self.assertEqual(batch[0].shape, (3, 3, 8, 8))
            image = Image.new("RGB", (12, 12), (255, 255, 255))
            output = transforms(cfg)[1](image)
            torch.testing.assert_close(output, torch.ones_like(output))

    def test_os_lock_and_stable_validation_only_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            with StudyLock(Path(temp) / "lock"):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    with StudyLock(Path(temp) / "lock"):
                        pass
        a = {"validation": {"harmonic_mean_top1": 0.5}, "test": 0.0}
        b = {"validation": {"harmonic_mean_top1": 0.5}, "test": 1.0}
        self.assertIs(choose([a, b]), a)

    def test_refit_keeps_same_lr_trajectory(self):
        t = {"warmup_epochs": 3, "max_epochs": 30, "min_lr_ratio": 0.01}
        self.assertEqual([lr_factor(e, t) for e in (1, 2, 3)], [1/3, 2/3, 1])
        self.assertEqual(lr_factor(30, t), 0.01)
        # Selected refit duration does not replace the original scheduler horizon.
        self.assertGreater(lr_factor(10, t), 0.01)

    def test_app_adapter_matches_saved_cache_scores_and_keeps_legacy_route(self):
        from independent_cnn_baseline.inference import IndependentCnnRunModel
        from dpd.cnn import compose_pair_scores
        from dpd.inference import build_run_model
        from dpd.resize_cache import standardise

        with tempfile.TemporaryDirectory() as temp:
            cfg, frames, pairs, cache = fixture(temp)
            cfg["implementation"] = "independent_cnn_baseline"
            original = Path(temp) / "original.png"
            Image.new("RGB", (19, 13), (60, 140, 90)).save(original)
            cached = standardise(original, cfg["input"]["resize_cache"])
            x = transforms(cfg)[1](cached).unsqueeze(0)
            logp = {}
            for target in ("crop", "disease"):
                model = TinyNet(cfg, target).eval()
                torch.save(model.state_dict(), Path(temp) / f"{target}_net.pt")
                with torch.no_grad():
                    logp[target] = torch.log_softmax(model(x), 1).numpy()
            expected = compose_pair_scores(logp["crop"], logp["disease"], pairs)[0]
            run = {"dir": Path(temp), "cfg": cfg, "metrics": {"gamma": 1.1}, "method": "cnn_independent"}
            with patch("independent_cnn_baseline.inference.IndependentNet", TinyNet), \
                 patch("torch.cuda.is_available", return_value=False):
                app = build_run_model(run, pairs)
                self.assertIsInstance(app, IndependentCnnRunModel)
                embedding, actual = app.predict(original)
            np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)
            legacy = {"cfg": {}, "method": "cnn_independent"}
            with patch("dpd.inference.model_class") as factory:
                build_run_model(legacy, pairs)
                factory.assert_called_once_with("cnn_independent")


class OrchestrationTests(unittest.TestCase):
    def test_exact_adaptive_queue_and_freeze_before_tests(self):
        spec = json.loads((PROJECT / "independent_cnn_baseline/configs/study_seed42.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec["runtime"].update(device="cpu", amp=False)
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            data = pd.DataFrame({"file": ["a", "b"], "crop_id": [0, 1], "disease_id": [0, 1], "pair_id": [0, 1]})
            frames = {k: data for k in ("fit", "validation_seen", "validation_unseen", "all_train")}
            prepared = {m["name"]: {"pretrained_file": "dummy", "sha256": "abc", "revision": "pinned", "batch_size": 4,
                                      "preprocessing": {"mean": [0.5]*3, "std": [0.5]*3, "interpolation": "bilinear"}} for m in spec["models"]}
            executed, official = [], []

            def fake_pair(cfg, folder, *_args):
                executed.append(copy.deepcopy(cfg))
                index = int(cfg["version"].split("_r")[1].split("_")[0])
                return {"name": cfg["version"], "config": cfg, "selected_epoch": 2,
                        "validation": {"harmonic_mean_top1": [0.1,0.2,0.3,0.35,0.34,0.4][index], "gamma": 1.0},
                        "training_epoch_seconds": 1}

            def fake_eval(cfg, chosen, *_args):
                self.assertEqual(len(executed), 18)
                self.assertTrue((root / spec["output_dir"] / "selected_models.json").exists())
                official.append(cfg["version"])
                return {"test_seen": {"top1": 0.5}, "test_unseen": {"top1": 0.5},
                        "harmonic_mean_top1": 0.5, "harmonic_mean_top5": 0.8, "gamma": 1.0, "selected_epoch": 2}

            with patch("independent_cnn_baseline.runner.PROJECT", root), \
                 patch("independent_cnn_baseline.runner.require_cuda"), \
                 patch("independent_cnn_baseline.runner.environment", return_value={"test_double": True}), \
                 patch("independent_cnn_baseline.runner.prepare_models", return_value=prepared), \
                 patch("independent_cnn_baseline.runner.audit_data", return_value=(data, None, frames, {"data_identity": "data"})), \
                 patch("dpd.resize_cache.ensure_resized", return_value=root), \
                 patch("independent_cnn_baseline.runner.run_pair", side_effect=fake_pair), \
                 patch("independent_cnn_baseline.runner.refit") as refit_mock, \
                 patch("independent_cnn_baseline.runner.evaluate_final", side_effect=fake_eval), \
                 redirect_stdout(io.StringIO()):
                study(path)
            self.assertEqual(len(executed), 18)
            self.assertEqual(len(official), 3)
            self.assertEqual(refit_mock.call_count, 3)
            self.assertTrue(all("_r0_" in c["version"] for c in executed[:3]))
            for c in executed:
                if "_r3_" in c["version"]:
                    self.assertEqual(c["optimizer"]["lr_backbone"], 1e-4)
                if "_r4_" in c["version"] or "_r5_" in c["version"]:
                    self.assertEqual(c["optimizer"]["weight_decay"], 0.01)
            state = read_json(root / spec["output_dir"] / "status.json")
            self.assertEqual(state["state"], "completed")


if __name__ == "__main__":
    unittest.main()
