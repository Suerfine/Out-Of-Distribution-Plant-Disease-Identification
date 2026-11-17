"""Tiny full-data refit scope and recovery, using generated images only."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2 as T

from independent_cnn_baseline import tiny_refit as tiny
from independent_cnn_baseline.engine import atomic_json, sha256

torch.set_num_threads(2)


class ToyTiny(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(4, tiny.TARGETS[target])

    def get_classifier(self):
        return self.head

    def forward(self, images):
        return self.forward_head(self.forward_features(images))

    def forward_features(self, images):
        return self.conv(images)

    def forward_head(self, features, pre_logits=False):
        embedding = self.dropout(features.mean((2, 3)))
        return embedding if pre_logits else self.head(embedding)


def train_fixture(folder):
    folder = Path(folder)
    image_dir = folder / "images"
    image_dir.mkdir()
    rows = []
    for i in range(9):
        file = f"image{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(image_dir / file)
        rows.append({"file": file, "crop_id": i % 3, "disease_id": i % 4, "pair_id": i % 3})
    config = {"model": "toy", "device": "cpu", "amp": False, "workers": 0, "batch_size": 2,
              "accumulation_steps": 2, "image_size": 8, "epochs": 30, "warmup_epochs": 3,
              "backbone_lr": .001, "head_lr": .01, "weight_decay": .05, "label_smoothing": .1}
    return pd.DataFrame(rows), image_dir, config


class TinyRefitTests(unittest.TestCase):
    def test_app_adapter_matches_native_model_scores_without_using_labels(self):
        from dpd.cnn import compose_pair_scores
        from dpd.inference import build_run_model

        engine = tiny.legacy_engine()
        transform = T.Compose([T.Resize((8, 8)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        with tempfile.TemporaryDirectory() as temp:
            frame, images, config = train_fixture(temp)
            folder = Path(temp)
            logp = {}
            with Image.open(images / frame.file.iloc[0]) as image:
                batch = transform(image.convert("RGB")).unsqueeze(0)
            for target in tiny.TARGETS:
                model = ToyTiny(target).eval()
                torch.save(model.state_dict(), folder / f"{target}_net.pt")
                with torch.no_grad():
                    logp[target] = torch.log_softmax(model(batch), dim=-1).numpy()
            pairs = pd.DataFrame({"crop_id": [0, 0, 1, 1], "disease_id": [0, 1, 0, 1]})
            expected = compose_pair_scores(logp["crop"], logp["disease"], pairs)[0]
            run = {"dir": folder, "cfg": {"implementation": "independent_cnn_baseline_tiny_refit", "legacy_recipe": config},
                   "method": "cnn_independent", "metrics": {"gamma": 0.0}}
            with patch.object(tiny, "create_model", side_effect=lambda cfg, target, pretrained: ToyTiny(target)), \
                 patch.object(engine, "transforms_for", return_value=(transform, transform, {})), \
                 patch("torch.cuda.is_available", return_value=False):
                app = build_run_model(run, pairs)
                embedding, actual = app.predict(images / frame.file.iloc[0])
            np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_frozen_source_epochs_and_hashes_reject_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            project, legacy = Path(temp) / "repo", Path(temp) / "legacy"
            source = legacy / "runs/original"
            source.mkdir(parents=True)
            project.mkdir()
            config = {"model": "convnext_tiny.fb_in22k", "device": "cuda", "pretrained": True, "epochs": 30}
            atomic_json(source / "config.resolved.json", config)
            for target, epoch in (("crop", 17), ("disease", 16)):
                atomic_json(source / target / "summary.json", {"epoch": epoch})
            weights = legacy / "weights.safetensors"
            weights.write_bytes(b"pinned synthetic fixture")
            spec = {"legacy_root": "../legacy", "source_run": "runs/original", "output_dir": "runs/baseline/tiny",
                    "target_epochs": {"crop": 17, "disease": 16}, "target_seeds": {"crop": 42, "disease": 43},
                    "gamma": 0., "lr_schedule_horizon": 30, "legacy_code_sha256": {},
                    "source_artifact_sha256": {"config.resolved.json": sha256(source / "config.resolved.json")},
                    "pretrained_file": weights.name, "pretrained_sha256": sha256(weights)}
            with patch.object(tiny, "PROJECT", project), patch.object(tiny, "LEGACY", legacy):
                self.assertEqual(tiny.validate_spec(spec)[2], config)
                changed = copy.deepcopy(spec)
                changed["target_epochs"]["disease"] = 27
                with self.assertRaisesRegex(ValueError, "original validation-selected"):
                    tiny.validate_spec(changed)
                weights.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "pretrained"):
                    tiny.validate_spec(spec)

    def test_full_data_training_uses_fixed_epochs_and_original_lr_horizon(self):
        engine = tiny.legacy_engine()
        transform = T.Compose([T.Resize((8, 8)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        with tempfile.TemporaryDirectory() as temp:
            frame, images, config = train_fixture(temp)
            root = Path(temp) / "run"
            root.mkdir()
            with patch.object(tiny, "create_model", side_effect=lambda cfg, target, pretrained: ToyTiny(target)), \
                 patch.object(engine, "transforms_for", return_value=(transform, transform, {})), \
                 redirect_stdout(io.StringIO()):
                result = tiny.refit_target("crop", frame, images, config, 3, 42, root, "fixed", lambda **kw: None, lambda: None)
            history = json.loads((root / "crop/history.json").read_text())
            self.assertEqual(result["training_images"], 9)
            self.assertEqual(result["epochs"], 3)
            self.assertEqual(result["lr_schedule_horizon"], 30)
            self.assertEqual([r["backbone_lr"] for r in history], [.001 / 3, .002 / 3, .001])
            self.assertTrue(all("validation_top1" not in r for r in history))
            with patch.object(tiny, "create_model", side_effect=AssertionError("completed target must be preserved")):
                self.assertEqual(tiny.refit_target("crop", frame, images, config, 3, 42, root, "fixed", lambda **kw: None, lambda: None), result)

    def test_partial_epoch_boundary_resume_preserves_rng_and_final_weights(self):
        engine = tiny.legacy_engine()
        transform = T.Compose([T.RandomHorizontalFlip(), T.Resize((8, 8)), T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        with tempfile.TemporaryDirectory() as temp:
            frame, images, config = train_fixture(temp)
            full, resumed = Path(temp) / "full", Path(temp) / "resumed"
            full.mkdir(); resumed.mkdir()
            calls = 0

            def stop_after_first_epoch():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise tiny.StopRefit("paused")

            with patch.object(tiny, "create_model", side_effect=lambda cfg, target, pretrained: ToyTiny(target)), \
                 patch.object(engine, "transforms_for", return_value=(transform, transform, {})), \
                 redirect_stdout(io.StringIO()):
                tiny.refit_target("crop", frame, images, config, 3, 42, full, "same", lambda **kw: None, lambda: None)
                with self.assertRaises(tiny.StopRefit):
                    tiny.refit_target("crop", frame, images, config, 3, 42, resumed, "same", lambda **kw: None, stop_after_first_epoch)
                tiny.refit_target("crop", frame, images, config, 3, 42, resumed, "same", lambda **kw: None, lambda: None)
            left = torch.load(full / "crop_net.pt", weights_only=True)
            right = torch.load(resumed / "crop_net.pt", weights_only=True)
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)

    def test_both_frozen_full_data_refits_finish_before_official_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            source = project / "source"
            for target in tiny.TARGETS:
                (source / target).mkdir(parents=True)
                (source / target / "learning_curves.png").write_bytes(b"source curve")
            root = project / "runs/baseline/tiny"
            root.mkdir(parents=True)
            atomic_json(root / "cuda_preflight.json", {"passed": True})
            spec = {"name": "tiny", "output_dir": "runs/baseline/tiny", "window_hours": 10,
                    "pretrained_file": "original.safetensors", "source_artifact_sha256": {},
                    "target_epochs": {"crop": 17, "disease": 16}, "target_seeds": {"crop": 42, "disease": 43}}
            path = project / "spec.json"
            atomic_json(path, spec)
            original = {s: {"composition": {"top1": .5, "top5": .8}} for s in ("seen", "unseen")}
            original["harmonic_mean"] = {"top1": .5, "top5": .8}
            atomic_json(source / "metrics.json", original)
            result = {s: {"top1": .5, "top5": .8} for s in ("test_seen", "test_unseen")}
            result.update(harmonic_mean_top1=.5, harmonic_mean_top5=.8)
            atomic_json(project / "runs/baseline/independent_cnn_baseline_convnext_base_seed42/uncalibrated_metrics.json", result)
            frame = pd.DataFrame({"file": ["all_train_0", "restored_validation_1"]})
            frames = {"train": frame, "test_seen": frame.iloc[:1], "test_unseen": frame.iloc[1:]}
            calls = []

            def fit(target, train, image_dir, config, epochs, seed, *args):
                self.assertIs(train, frame)
                calls.append((target, epochs, seed))
                return {"training_epoch_seconds": 1}

            def test_once(*args):
                self.assertEqual(calls, [("crop", 17, 42), ("disease", 16, 43)])
                return result

            with patch.object(tiny, "PROJECT", project), \
                 patch.object(tiny, "validate_spec", return_value=(root, source, {"model": "convnext_tiny.fb_in22k"})), \
                 patch.object(tiny, "audit_data", return_value=(frames, None, project, {"image_hash_fingerprint": "data"})), \
                 patch.object(tiny, "refit_target", side_effect=fit), \
                 patch.object(tiny, "evaluate_final", side_effect=test_once) as evaluate_mock, \
                 patch("torch.cuda.is_available", return_value=True):
                tiny.study(path)
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main()
