"""Reproduce the saved Base development objective on CPU; train nothing."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECT = ROOT / "Out-Of-Distribution-Plant-Disease-Identification"
sys.path.insert(0, str(PROJECT))
from independent_cnn_baseline.uncalibrated_objective import validation_uncalibrated


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    study = PROJECT / "runs/baseline/independent_cnn_baseline_study_seed42"
    search = PROJECT / "runs/baseline/independent_cnn_baseline_convnext_base_optuna_gamma0_seed42"
    trial = search / "trials/trial_000"
    frames = {name: pd.read_csv(study / f"{name}.csv")
              for name in ("validation_seen", "validation_unseen")}
    pairs = pd.DataFrame(json.loads((study / "pairs.json").read_text()))
    bank_path = trial / "validation_epoch_15.npz"
    with np.load(bank_path) as bank:
        assert bank["crop"].shape == (1395, 55)
        assert bank["disease"].shape == (1395, 175)
        result = validation_uncalibrated(bank["crop"], bank["disease"], frames, pairs)
    expected = json.loads((search / "selected_recipe.json").read_text())["validation"]
    for key, value in result.items():
        if isinstance(value, (int, float)):
            assert abs(value - expected[key]) < 1e-12, (key, value, expected[key])
        else:
            assert value == expected[key]
    proposal = json.loads((HERE / "development_protocol_proposal.json").read_text())
    assert proposal["existing_base_partition_counts"] == {
        "fit": 5798, "validation_seen": 862, "validation_unseen": 533,
        "selection_excluded": 807, "all_train": 8000,
    }
    for name in ("fit", "validation_seen", "validation_unseen", "selection_excluded", "all_train"):
        assert sha(study / f"{name}.csv") == sha(HERE / "reference_manifests" / f"{name}.csv")
    paths = [bank_path, search / "selected_recipe.json", study / "pairs.json",
             trial / "identity.json", trial / "config.json",
             trial / "crop/best_epoch_15.pt", trial / "disease/best_epoch_15.pt"]
    output = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "device": "CPU", "training_started": False,
        "new_split_created": False, "existing_split_applied_to_new_training": False,
        "metrics_match_saved_recipe_tolerance": 1e-12,
        "metrics": result,
        "validation_order": "validation_seen.csv, then validation_unseen.csv; original saved row order",
        "development_candidates": 323, "official_unseen_candidates_masked": 10,
        "partition_counts": proposal["existing_base_partition_counts"],
        "original_manifests_copied_byte_identically": True,
        "saved_development_checkpoints_exist": True,
        "baseline_retraining_required": False,
        "source_sha256": {str(path): sha(path) for path in paths},
    }
    (HERE / "development_reference_verified.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "metrics": result,
                      "baseline_retraining_required": False}, indent=2))


if __name__ == "__main__":
    main()
