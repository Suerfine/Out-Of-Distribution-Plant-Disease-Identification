"""Independent NumPy verification of the completed scientific result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from protocol import ARMS, sha256, stamp, write_json


def numpy_metrics(scores, targets):
    order = np.argsort(-scores, axis=1, kind="stable")
    correct = order[:, 0] == targets
    return {"top1": float(correct.mean()),
            "top5": float((order[:, :min(5, scores.shape[1])] == targets[:, None]).any(1).mean()),
            "macro_top1": float(np.mean([correct[targets == c].mean() for c in np.unique(targets)]))}, order


def verify_run(run):
    run = Path(run)
    prepared = json.loads((run / "prepared.json").read_text())
    report = json.loads((run / "metrics.json").read_text())
    frozen = json.loads((run / "selection_frozen_before_test.json").read_text())
    cfg, ontology = prepared["config"], prepared["ontology"]
    roles = prepared["roles"]
    checks = 0

    def check(condition, description):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(description)

    def close(a, b, description):
        check(abs(float(a)-float(b)) < 1e-10, description)

    check(report["candidate_count"] == 333, "Full ontology")
    check(set(report["arms"]) == set(ARMS), "Both arms reported")
    check(frozen["fingerprint"] == prepared["fingerprint"], "Frozen provenance")
    check(not frozen["official_scores_used"], "No final scores used for selection")
    for path, expected in {**prepared["source_hashes"], **prepared["csv_hashes"], **prepared["manifest_hashes"], **prepared["historical_hashes"]}.items():
        check(sha256(path) == expected, f"Source or historical file changed: {path}")
    check(sha256(cfg["pretrained_weights"]) == prepared["pretrained_sha256"], "Original pretrained weights")
    for path, expected in prepared["image_hashes"].items():
        check(sha256(path) == expected, f"Image content changed: {path}")
    fitting=roles['fit']
    check(len(fitting)==6746 and len(roles['seen_validation'])==1254,'Original split sizes')
    check(not ({prepared['image_hashes'][r['path']] for r in fitting} & {prepared['image_hashes'][r['path']] for r in roles['seen_validation']}),'No content leakage')
    import csv as csv_module
    for role,name in (('fit','fit.csv'),('seen_validation','validation.csv')):
        with (Path(cfg['baseline_run'])/name).open(encoding='utf-8-sig',newline='') as f: original=list(csv_module.DictReader(f))
        check([r['filename'] for r in original]==[r['filename'] for r in roles[role]],'Exact original manifest membership/order')
    check(sha256(prepared['crop_checkpoint'])==prepared['crop_checkpoint_sha256']==frozen['crop_sha256'],'Original crop unchanged')
    initial_hashes = []
    from safetensors.torch import load_file
    original_weights = load_file(cfg["pretrained_weights"])
    parameter_changes = {}
    for arm in ARMS:
        best_path = run / arm / "best.pt"
        check(sha256(best_path) == frozen["selected"][arm]["sha256"], f"Frozen {arm} checkpoint unchanged")
        last = torch.load(run / arm / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        history = json.loads((run / arm / "history.json").read_text())
        check(last["complete"], f"{arm} training complete")
        check(last["fingerprint"] == best["fingerprint"] == prepared["fingerprint"], f"{arm} training provenance")
        check(history == last["history"], f"{arm} history matches checkpoint")
        check(cfg["minimum_epochs"] <= len(history) <= cfg["epochs"], f"{arm} bounded epoch budget")
        keys = [(h["validation"]["selection_score"], -h["validation"]["mean_ce"]) for h in history]
        selected = max(range(len(keys)), key=keys.__getitem__)
        check(best["epoch"] == selected+1 == frozen["selected"][arm]["epoch"] == last["best_epoch"], f"{arm} correct checkpoint selection")
        check(best["validation"] == history[selected]["validation"], f"{arm} selected validation")
        for tensor in best["model"].values():
            check(bool(torch.isfinite(tensor).all()), f"{arm} finite selected tensor")
        if arm in ARMS:
            initial_hashes.append(best["initial_hash"])
        changed = 0
        for key, value in original_weights.items():
            if not key.startswith("head.fc."):
                changed += int(torch.count_nonzero(best["model"]["net."+key] != value))
        check(changed > 0, f"{arm} pretrained backbone actually fine-tuned")
        parameter_changes[arm] = changed
        for item in history:
            exposure = run / arm / f"exposures_epoch{item['epoch']:02}.npz"
            bank = np.load(exposure)
            check(len(bank["indices"]) == item["training"]["samples"] == len(fitting), f"{arm} matched image budget")
            check(np.array_equal(bank["image_counts"], np.bincount(bank["indices"], minlength=len(fitting))), f"{arm} exposure counts")
            check(sha256(exposure) == item["training"]["exposure_sha256"], f"{arm} exposure hash")
            if arm in ARMS:
                check(len(np.unique(bank["indices"])) == len(fitting), f"{arm} ordinary image shuffle covers fit once")
        del last, best
    check(len(set(initial_hashes)) == 1, "Identical initialization of all disease arms")
    grouped_histories = [json.loads((run / a / "history.json").read_text()) for a in ("ce_random", "ce_memory")]
    for epoch in range(1, min(map(len, grouped_histories))+1):
        a = np.load(run / "ce_random" / f"exposures_epoch{epoch:02}.npz")["indices"]
        b = np.load(run / "ce_memory" / f"exposures_epoch{epoch:02}.npz")["indices"]
        check(np.array_equal(a, b), "Sampler-matched disease arms received identical indices")
    ci = {v: i for i, v in enumerate(ontology["crop"])}
    di = {v: i for i, v in enumerate(ontology["disease"])}
    pi = {c["composition_id"]: i for i, c in enumerate(ontology["candidates"])}
    cidx = np.array([ci[c["crop_id"]] for c in ontology["candidates"]])
    didx = np.array([di[c["disease_id"]] for c in ontology["candidates"]])
    pids = np.array([c["composition_id"] for c in ontology["candidates"]])
    freeze_time = datetime.fromisoformat(frozen["frozen_utc"]).timestamp()
    for arm in ARMS:
        for role in ("seen_validation", "official_seen", "official_unseen"):
            rows = roles[role]
            path = run / arm / f"logits_{role}.npz"
            bank = np.load(path)
            check(bank["crop"].shape == (len(rows), len(ci)), f"{arm}/{role} crop shape")
            check(bank["disease"].shape == (len(rows), len(di)), f"{arm}/{role} disease shape")
            check(np.isfinite(bank["crop"]).all() and np.isfinite(bank["disease"]).all(), f"{arm}/{role} finite logits")
            check(bank["filenames"].tolist() == [r["filename"] for r in rows], f"{arm}/{role} image order")
            check(bank["composition_targets"].tolist() == [r["composition_id"] for r in rows], f"{arm}/{role} targets")
            check(bank["checkpoint_hash"].item() == frozen["selected"][arm]["sha256"], f"{arm}/{role} logits provenance")
            check(np.array_equal(bank["crop"], np.load(run / f"crop_{role}.npz")["logits"]), "Identical crop predictions in every arm")
            if role in ("audit", "official_seen", "official_unseen"):
                check(path.stat().st_mtime >= freeze_time, "Final predictions written after selection freeze")
            scores = bank["crop"].astype(np.float64)[:, cidx] + bank["disease"].astype(np.float64)[:, didx]
            targets = np.array([pi[r["composition_id"]] for r in rows])
            result, ranking = numpy_metrics(scores, targets)
            expected = report["arms"][arm][role]
            for kind, source, y in (("crop", bank["crop"], np.array([ci[r["crop_id"]] for r in rows])),
                                     ("disease", bank["disease"], np.array([di[r["disease_id"]] for r in rows])),
                                     ("composition", scores, targets)):
                computed, _ = numpy_metrics(source, y)
                for metric in computed:
                    close(computed[metric], expected[kind][metric], f"{arm}/{role}/{kind}/{metric}")
            with (run / arm / f"predictions_{role}.csv").open(encoding="utf-8", newline="") as f:
                exported = list(csv.DictReader(f))
            check(len(exported) == len(rows), "Prediction row count")
            for i, row in enumerate(exported):
                check(row["filename"] == rows[i]["filename"] and int(row["true_composition_id"]) == rows[i]["composition_id"], "Exported row identity")
                check(json.loads(row["top5_ids"]) == pids[ranking[i, :5]].tolist(), "Exported full Top-5 ranking")
                check(int(row["predicted_composition_id"]) == pids[ranking[i, 0]], "Exported Top-1 ranking")
            for comp, detail in expected["per_composition"].items():
                mask = bank["composition_targets"] == int(comp)
                check(int(mask.sum()) == detail["images"], "Per-composition denominator")
                check(int((ranking[mask, 0] == targets[mask]).sum()) == detail["correct"], "Per-composition correct count")
        for key in ("top1", "top5", "macro_top1"):
            s = report["arms"][arm]["official_seen"]["composition"][key]
            u = report["arms"][arm]["official_unseen"]["composition"][key]
            close(2*s*u/(s+u) if s+u else 0, report["arms"][arm]["official_H"][key], "Official harmonic mean")
    value = {"passed": True, "at_utc": stamp(), "checks": checks, "verified_by": "independent NumPy stable rankings",
             "changed_pretrained_backbone_values": parameter_changes, "historical_artifacts_unchanged": True,
             "identical_disease_initialization": True, "matched_full_coverage_exposures": True}
    write_json(run / "independent_verification.json", value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(json.dumps(verify_run(parser.parse_args().run), indent=2))
