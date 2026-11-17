"""CLIP disease-only scores from the v10.3 runs (2026-11-16 night; idea from Hein's disease-transfer experiments, where a
disease-only CLIP head named the disease of 45% of unseen test images against 14.5% for a ConvNeXt disease head).
v10.3 already learns one prompt per disease ("a photo of leaves with {disease}: {description}."), so its disease-only score is
the cosine between the image embedding and that disease prompt; no new model is trained for test.
  test:       the two finished runs (runs/vlm/v10.3, runs/vlm/v10.3_s45): crop, disease and pair cosines of both test sets;
              the pair cosines must reproduce the run's saved scores.npz.
  validation: for each run, its three validation splits (seeds 42-44 and 45-47) re-trained as train/ensemble.py does
              (fit part, the run's selected epoch) and scored on the validation images, saving the same three cosines.
Writes test_primitives.npz and val_primitives_split{1,2,3}.npz into each run folder.
Usage (repo root, DPD_ROOT set): python train/supcon_eval/clip_disease_scores.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.clip_utils import clip_transforms  # noqa: E402
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.resize_cache import ensure_resized  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402
from train import train_lora as tl  # noqa: E402

DEV = "cuda"
RUNS = ["v10.3", "v10.3_s45"]


def scores(model, frame, image_dir, tf, cfg):
    dl = tl.loader(frame, image_dir, tf, cfg, shuffle=False)
    amp = cfg["runtime"]["amp"]
    feats = tl.encode(model, dl, DEV, amp, tuple(cfg["evaluation"].get("tta", ["identity"])))
    tl.release_loader(dl)
    crop, dis = tl.primitive_scores(model, feats, amp)
    return tl.pair_scores(model, feats, amp), crop, dis


def main():
    df = read_metadata()
    pairs = pair_table(df)
    train = df[df["split"] == "train"]
    for run in RUNS:
        d = REPO / "runs" / "vlm" / run
        cfg = json.load(open(d / "config.json", encoding="utf-8"))
        metrics = json.load(open(d / "metrics.json"))
        assert not metrics.get("lora_scale") and not metrics.get("fusion_crop_weight"), "plain scoring expected"
        image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
        tfs = clip_transforms(cfg["input"], cfg["augmentation"])
        if not (d / "test_primitives.npz").exists():
            model, _ = tl.build(cfg, pairs, DEV)
            st = torch.load(d / "prompt_vectors.pt", map_location=DEV)
            model.crop_vecs.data.copy_(st["crop_vecs"])
            model.disease_vecs.data.copy_(st["disease_vecs"])
            model.clip.visual.load_state_dict(torch.load(d / "lora_image_encoder.pt", map_location="cpu"), strict=False)
            model.eval()
            saved = np.load(d / "scores.npz")
            out = {}
            for s, tag in (("test_seen", "ts"), ("test_unseen", "tu")):
                pair, crop, dis = scores(model, df[df["split"] == s], image_dir, tfs[1], cfg)
                gap = np.abs(pair - saved[f"{s}_scores"]).max()
                agree = (pair.argmax(1) == saved[f"{s}_scores"].argmax(1)).mean()
                print(f"{run} {s}: pair cosines vs saved scores max diff {gap:.4f}, same top-1 {agree:.4f}", flush=True)
                out.update({f"pair_{tag}": pair, f"crop_{tag}": crop, f"dis_{tag}": dis, f"y_{tag}": saved[f"{s}_labels"]})
            np.savez_compressed(d / "test_primitives.npz", **out)
            del model
            tl.free_memory()
        v = cfg["validation"]
        for k in range(3):
            f = d / f"val_primitives_split{k + 1}.npz"
            if f.exists():
                continue
            fit_df, val_seen, val_unseen, _ = make_validation_split(train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
                                                                    cfg["seed"] + k, tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
            ep = metrics["selected_epoch"]
            print(f"{run} split {k + 1} (seed {cfg['seed'] + k}): retraining {ep} epochs on {len(fit_df)} images", flush=True)
            tl.seed_everything(cfg["seed"])
            model, lora = tl.build(cfg, pairs, DEV)
            tl.fit(model, lora, fit_df, sorted(set(fit_df["pair_id"])), ep, cfg, image_dir, tfs,
                   log=lambda r: print(f"  epoch {r['epoch']} loss {r['loss']}", flush=True))
            val_df = pd.concat([val_seen, val_unseen])
            pair, crop, dis = scores(model, val_df, image_dir, tfs[1], cfg)
            np.savez_compressed(f, pair=pair, crop=crop, dis=dis, y_val_seen=val_seen["pair_id"].to_numpy(),
                                y_val_unseen=val_unseen["pair_id"].to_numpy(), fit_pairs=np.sort(fit_df["pair_id"].unique()))
            print(f"  saved {f.name}", flush=True)
            del model, lora
            tl.free_memory()


if __name__ == "__main__":
    main()
