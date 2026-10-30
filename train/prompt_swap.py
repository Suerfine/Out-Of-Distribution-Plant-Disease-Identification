"""Re-score a finished clip_prompt_lora run with different prompt text, WITHOUT retraining (exploratory probe).

Usage:  python -m train.prompt_swap --config config/vlm/v15.1_config.json

The config is the source run's config with changed `prompts` and a `prompt_swap.source_run` entry. The source run's
learned crop/disease vectors and LoRA adapters are loaded unchanged; only the pair text embeddings are rebuilt from the
new prompts, and the test images are re-encoded with the adapted image encoder. There is no validation model, so gamma
is the source run's (chosen for the old prompts); the test numbers are a probe, not a selected version.
"""

import argparse
import json
import shutil

import numpy as np
import torch

from dpd.clip_utils import clip_transforms, load_clip
from dpd.config import load_config, output_dir
from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary, hm_top1_over_gammas
from dpd.lora import load_lora
from dpd.prompt_learner import build_model
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from train.train_lora import encode, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "clip_prompt_lora" and "prompt_swap" in cfg, f"{args.config} is not a prompt-swap config"
    src = PROJECT_ROOT / cfg["prompt_swap"]["source_run"]
    src_metrics = json.loads((src / "metrics.json").read_text())
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")

    df = read_metadata()
    pairs = pair_table(df)
    clip_model, _, tokenizer = load_clip(cfg["model"], device)
    model = build_model(cfg, clip_model, tokenizer, pairs).to(device)
    vecs = torch.load(src / "prompt_vectors.pt", map_location=device)
    model.crop_vecs.data.copy_(vecs["crop_vecs"])
    model.disease_vecs.data.copy_(vecs["disease_vecs"])
    lcfg = cfg["image_encoder"]["lora"]
    load_lora(clip_model.visual, torch.load(src / "lora_image_encoder.pt", map_location="cpu"),
              lcfg["last_n_blocks"], lcfg["rank"], lcfg["alpha"], lcfg["targets"])
    model.to(device).eval()
    with torch.no_grad(), torch.autocast(device, enabled=amp):
        text = model.encode("pairs").float().cpu().numpy()

    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    test_tf = clip_transforms(cfg["input"], cfg["augmentation"])[1]
    tta = tuple(cfg["evaluation"].get("tta", ["identity"]))
    scores, labels, files = {}, {}, {}
    for split in cfg["data"]["eval_splits"]:
        part = df[df["split"] == split]
        scores[split] = encode(model, loader(part, image_dir, test_tf, cfg, shuffle=False), device, amp, tta) @ text.T
        labels[split], files[split] = part["pair_id"].to_numpy(), part["file"].tolist()

    gamma = float(src_metrics["gamma"])
    top_k = tuple(cfg["evaluation"]["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    # Diagnostic only, NOT for selection: how far the borrowed gamma is from this probe's own best test gamma.
    grid = np.arange(*cfg["validation"]["gamma_grid"])
    hm = hm_top1_over_gammas(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, grid)
    result.update(version=cfg["version"], source_run=cfg["prompt_swap"]["source_run"], selected_epoch=src_metrics.get("selected_epoch"),
                  val=src_metrics.get("val"), harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"],
                  diagnostic_test_oracle_gamma={"gamma": float(grid[hm.argmax()]), "hm_top1": float(hm.max())})
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print(f"\nTest, gamma {gamma} borrowed from {src.name}:\n" + format_summary(result))
    print(f"(diagnostic) best test gamma {grid[hm.argmax()]:.4f}: HM {hm.max():.2%}")

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "pair_text_embeddings.npy", text)
    for f in ("prompt_vectors.pt", "lora_image_encoder.pt"):
        shutil.copy(src / f, out / f)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
