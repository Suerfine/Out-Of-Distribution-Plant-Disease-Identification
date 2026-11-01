"""Zero-shot CLIP: score every test image against a text prompt for each of the 333 pairs. No training.

Usage:  python zeroshot.py --config config/vlm/v3_knowledge_base_config.json
Writes to the config's output_dir: metrics.json, confusion_test_seen.csv, predictions_<split>.csv,
scores.npz (raw cosine scores, for later calibration analysis) and a copy of the config.
"""

import argparse
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch

from dpd.clip_utils import build_prompts, encode_images, encode_prompts, load_clip, truncated_prompts
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "clip_zeroshot", f"{args.config} is a '{cfg['method']}' config, not clip_zeroshot"

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    runtime = cfg["runtime"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")

    df = read_metadata()
    pairs = pair_table(df)
    model, preprocess, tokenizer = load_clip(cfg["model"], runtime["device"])

    prompts = build_prompts(pairs, cfg["names"], cfg["prompts"])
    dupes = pd.Series([" | ".join(p) for p in prompts]).duplicated(keep=False)
    if dupes.any():
        raise SystemExit(f"Identical prompts for different pairs: {pairs.loc[dupes.to_numpy(), 'pair'].tolist()}")
    too_long = truncated_prompts(tokenizer, prompts)
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed CLIP's 77 tokens, e.g. {too_long[0]!r}")
    print("Example prompts:", *[p[0] for p in prompts[:3]], sep="\n  ")
    # prompts.class_score "mean_of_scores": average the cosine scores of a class's prompts (classification by description)
    renormalize = cfg["prompts"].get("class_score", "normalized_mean") != "mean_of_scores"
    text = encode_prompts(model, tokenizer, prompts, runtime["device"], renormalize)

    scores, labels, files = {}, {}, {}
    for split in ("test_seen", "test_unseen"):
        sub = df[df["split"] == split]
        img = encode_images(model, preprocess, sub["file"], cfg["model"], runtime, cache_name=split)
        scores[split], labels[split], files[split] = img @ text.T, sub["pair_id"].to_numpy(), sub["file"].to_numpy()

    ev = cfg["evaluation"]
    gamma, top_k = ev["calibration_gamma"], tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    result["version"] = cfg["version"]
    print(format_summary(result))

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    with open(out / "prompts.json", "w", encoding="utf-8") as f:
        json.dump({p: ps for p, ps in zip(pairs["pair"], prompts)}, f, indent=1)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
