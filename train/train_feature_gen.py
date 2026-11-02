"""Feature generation for unseen pairs (f-CLSWGAN) on frozen CLIP image embeddings; see dpd/feature_gen.py.

Usage:  python -m train.train_feature_gen --config config/vlm/v11_feature_gen_config.json

Per run (a validation split, or the final fit on all train):
  1. aux classifier: a cosine classifier on the real features of the trained pairs (for the GAN's cls loss);
  2. GAN: G(noise, text condition) -> image feature, trained on the real features of the trained pairs;
  3. synthesis: synthesis.n_per_unseen features for each untrained target pair (the pseudo-unseen pairs in
     validation, the 10 real unseen pairs at the end), optionally topping up rare seen pairs;
  4. final classifier over all pairs on real + synthetic features, initialised from the pair text embeddings.
Stage 1 picks the final classifier's epoch and gamma on validation (mean HM over validation.n_splits splits);
stage 2 refits everything on all train and tests once. Scores are logits / 100, i.e. on CLIP's cosine scale, so
the gamma grid, the app (ClipRunModel) and the ensemble treat this run like the other CLIP runs.
"""

import argparse
import json
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dpd.clip_utils import build_prompts, display_name, encode_images, encode_prompts, load_clip, load_descriptions
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.feature_gen import fit_classifier, synthesise, train_gan
from dpd.report import save_test_outputs
from dpd.splits import make_validation_split

METHOD = "clip_feature_gen"
CLIP_SCALE = 100.0  # scores are saved as logits / CLIP_SCALE


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def text_conditions(cfg, clip_model, tokenizer, pairs, device) -> tuple[np.ndarray, np.ndarray]:
    """(pair text embeddings, generator condition per pair). The condition concatenates condition.parts:
    "pair" (the pair's prompts), "crop" and "disease" (each part's own prompts, looked up per pair)."""
    names, pcfg = cfg["names"], cfg["prompts"]
    u2s = names["underscore_to_space"]
    pair_emb = encode_prompts(clip_model, tokenizer, build_prompts(pairs, names, pcfg), device)
    descriptions = load_descriptions(pcfg)
    parts = []
    for part in cfg["condition"]["parts"]:
        if part == "pair":
            parts.append(pair_emb)
            continue
        ids = pairs[f"{part}_id"].to_numpy()
        raw = pairs.groupby(f"{part}_id")[part].first()  # id -> raw name
        texts = []
        for i in range(int(ids.max()) + 1):
            name = display_name(raw[i], names[f"{part}_overrides"], u2s)
            texts.append([t.format(**{part: name, "description": descriptions.get(raw[i], "")}) for t in pcfg[f"{part}_templates"]])
        parts.append(encode_prompts(clip_model, tokenizer, texts, device)[ids])
    return pair_emb, np.concatenate(parts, axis=1)


def fit_pipeline(x, y, trained_ids, target_ids, hidden, pair_emb, cond, cfg, epochs, on_epoch=None):
    """Steps 1-4 of the module docstring; returns the final classifier. x, y: real features and pair ids (tensors)."""
    device, n_pairs = x.device, len(pair_emb)
    trained = torch.zeros(n_pairs, dtype=torch.bool, device=device)
    trained[trained_ids] = True
    aux = fit_classifier(x, y, trained, pair_emb, cfg["aux_classifier"], cfg["aux_classifier"]["epochs"], tag="aux")
    g = train_gan(x, y, cond, aux, cfg["gan"])

    scfg = cfg["synthesis"]
    counts = {int(p): scfg["n_per_unseen"] for p in target_ids}
    if scfg["seen_min_per_pair"]:
        real = torch.bincount(y, minlength=n_pairs).cpu().numpy()
        counts.update({int(p): max(0, scfg["seen_min_per_pair"] - int(real[p])) for p in trained_ids})
    xs, ys = synthesise(g, cond, counts)
    print(f"synthesised {len(xs)} features for {sum(k > 0 for k in counts.values())} pairs", flush=True)

    allowed = ~torch.as_tensor(hidden, device=device)
    return fit_classifier(torch.cat([x, xs]), torch.cat([y, ys]), allowed, pair_emb, cfg["classifier"], epochs, on_epoch)


@torch.no_grad()
def pair_scores(clf, x) -> np.ndarray:
    return (clf(x) / CLIP_SCALE).float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == METHOD, f"{args.config} is a '{cfg['method']}' config"
    runtime, vcfg, ccfg = cfg["runtime"], cfg["validation"], cfg["classifier"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    device = runtime["device"]
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")
    seed_everything(cfg["seed"])

    df = read_metadata()
    pairs = pair_table(df)
    clip_model, preprocess, tokenizer = load_clip(cfg["model"], device)
    feats = {s: encode_images(clip_model, preprocess, df.loc[df["split"] == s, "file"], cfg["model"], runtime, cache_name=s)
             for s in ("train", "test_seen", "test_unseen")}
    pair_emb_np, cond_np = text_conditions(cfg, clip_model, tokenizer, pairs, device)
    pair_emb, cond = torch.as_tensor(pair_emb_np, device=device), torch.as_tensor(cond_np, device=device)
    print(f"condition: {cfg['condition']['parts']} -> {cond.shape[1]} dims", flush=True)
    train = df[df["split"] == "train"].copy()
    train["row"] = np.arange(len(train))
    x_train = torch.as_tensor(feats["train"], device=device)

    # ---- Stage 1: choose the classifier's epoch count and gamma on validation split(s) ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    real_unseen = pairs["is_unseen"].to_numpy()
    hidden = real_unseen if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, ccfg["max_epochs"], len(gammas)))
    history, holdouts = [], []
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        holdouts.append(pairs.loc[holdout, "pair"].tolist())
        print(f"validation split {k + 1}/{n_splits} (seed {split_seed}): fit {len(fit_df)} images, val-seen {len(val_seen)}, "
              f"pseudo-unseen {len(val_unseen)} ({len(holdout)} pairs)", flush=True)
        trained_ids = sorted(set(fit_df["pair_id"]))
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(trained_ids))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        x_vs, x_vu = x_train[val_seen["row"].to_numpy()], x_train[val_unseen["row"].to_numpy()]

        def validate(epoch, clf, k=k, val_pairs=val_pairs, y_vs=y_vs, y_vu=y_vu, x_vs=x_vs, x_vu=x_vu):
            s_seen, s_unseen = pair_scores(clf, x_vs), pair_scores(clf, x_vu)
            s_seen[:, hidden] = s_unseen[:, hidden] = -1e9
            results = [evaluate(s_seen, y_vs, s_unseen, y_vu, val_pairs, g, (1,)) for g in gammas]
            hm_grid[k, epoch - 1] = [r["harmonic_mean_top1"] for r in results]
            best = results[int(hm_grid[k, epoch - 1].argmax())]
            record = {"split": k + 1, "val_hm_g0": round(results[0]["harmonic_mean_top1"], 4), "val_hm": round(best["harmonic_mean_top1"], 4),
                      "gamma": best["gamma"], "val_seen": round(best["test_seen"]["top1"], 4), "val_unseen": round(best["test_unseen"]["top1"], 4)}
            history.append({"epoch": epoch, **record})
            return record

        seed_everything(split_seed)
        rows = fit_df["row"].to_numpy()
        fit_pipeline(x_train[rows], torch.as_tensor(fit_df["pair_id"].to_numpy(), device=device), trained_ids, holdout, hidden,
                     pair_emb, cond, cfg, ccfg["max_epochs"], validate)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    mean_grid = hm_grid.mean(axis=0)
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {
        "epoch": best_epoch,
        "gamma": gamma,
        "val_hm_mean": round(float(mean_grid[e, g]), 4),
        "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]],
        "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4),
        "best_gamma_per_split": [float(gammas[hm_grid[s, e].argmax()]) for s in range(n_splits)],
    }
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%} "
          f"(per split {best['val_hm_per_split']}; each split's own best gamma {best['best_gamma_per_split']})", flush=True)

    # ---- Stage 2: refit on all train, generating the real unseen pairs ----
    seed_everything(cfg["seed"])
    clf = fit_pipeline(x_train, torch.as_tensor(train["pair_id"].to_numpy(), device=device), sorted(set(train["pair_id"])),
                       np.flatnonzero(real_unseen), np.zeros(len(pairs), bool), pair_emb, cond, cfg, best_epoch)

    # ---- Test (once) ----
    scores, labels, files = {}, {}, {}
    for split in ("test_seen", "test_unseen"):
        sub = df[df["split"] == split]
        scores[split] = pair_scores(clf, torch.as_tensor(feats[split], device=device))
        labels[split], files[split] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(cfg["evaluation"]["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)
    # The classifier is linear in the normalised image embedding, so the app can score with ClipRunModel:
    # score = image_emb @ (scale / 100 * normalised weight).
    w = clf.scale / CLIP_SCALE * F.normalize(clf.weight.detach(), dim=-1)
    np.save(out / "pair_text_embeddings.npy", w.float().cpu().numpy())
    (out / "validation_split.json").write_text(json.dumps({"gammas": gammas.tolist(), "holdout_pairs_per_split": holdouts}, indent=1))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
