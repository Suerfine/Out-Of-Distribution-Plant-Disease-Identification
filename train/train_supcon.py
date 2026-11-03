"""Supervised contrastive learning (SupCon) on crop and disease, evaluated with the shared 333-pair protocol.

Usage (from the repository root):
    python -m train.train_supcon --config config/contrastive/supcon_resnet50.json
    python -m train.train_supcon --config config/contrastive/supcon_resnet50.json --pilot    # tiny smoke run

method "cnn_supcon": a ResNet-50 with a crop projection head and a disease projection head, each trained with the
SupCon loss on its own label (dpd/supcon.py). Classification uses class prototypes (mean training embedding per crop /
disease); pair score = log P(crop) + log P(disease), so unseen pairs can be predicted.

Stage 1: train on train minus a validation split (dpd/splits.py, same as train_cnn.py); after every epoch, build the
prototypes from the fit images, score the validation images and sweep gamma; pick the (epoch, gamma) with the best
validation HM (mean over splits).
Stage 2: refit on all training images for the chosen epoch count, evaluate once on test.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpd.cnn import HEAD_SIZES, ImageLabels, build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from dpd.supcon import (HEADS, ContrastiveNet, TwoViewImageLabels, class_prototypes, embed, fit_linear_heads,
                        linear_log_probs, log_probs, supcon_loss)
from train.train_cnn import make_loader, seed_everything

PILOT_IMAGES_PER_PAIR, PILOT_TEST_IMAGES = 6, 400


def apply_pilot(cfg: dict) -> dict:
    """Shrink a config for a quick end-to-end check: 2 epochs, one validation split, small holdout, own output folder."""
    cfg = {**cfg, "version": "_pilot", "output_dir": "runs/contrastive/_pilot"}
    cfg["training"] = {**cfg["training"], "max_epochs": 2}
    cfg["validation"] = {**cfg["validation"], "n_splits": 1, "n_holdout_pairs": 10, "min_holdout_images": 4,
                         "max_images_per_holdout_pair": 6, "val_fraction": 0.35, "gamma_grid": [0.0, 10.0, 0.5]}
    return cfg


def make_two_view_loader(df, image_dir, transform, cfg):
    rt = cfg["runtime"]
    return DataLoader(
        TwoViewImageLabels(df, image_dir, transform), batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=True,
    )


def train_network(train_df, epochs, cfg, image_dir, tfs, on_epoch=None, log=None):
    """Train with SupCon; `on_epoch(model)` runs after every epoch and its results are returned as a list."""
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    mcfg, ocfg, lcfg = cfg["model"], cfg["optimizer"], cfg["loss"]
    model = ContrastiveNet(mcfg["backbone"], mcfg["pretrained"], mcfg["proj_dim"], mcfg.get("proj_hidden"),
                           mcfg.get("ce_dropout"), mcfg.get("drop_path_rate", 0.0)).to(device)
    if cfg["runtime"].get("grad_checkpointing") and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)  # same maths, less memory: recompute activations in the backward pass
    head_params = list(model.proj.parameters()) + (list(model.cls.parameters()) if model.cls is not None else [])
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
            {"params": head_params, "lr": ocfg["lr_head"]},
        ],
        weight_decay=ocfg["weight_decay"],
    )
    loader = make_two_view_loader(train_df, image_dir, tfs[0], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, tot, parts, n = time.time(), 0.0, {h: 0.0 for h in HEADS}, 0
        for v1, v2, y_crop, y_dis in loader:
            x = torch.cat([v1, v2]).to(device, non_blocking=True)
            labels = {"crop": torch.cat([y_crop, y_crop]).to(device), "disease": torch.cat([y_dis, y_dis]).to(device)}
            with torch.autocast(device, enabled=amp):
                f, z = model.encode(x)
                logits = model.classify(f) if model.cls is not None else None
            losses = {h: supcon_loss(z[h], labels[h], lcfg["temperature"]) for h in HEADS}
            loss = sum(lcfg["weights"][h] * losses[h] for h in HEADS)
            if logits is not None:  # cross-entropy on both views, as in the baselines
                loss = loss + lcfg["ce_weight"] * sum(
                    F.cross_entropy(logits[h].float(), labels[h], label_smoothing=lcfg["label_smoothing"]) for h in HEADS
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(v1)
            n += len(v1)
            for h in HEADS:
                parts[h] += losses[h].item() * len(v1)
        rec = {"epoch": epoch, "loss": round(tot / n, 4), **{f"loss_{h}": round(parts[h] / n, 4) for h in HEADS},
               "sec": round(time.time() - t0)}
        if on_epoch is not None:
            per_epoch.append(on_epoch(model))
        if log:
            log(rec)
    return model, per_epoch


def make_scorer(emb_fit, fit_df, cfg, pairs):
    """Fit the scoring step on the fit images; returns (scorer, classifier). `scorer(embeddings)` gives the
    (images x 333) pair scores. scoring.mode "prototype" (default) uses class prototypes of the projection embeddings;
    "classifier" trains a linear softmax classifier per head on the frozen backbone features (the classifier is None
    for prototypes). Built once per fit, so every target split is scored by the same classifier."""
    scfg = cfg["scoring"]
    labels = {h: fit_df[f"{h}_id"].to_numpy() for h in HEADS}
    if scfg.get("mode", "prototype") == "heads":  # the cross-entropy heads trained with the model
        return lambda emb: compose_pair_scores(emb["logp_crop"], emb["logp_disease"], pairs), None
    if scfg.get("mode", "prototype") == "classifier":
        torch.manual_seed(cfg["seed"])
        clf = fit_linear_heads(emb_fit["feat"], labels, scfg["classifier"], cfg["runtime"]["device"])
        return lambda emb: compose_pair_scores(*(linear_log_probs(emb["feat"], clf, h) for h in HEADS), pairs), clf
    protos = {h: class_prototypes(emb_fit[h], labels[h], HEAD_SIZES[h]) for h in HEADS}
    return lambda emb: compose_pair_scores(*(log_probs(emb[h], *protos[h], scfg["temperature"]) for h in HEADS), pairs), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot", action="store_true", help="tiny end-to-end check: few images, 2 epochs, 1 validation split")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "cnn_supcon", f"{args.config} is a '{cfg['method']}' config"
    if args.pilot:
        cfg = apply_pilot(cfg)
    runtime, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    temperature = cfg["scoring"]["temperature"]
    device, amp = runtime["device"], runtime["amp"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    train = df[df["split"] == "train"]
    if args.pilot:
        train = train.sample(frac=1, random_state=cfg["seed"]).groupby("pair_id").head(PILOT_IMAGES_PER_PAIR)
        print(f"PILOT: {len(train)} training images, {PILOT_TEST_IMAGES} per test split", flush=True)
    history = []

    def logger(prefix):
        def log(rec):
            rec = {"stage": prefix, **rec}
            history.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        return log

    def eval_loader(frame):
        return DataLoader(ImageLabels(frame, image_dir, tfs[1]), batch_size=128, shuffle=False,
                          num_workers=runtime["num_workers"], pin_memory=True)

    # ---- Stage 1: choose epoch and gamma on validation ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(gammas)))
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        val_df = pd.concat([val_seen, val_unseen])
        n_vs = len(val_seen)
        fit_loader, val_loader = eval_loader(fit_df), eval_loader(val_df)
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()

        def on_epoch(model):
            scorer, _ = make_scorer(embed(model, fit_loader, device, amp), fit_df, cfg, pairs)
            return scorer(embed(model, val_loader, device, amp))

        seed_everything(split_seed)
        _, val_scores = train_network(fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, on_epoch, logger(f"split{k + 1}"))
        for e, s in enumerate(val_scores):
            s[:, hidden] = -1e9
            hm_grid[k, e] = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
            print(f"split {k + 1} epoch {e + 1}: val HM {hm_grid[k, e, 0]:.2%} at gamma 0, best {hm_grid[k, e].max():.2%} "
                  f"at gamma {gammas[hm_grid[k, e].argmax()]}", flush=True)
    mean_grid = hm_grid.mean(axis=0)
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]], "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4)}
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    seed_everything(cfg["seed"])
    model, _ = train_network(train, best_epoch, cfg, image_dir, tfs, log=logger("final"))
    torch.save(model.state_dict(), out / "supcon_net.pt")
    emb_train = embed(model, eval_loader(train), device, amp)
    protos = {}
    for h in HEADS:
        protos[f"{h}_prototypes"], protos[f"{h}_valid"] = class_prototypes(emb_train[h], train[f"{h}_id"].to_numpy(), HEAD_SIZES[h])
    np.savez_compressed(out / "prototypes.npz", temperature=temperature, **protos)
    scorer, clf = make_scorer(emb_train, train, cfg, pairs)
    if clf is not None:
        torch.save({"mean": clf["mean"].cpu(), "std": clf["std"].cpu(),
                    **{h: clf["heads"][h][0].state_dict() for h in HEADS}}, out / "classifier.pt")

    scores, labels, files = {}, {}, {}
    for s in ("test_seen", "test_unseen"):
        sub = df[df["split"] == s]
        if args.pilot:
            sub = sub.sample(n=min(PILOT_TEST_IMAGES, len(sub)), random_state=cfg["seed"])
        scores[s] = scorer(embed(model, eval_loader(sub), device, amp))
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
