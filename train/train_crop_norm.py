"""Crop-normalised disease prototypes (dpd/crop_norm.py), evaluated with the same protocol as the CNN baselines.

Usage:  python -m train.train_crop_norm --config config/cropnorm/aniq_cropnorm_v1.json

method "cnn_crop_norm": built on the independent baseline. The crop network is the baseline's crop CNN (same code,
settings and seeds, via train/train_cnn.py); the disease network classifies the image feature after removing each
candidate crop's transform, so pair scores are crop-conditional:
    score(c, d) = 2 * (w * log P(c | x) + (1 - w) * log P(d | x normalised for c))
With crop_norm.shared_backbone one network has both the crop head and the normalised disease branch (compare with the
multi-task baseline).

Stage 1 and stage 2 are those of train_cnn.py: choose epoch, crop weight, tau and gamma on the mean validation HM over
the splits, then refit on all training images and evaluate once on test. Both networks are scored at the same epoch.
The crop network is identical in every config with the same base settings, so its per-epoch validation log-probs and
its final fit are cached in runs/cache/crop_net/<key>/ and shared between configs.
"""

import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dpd.cnn import ImageLabels, build_transforms, log_priors
from dpd.config import load_config, output_dir
from dpd.crop_norm import CropNormNet, CrossCropBatchSampler, compose_pair_scores, crop_diversity, losses
from dpd.data import PROJECT_ROOT, pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary, hm_top1_over_gammas
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from train import train_cnn as tc

METHOD = "cnn_crop_norm"
CROP_CACHE = PROJECT_ROOT / "runs" / "cache" / "crop_net"
SPLIT_KEYS = ("n_holdout_pairs", "min_holdout_images", "val_fraction", "exclude_holdout_diseases", "max_images_per_holdout_pair")


def _doc_free(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def crop_net_cache(cfg) -> "Path":
    """Cache folder for the crop network, keyed by every setting train_cnn.train_network and the splits read."""
    key = {
        "seed": cfg["seed"], "model": _doc_free(cfg["model"]), "input": _doc_free(cfg["input"]),
        "augmentation": _doc_free(cfg["augmentation"]), "optimizer": _doc_free(cfg["optimizer"]),
        "loss": {"label_smoothing": cfg["loss"]["label_smoothing"], "crop": cfg["loss"]["weights"]["crop"]},
        "batch_size": cfg["training"]["batch_size"], "max_epochs": cfg["training"]["max_epochs"],
        "split": {k: cfg["validation"].get(k) for k in SPLIT_KEYS}, "amp": cfg["runtime"]["amp"],
    }
    d = CROP_CACHE / hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    (d / "key.json").write_text(json.dumps(key, indent=2))
    return d


def crop_val_logp(cfg, cache, split_seed, fit_df, eval_loader, image_dir, tfs, log) -> np.ndarray:
    """Crop log-probs of the validation images after every epoch (epochs x images x crops), cached."""
    path = cache / f"val_split_seed{split_seed}.npy"
    if path.exists():
        print(f"crop network: cached validation log-probs {path}", flush=True)
        return np.load(path)
    tc.seed_everything(split_seed)
    model, per_epoch = tc.train_network(["crop"], fit_df, cfg["training"]["max_epochs"], cfg, image_dir, tfs, eval_loader, log)
    del model
    torch.cuda.empty_cache()
    out = np.stack([p["crop"] for p in per_epoch])
    np.save(path, out)
    return out


def crop_final(cfg, cache, epochs, train, test_loaders, image_dir, tfs, out, log) -> dict[str, np.ndarray]:
    """Crop network refit on all training images for `epochs`; copies it to the run folder, returns test log-probs."""
    weights, logp_path = cache / f"final_e{epochs}_crop_net.pt", cache / f"final_e{epochs}_test_logp.npz"
    if not (weights.exists() and logp_path.exists()):
        tc.seed_everything(cfg["seed"])
        model, _ = tc.train_network(["crop"], train, epochs, cfg, image_dir, tfs, log=log)
        torch.save(model.state_dict(), weights)
        np.savez(logp_path, **{s: tc.predict(model, l, cfg["runtime"]["device"], cfg["runtime"]["amp"])["crop"]
                               for s, l in test_loaders.items()})
        del model
        torch.cuda.empty_cache()
    else:
        print(f"crop network: cached final fit {weights}", flush=True)
    shutil.copy(weights, out / "crop_net.pt")
    return dict(np.load(logp_path))


def train_loader(df, image_dir, transform, cfg):
    cn, rt = cfg["crop_norm"], cfg["runtime"]
    if not cn["lambda_consistency"]:
        return tc.make_loader(df, image_dir, transform, cfg, shuffle=True)
    sampler = CrossCropBatchSampler(df["crop_id"].to_numpy(), df["disease_id"].to_numpy(), cfg["training"]["batch_size"],
                                    cn["consistency_pairs_per_batch"])
    return DataLoader(ImageLabels(df, image_dir, transform), batch_sampler=sampler, num_workers=rt["num_workers"],
                      pin_memory=True, persistent_workers=rt["num_workers"] > 0)


@torch.no_grad()
def predict(model, loader, device, amp) -> tuple[np.ndarray | None, np.ndarray]:
    """(crop log-probs or None, disease logits under every crop's normalisation: images x crops x diseases)."""
    model.eval()
    crop, dis = [], []
    for x, _, _ in loader:
        with torch.autocast(device, enabled=amp):
            f = model.backbone(x.to(device, non_blocking=True))
            z = model.proj(f)
        dis.append(model.disease_logits(model.normalise(z.float())).cpu())
        if model.crop_head is not None:
            crop.append(F.log_softmax(model.crop_head(f.float()), dim=-1).cpu())
    return (torch.cat(crop).numpy() if crop else None), torch.cat(dis).numpy()


def train_crop_norm(train_df, epochs, cfg, image_dir, tfs, on_epoch=None, log=None):
    """Train the crop-normalised disease network; on_epoch(epoch index, model) runs after every epoch."""
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    mcfg, ocfg, lcfg, cn = cfg["model"], cfg["optimizer"], cfg["loss"], cfg["crop_norm"]
    model = CropNormNet(mcfg["backbone"], mcfg["pretrained"], cn, mcfg["head_dropout"], crop_diversity(train_df)).to(device)
    no_decay = {id(p) for p in model.no_decay_parameters()}
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and id(p) not in no_decay]
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]},
            {"params": heads, "lr": ocfg["lr_head"]},
            {"params": [p for p in model.no_decay_parameters() if p.requires_grad], "lr": ocfg["lr_head"], "weight_decay": 0.0},
        ],
        weight_decay=ocfg["weight_decay"],
    )
    loader = train_loader(train_df, image_dir, tfs[0], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, n, terms, correct = time.time(), 0, {}, {"disease": 0, "crop": 0}
        for x, y_crop, y_dis in loader:
            x, y_crop, y_dis = x.to(device, non_blocking=True), y_crop.to(device), y_dis.to(device)
            with torch.autocast(device, enabled=amp):
                f = model.backbone(x)
                z = model.proj(f)
            z = z.float()
            model.update_centroids(z.detach(), y_crop)
            parts, logits, crop_logits = losses(model, f, z, y_crop, y_dis, cn, lcfg)
            loss = sum(parts.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += len(x)
            for k, v in parts.items():
                terms[k] = terms.get(k, 0.0) + v.item() * len(x)
            correct["disease"] += (logits.argmax(1) == y_dis).sum().item()
            if crop_logits is not None:
                correct["crop"] += (crop_logits.argmax(1) == y_crop).sum().item()
        rec = {"epoch": epoch, **{f"loss_{k}": round(v / n, 4) for k, v in terms.items()},
               "train_disease_acc": round(correct["disease"] / n, 4), "scale": round(model.log_scale.exp().item(), 2)}
        if model.crop_head is not None:
            rec["train_crop_acc"] = round(correct["crop"] / n, 4)
        rec["sec"] = round(time.time() - t0)
        if on_epoch is not None:
            on_epoch(epoch - 1, model)
        if log:
            log(rec)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == METHOD, f"{args.config} is a '{cfg['method']}' config"
    runtime, tcfg, vcfg, ev, cn = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"], cfg["crop_norm"]
    shared = cn["shared_backbone"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")
    cache = None if shared else crop_net_cache(cfg)

    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    train = df[df["split"] == "train"]
    history = []

    def logger(prefix):
        def log(rec):
            rec = {"stage": prefix, **rec}
            history.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        return log

    # ---- Stage 1: choose epoch, scoring (crop weight, logit adjustment) and gamma on validation ----
    gammas = np.round(np.arange(*vcfg["gamma_grid"]), 6)
    ccfg = cfg.get("composition", {})
    weights = np.round(np.arange(*ccfg["crop_weight_grid"]), 6) if "crop_weight_grid" in ccfg else np.array([0.5])
    taus = np.round(np.arange(*ccfg["tau_grid"]), 6) if "tau_grid" in ccfg else np.array([0.0])
    hidden = pairs["is_unseen"].to_numpy() if vcfg["exclude_real_unseen_from_label_space"] else np.zeros(len(pairs), bool)
    n_splits = vcfg.get("n_splits", 1)
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(weights), len(taus), len(gammas)))
    for k in range(n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"],
        )
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        val_df = pd.concat([val_seen, val_unseen])
        n_vs = len(val_seen)
        eval_loader = tc.make_loader(val_df, image_dir, tfs[1], cfg, shuffle=False)
        crop_logp = None if shared else crop_val_logp(cfg, cache, split_seed, fit_df, eval_loader, image_dir, tfs,
                                                      logger(f"split{k + 1}/crop_net"))
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        prior = log_priors(fit_df)  # the frequencies the validation networks were trained on

        def on_epoch(e, model, k=k, crop_logp=crop_logp, eval_loader=eval_loader, val_pairs=val_pairs, y_vs=y_vs,
                     y_vu=y_vu, n_vs=n_vs, prior=prior):
            own_crop, dis_logits = predict(model, eval_loader, runtime["device"], runtime["amp"])
            lc = own_crop if shared else crop_logp[e]
            for i, w in enumerate(weights):
                for j, t in enumerate(taus):
                    s = compose_pair_scores(lc, dis_logits, pairs, w, t, prior)
                    s[:, hidden] = -1e9
                    hm_grid[k, e, i, j] = hm_top1_over_gammas(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, gammas)
            i, j, g = np.unravel_index(hm_grid[k, e].argmax(), hm_grid[k, e].shape)
            print(f"split {k + 1} epoch {e + 1}: best val HM {hm_grid[k, e, i, j, g]:.2%} at crop weight {weights[i]}, "
                  f"tau {taus[j]}, gamma {gammas[g]}", flush=True)

        tc.seed_everything(split_seed)
        model = train_crop_norm(fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, on_epoch, logger(f"split{k + 1}/disease_net"))
        del model
        torch.cuda.empty_cache()
    mean_grid = hm_grid.mean(axis=0)
    e, i, j, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, crop_weight, tau, gamma = int(e) + 1, float(weights[i]), float(taus[j]), float(gammas[g])
    best = {"epoch": best_epoch, "crop_weight": crop_weight, "tau": tau, "gamma": gamma,
            "val_hm_mean": round(float(mean_grid[e, i, j, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, i, j, g]],
            "val_hm_g0_mean": round(float(mean_grid[e, i, j, 0]), 4)}
    print(f"selected epoch {best_epoch}, crop weight {crop_weight}, tau {tau}, gamma {gamma}: "
          f"mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    test_loaders = {s: tc.make_loader(df[df["split"] == s], image_dir, tfs[1], cfg, shuffle=False) for s in ("test_seen", "test_unseen")}
    if not shared:
        test_crop = crop_final(cfg, cache, best_epoch, train, test_loaders, image_dir, tfs, out, logger("final/crop_net"))
    tc.seed_everything(cfg["seed"])
    model = train_crop_norm(train, best_epoch, cfg, image_dir, tfs, log=logger("final/disease_net"))
    torch.save(model.state_dict(), out / "disease_net.pt")

    scores, labels, files = {}, {}, {}
    prior = log_priors(train)
    for s, loader in test_loaders.items():
        own_crop, dis_logits = predict(model, loader, runtime["device"], runtime["amp"])
        sub = df[df["split"] == s]
        scores[s] = compose_pair_scores(own_crop if shared else test_crop[s], dis_logits, pairs, crop_weight, tau, prior)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, crop_weight=crop_weight, tau=tau, val=best,
                  harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"],
                  crop_net_cache=None if shared else cache.name)
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))
    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)  # splits x epochs x crop weights x taus x gammas
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
