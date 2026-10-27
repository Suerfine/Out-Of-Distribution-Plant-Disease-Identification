"""Compositional prompts + LoRA on the CLIP image encoder, evaluated with the shared protocol.

Usage:  python -m train.train_lora --config config/vlm/v9_config.json

Same prompt model as train.py (one learnable vector per crop and per disease inside fixed sentences, frozen text
encoder), but the image encoder is no longer frozen: LoRA adapters (dpd/lora.py) in its last blocks are trained
jointly with the vectors. Image embeddings therefore change during training and cannot be cached, so images are
read from the 256 px resize cache with augmentation every epoch.

Stage 1: train on train minus the validation split(s); after every epoch score the validation images and sweep
gamma; pick the (epoch, gamma) with the best mean validation HM. Stage 2: refit a fresh model on all training
images for that many epochs and evaluate once on test.
"""

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dpd.clip_utils import clip_transforms, load_clip
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import evaluate, format_summary
from dpd.lora import apply_lora, lora_state
from dpd.prompt_learner import build_model
from dpd.report import save_test_outputs
from dpd.resize_cache import cached_name, ensure_resized
from dpd.splits import make_validation_split
from train.train import seed_everything, text_embeddings

class Images(Dataset):
    def __init__(self, df, image_dir, transform):
        self.files = [image_dir / cached_name(f) for f in df["file"]]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as img:
            return self.transform(img.convert("RGB")), i


def loader(df, image_dir, transform, cfg, shuffle):
    rt = cfg["runtime"]
    return DataLoader(Images(df, image_dir, transform), batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
                      num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=shuffle)


def build(cfg, pairs, device):
    """Fresh CLIP + learnable prompt vectors + LoRA adapters on the image encoder."""
    clip_model, _, tokenizer = load_clip(cfg["model"], device)
    clip_model.transformer.grad_checkpointing = cfg["runtime"]["grad_checkpointing"]
    model = build_model(cfg, clip_model, tokenizer, pairs).to(device)  # freezes all of CLIP
    lcfg = cfg["image_encoder"]["lora"]
    lora_params = apply_lora(clip_model.visual, lcfg["last_n_blocks"], lcfg["rank"], lcfg["alpha"], lcfg["targets"])
    return model, lora_params


@torch.no_grad()
def encode(model, dl, device, amp) -> np.ndarray:
    model.eval()
    feats = []
    for x, _ in dl:
        with torch.autocast(device, enabled=amp):
            f = model.clip.encode_image(x.to(device, non_blocking=True))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy()


@torch.no_grad()
def pair_scores(model, img_feats, amp) -> np.ndarray:
    model.eval()
    t = text_embeddings(model, "pairs", amp).float().cpu().numpy()
    return img_feats @ t.T


def fit(model, lora_params, train_df, trained_pairs, epochs, cfg, image_dir, tfs, on_epoch_end=None, log=None):
    rt, tcfg = cfg["runtime"], cfg["training"]
    device, amp = rt["device"], rt["amp"]
    opt = torch.optim.AdamW([
        {"params": [model.crop_vecs, model.disease_vecs], "lr": tcfg["lr"]},
        {"params": lora_params, "lr": cfg["image_encoder"]["lora"]["lr"]},
    ], weight_decay=tcfg["weight_decay"])
    dl = loader(train_df, image_dir, tfs[0], cfg, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    scale, lw = tcfg["logit_scale"], tcfg["loss_weights"]
    trained = torch.as_tensor(trained_pairs, device=device)
    col = torch.full((int(trained.max()) + 1,), -1, device=device, dtype=torch.long)
    col[trained] = torch.arange(len(trained), device=device)
    y_pair = torch.as_tensor(train_df["pair_id"].to_numpy(), device=device)
    y_crop = torch.as_tensor(train_df["crop_id"].to_numpy(), device=device)
    y_dis = torch.as_tensor(train_df["disease_id"].to_numpy(), device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, totals, n_batches = time.time(), {"loss": 0.0, "pair": 0.0, "crop": 0.0, "disease": 0.0}, 0
        for x, idx in dl:
            idx = idx.to(device)
            with torch.autocast(device, enabled=amp):
                v = F.normalize(model.clip.encode_image(x.to(device, non_blocking=True)).float(), dim=-1)
            losses = {"pair": F.cross_entropy(scale * v @ text_embeddings(model, "pairs", amp)[trained].T, col[y_pair[idx]])}
            if lw["crop"] > 0:
                losses["crop"] = F.cross_entropy(scale * v @ text_embeddings(model, "crops", amp).T, y_crop[idx])
            if lw["disease"] > 0:
                losses["disease"] = F.cross_entropy(scale * v @ text_embeddings(model, "diseases", amp).T, y_dis[idx])
            loss = sum(lw[k] * l for k, l in losses.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            n_batches += 1
            totals["loss"] += loss.item()
            for k, l in losses.items():
                totals[k] += l.item()
        rec = {"epoch": epoch, **{k: round(v / n_batches, 4) for k, v in totals.items()}, "sec": round(time.time() - t0)}
        if on_epoch_end:
            rec.update(on_epoch_end(epoch))
        if log:
            log(rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "clip_prompt_lora", f"{args.config} is a '{cfg['method']}' config"
    rt, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    device, amp = rt["device"], rt["amp"]
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Config asks for cuda but torch.cuda.is_available() is False.")
    out = output_dir(cfg)
    shutil.copy(args.config, out / "config.json")

    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = clip_transforms(cfg["input"], cfg["augmentation"])
    train = df[df["split"] == "train"]
    history = []

    def logger(stage):
        def log(rec):
            rec = {"stage": stage, **rec}
            history.append(rec)
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        return log

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
        trained_ids = sorted(set(fit_df["pair_id"]))
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(trained_ids))
        val_df = pd.concat([val_seen, val_unseen])
        n_vs, y_vs, y_vu = len(val_seen), val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()
        val_dl = loader(val_df, image_dir, tfs[1], cfg, shuffle=False)

        seed_everything(split_seed)
        model, lora_params = build(cfg, pairs, device)

        def validate(epoch, k=k):
            s = pair_scores(model, encode(model, val_dl, device, amp), amp)
            s[:, hidden] = -1e9
            res = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,)) for g in gammas]
            hm_grid[k, epoch - 1] = [r["harmonic_mean_top1"] for r in res]
            best = res[int(hm_grid[k, epoch - 1].argmax())]
            return {"val_hm_g0": round(res[0]["harmonic_mean_top1"], 4), "val_hm": round(best["harmonic_mean_top1"], 4),
                    "gamma": best["gamma"], "val_seen": round(best["test_seen"]["top1"], 4),
                    "val_unseen": round(best["test_unseen"]["top1"], 4)}

        fit(model, lora_params, fit_df, trained_ids, tcfg["max_epochs"], cfg, image_dir, tfs, validate, logger(f"split{k + 1}"))
        del model, lora_params
        torch.cuda.empty_cache()

    mean_grid = hm_grid.mean(axis=0)
    e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]], "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4)}
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    seed_everything(cfg["seed"])
    model, lora_params = build(cfg, pairs, device)
    fit(model, lora_params, train, sorted(set(train["pair_id"])), best_epoch, cfg, image_dir, tfs, log=logger("final"))

    scores, labels, files = {}, {}, {}
    for s in ("test_seen", "test_unseen"):
        sub = df[df["split"] == s]
        scores[s] = pair_scores(model, encode(model, loader(sub, image_dir, tfs[1], cfg, shuffle=False), device, amp), amp)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid)
    with torch.no_grad():
        np.save(out / "pair_text_embeddings.npy", text_embeddings(model, "pairs", amp).float().cpu().numpy())
    torch.save({"crop_vecs": model.crop_vecs.detach().cpu(), "disease_vecs": model.disease_vecs.detach().cpu(), "gamma": gamma},
               out / "prompt_vectors.pt")
    torch.save(lora_state(model.clip.visual), out / "lora_image_encoder.pt")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
