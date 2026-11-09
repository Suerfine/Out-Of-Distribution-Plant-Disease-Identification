"""Compositional prompts + LoRA on the CLIP image encoder, evaluated with the shared protocol.

Usage:  python -m train.train_lora --config config/vlm/v5_lora_config.json

Same prompt model as train.py (one learnable vector per crop and per disease inside fixed sentences, frozen text
encoder), but the image encoder is no longer frozen: LoRA adapters (dpd/lora.py) in its last blocks are trained
jointly with the vectors. Image embeddings therefore change during training and cannot be cached, so images are
read from the 256 px resize cache with augmentation every epoch.

Options (all in the config, each off by default so older configs behave exactly as before):
  model.image_size            input resolution, e.g. 336 (needs input.resize_cache >= that size)
  runtime.grad_checkpointing_image   recompute image-encoder activations in backward to save memory
  evaluation.tta              test-time augmentation views averaged at validation and test time
  training.sampling           {"type": "class_balanced", "power": p}: sample images with weight n_pair^-p
  model.patch_disease         {"top_k": k, "lambda": l}: add patch-level disease scores (dpd/patch_disease.py); the
                              disease loss is then computed on the patch scores
  training.loss_weights.within_crop   weight of an extra pair loss ranked only against pairs of the image's own
                              crop (hard negatives: e.g. tomato early blight vs other tomato diseases)
  evaluation.posthoc          {"lora_scale_grid": [...], "crop_weight_grid": [...], "disease_weight_grid": [...]}:
                              prediction-time knobs chosen on validation with the epoch and gamma (training is
                              unchanged). LoRA scale s uses W + s * delta (dpd/lora.py); crop / disease weights fuse the
                              crop and disease prompt scores into each pair's score (dpd/fusion.py). The neutral point
                              (s 1, weights 0) is plain scoring, so the run also reports a paired bootstrap of the
                              selected knobs against it on the same validation images.
  training.loss_weights.image_consistency   weight of an L1 loss between the LoRA CLS embedding and frozen CLIP's
                              embedding of the same augmented image (LoRA scale 0, no grad; PromptSRC / CoPrompt)
  training.loss_weights.text_anchor   weight of an L1 loss between the current pair / crop / disease text embeddings
                              and their values at initialisation (KgCoOp-style)
  training.weight_average     {"type": "ema", "decay": d}: exponential moving average of the prompt vectors and LoRA
                              weights, updated every step (decay warmed up as min(d, (1+t)/(10+t))); validation and the
                              final model use the averaged weights
  training.loss_weights.ssl   weight of a local-to-global self-supervised loss (no labels): each training image also
                              gets a small "local" crop (augmentation.ssl_local: scale, optional color_jitter / flips,
                              batch_fraction = share of the batch that gets a local view), and 1 - cos(local CLS,
                              stop-grad global CLS) is minimised, so the encoder must recognise the image from a small
                              region (lesion-level detail). Training only; validation, test and inference are unchanged
  training.alternating_primitives   {"period_epochs": p, "first": "disease" | "crop"}: DRPT-style schedule (Lu et
                              al. 2023). Only one of the disease / crop vector sets is updated at a time, swapping every
                              p epochs (LoRA always trains), so the two sets do not drag each other into one entangled
                              optimum. Training only
  model.primitive_branches    {"lr": lr}: Troika-style crop and disease visual branches (dpd/primitive_branches.py): the
                              crop / disease losses use their own residual linear map of the CLS embedding, and primitive
                              fusion (evaluation.posthoc crop / disease weights) scores with those branch features

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dpd.clip_utils import TTA_VIEWS, TwoViews, clip_transforms, load_clip, ssl_local_transform
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.evaluate import calibrate, evaluate, format_summary, hm_top1_over_gammas
from dpd.fusion import bootstrap_hm_diff, fuse
from dpd.lora import apply_lora, lora_state, set_lora_scale
from dpd.patch_disease import combine, image_features
from dpd.primitive_branches import branch_features, build_branches
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
    sampling = cfg["training"].get("sampling", {"type": "uniform"})
    sampler = None
    if shuffle and sampling["type"] == "class_balanced":
        # Weight each image by n_pair^-power: power 1 = every pair equally often, 0.5 = square-root balancing.
        n = df["pair_id"].map(df["pair_id"].value_counts()).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(n ** -sampling["power"], dtype=torch.double), num_samples=len(df), replacement=True)
    elif sampling["type"] not in ("uniform", "class_balanced"):
        raise ValueError(f"unknown sampling {sampling['type']!r}")
    return DataLoader(Images(df, image_dir, transform), batch_size=cfg["training"]["batch_size"],
                      shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=shuffle)


def build(cfg, pairs, device):
    """Fresh CLIP + learnable prompt vectors + LoRA adapters on the image encoder."""
    clip_model, _, tokenizer = load_clip(cfg["model"], device)
    clip_model.transformer.grad_checkpointing = cfg["runtime"]["grad_checkpointing"]
    clip_model.visual.transformer.grad_checkpointing = cfg["runtime"].get("grad_checkpointing_image", False)
    model = build_model(cfg, clip_model, tokenizer, pairs).to(device)  # freezes all of CLIP
    lcfg = cfg["image_encoder"]["lora"]
    lora_params = apply_lora(clip_model.visual, lcfg["last_n_blocks"], lcfg["rank"], lcfg["alpha"], lcfg["targets"])
    model.patch = cfg["model"].get("patch_disease")
    model.pair_disease = torch.as_tensor(pairs["disease_id"].to_numpy(), device=device)
    model.pair_crop = torch.as_tensor(pairs["crop_id"].to_numpy(), device=device)
    pb = cfg["model"].get("primitive_branches")
    model.branches = build_branches(clip_model.visual.output_dim).to(device) if pb else None
    return model, lora_params


def disease_text(model, amp):
    """Disease text embeddings for patch scoring, or None when the run scores with CLS only."""
    return text_embeddings(model, "diseases", amp).float() if model.patch else None


@torch.no_grad()
def encode(model, dl, device, amp, tta=("identity",)):
    """(L2-normalised image embeddings, per-disease patch scores or None); with several TTA views, the mean of the
    views' normalised embeddings (and of their patch scores)."""
    model.eval()
    dt = disease_text(model, amp)
    top_k = model.patch["top_k"] if model.patch else None
    feats, patch = [], []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        views = []
        for v in tta:
            with torch.autocast(device, enabled=amp):
                views.append(image_features(model.clip, TTA_VIEWS[v](x), dt, top_k))
        feats.append(F.normalize(torch.stack([c for c, _ in views]).mean(0), dim=-1).cpu())
        if dt is not None:
            patch.append(torch.stack([p for _, p in views]).mean(0).cpu())
    return torch.cat(feats).numpy(), (torch.cat(patch).numpy() if patch else None)


@torch.no_grad()
def pair_scores(model, img_feats, amp) -> np.ndarray:
    """Pair scores from encode()'s output (CLS cosine, plus lambda * patch disease score when enabled)."""
    model.eval()
    cls, patch = img_feats
    t = text_embeddings(model, "pairs", amp).float().cpu().numpy()
    lam = model.patch["lambda"] if model.patch else 0.0
    return combine(cls @ t.T, patch, model.pair_disease.cpu().numpy(), lam)


@torch.no_grad()
def primitive_scores(model, img_feats, amp):
    """Cosines of the image embeddings with the crop and with the disease prompts (for primitive fusion)."""
    model.eval()
    cls = img_feats[0]
    br = model.branches
    return (branch_features(br, "crop", cls) @ text_embeddings(model, "crops", amp).float().cpu().numpy().T,
            branch_features(br, "disease", cls) @ text_embeddings(model, "diseases", amp).float().cpu().numpy().T)


def posthoc_grids(ev):
    """(LoRA scales, crop weights, disease weights) from evaluation.posthoc; the neutral values come first, so exact
    ties keep plain scoring."""
    ph = ev.get("posthoc", {})
    s = sorted(set(ph.get("lora_scale_grid", [1.0])), reverse=True)
    a = sorted(set(ph.get("crop_weight_grid", [0.0])))
    b = sorted(set(ph.get("disease_weight_grid", [0.0])))
    assert s[0] == 1.0 and a[0] == 0.0 and b[0] == 0.0, "posthoc grids must include scale 1 (as max) and weights 0"
    return s, a, b


def knobs_from_metrics(metrics: dict) -> dict:
    """The post-hoc knobs a finished run selected (plain scoring for runs without them)."""
    return {"lora_scale": metrics.get("lora_scale", 1.0), "crop_weight": metrics.get("fusion_crop_weight", 0.0),
            "disease_weight": metrics.get("fusion_disease_weight", 0.0)}


def encode_scaled(model, dl, device, amp, tta, s):
    """encode() with the LoRA update scaled by s; always restores the full update afterwards."""
    set_lora_scale(model.clip.visual, s)
    try:
        return encode(model, dl, device, amp, tta)
    finally:
        set_lora_scale(model.clip.visual, 1.0)


def score_all(model, dl, device, amp, tta, knobs):
    """Pair scores (images x 333) with the post-hoc knobs applied; knobs {} = plain scoring."""
    s, a, b = knobs.get("lora_scale", 1.0), knobs.get("crop_weight", 0.0), knobs.get("disease_weight", 0.0)
    feats = encode_scaled(model, dl, device, amp, tta, s)
    pair = pair_scores(model, feats, amp)
    if not a and not b:
        return pair
    crop, dis = primitive_scores(model, feats, amp)
    return fuse(pair, crop, dis, model.pair_crop.cpu().numpy(), model.pair_disease.cpu().numpy(), a, b)


def within_crop_loss(pair_logits, target_col, col_crop, y_crop):
    """Cross-entropy over only the trained pairs that share the image's crop (its true pair is always among them)."""
    other_crop = col_crop[None, :] != y_crop[:, None]
    return F.cross_entropy(pair_logits.masked_fill(other_crop, float("-inf")), target_col)


class WeightEMA:
    """Exponential moving average of a list of parameters; swap() exchanges the live and averaged values in place."""

    def __init__(self, params, decay):
        self.params, self.decay, self.t = list(params), decay, 0
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p, 1 - d)

    @torch.no_grad()
    def swap(self):
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)


@torch.no_grad()
def frozen_features(model, x, amp):
    """Frozen CLIP's normalised CLS embedding of x (LoRA update switched off), for the image-consistency loss."""
    set_lora_scale(model.clip.visual, 0.0)
    try:
        with torch.autocast(x.device.type, enabled=amp):
            return F.normalize(model.clip.encode_image(x).float(), dim=-1)
    finally:
        set_lora_scale(model.clip.visual, 1.0)


def fit(model, lora_params, train_df, trained_pairs, epochs, cfg, image_dir, tfs, on_epoch_end=None, log=None):
    rt, tcfg = cfg["runtime"], cfg["training"]
    device, amp = rt["device"], rt["amp"]
    wa = tcfg.get("weight_average")
    assert wa is None or wa["type"] == "ema", f"unknown weight_average {wa}"
    branch_params = list(model.branches.parameters()) if model.branches is not None else []
    ema = WeightEMA([model.crop_vecs, model.disease_vecs, *lora_params, *branch_params], wa["decay"]) if wa else None
    groups_opt = [
        {"params": [model.crop_vecs, model.disease_vecs], "lr": tcfg["lr"]},
        {"params": lora_params, "lr": cfg["image_encoder"]["lora"]["lr"]},
    ]
    if branch_params:
        groups_opt.append({"params": branch_params, "lr": cfg["model"]["primitive_branches"]["lr"]})
    opt = torch.optim.AdamW(groups_opt, weight_decay=tcfg["weight_decay"])
    alt = tcfg.get("alternating_primitives")
    if alt:
        assert alt["first"] in ("disease", "crop"), alt
        vec_sets = {"crop": model.crop_vecs, "disease": model.disease_vecs}
    ssl_w = tcfg["loss_weights"].get("ssl", 0)
    local_tf = ssl_local_transform(cfg["input"], cfg["augmentation"]) if ssl_w > 0 else None
    assert ssl_w == 0 or local_tf is not None, "loss_weights.ssl > 0 needs augmentation.ssl_local"
    ssl_frac = cfg["augmentation"].get("ssl_local", {}).get("batch_fraction", 1.0)
    dl = loader(train_df, image_dir, TwoViews(tfs[0], local_tf) if local_tf else tfs[0], cfg, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    scale, lw = tcfg["logit_scale"], tcfg["loss_weights"]
    trained = torch.as_tensor(trained_pairs, device=device)
    col = torch.full((int(trained.max()) + 1,), -1, device=device, dtype=torch.long)
    col[trained] = torch.arange(len(trained), device=device)
    y_pair = torch.as_tensor(train_df["pair_id"].to_numpy(), device=device)
    y_crop = torch.as_tensor(train_df["crop_id"].to_numpy(), device=device)
    y_dis = torch.as_tensor(train_df["disease_id"].to_numpy(), device=device)
    col_crop = model.pair_crop[trained]
    groups = ("pairs", "crops", "diseases")
    if lw.get("text_anchor", 0) > 0:
        with torch.no_grad():
            text_init = {g: text_embeddings(model, g, amp).float() for g in groups}
    for epoch in range(1, epochs + 1):
        model.train()
        t0, totals, n_batches = time.time(), {"loss": 0.0, "pair": 0.0, "crop": 0.0, "disease": 0.0}, 0
        frozen_vecs = None
        if alt:  # DRPT: train one primitive's vectors per phase; the other set gets no update (grad None -> AdamW skips it)
            first, second = alt["first"], "crop" if alt["first"] == "disease" else "disease"
            training_set = first if ((epoch - 1) // alt["period_epochs"]) % 2 == 0 else second
            frozen_vecs = vec_sets[second if training_set == first else first]
        for x, idx in dl:
            idx = idx.to(device)
            x_local = None
            if local_tf:
                x, x_local = x
                x_local = x_local[: max(1, round(ssl_frac * len(x_local)))].to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            dt = disease_text(model, amp)
            with torch.autocast(device, enabled=amp):
                v, patch = image_features(model.clip, x, dt, model.patch and model.patch["top_k"])
            text = {}  # each group's text embeddings, encoded at most once per step

            def t(g):
                if g not in text:
                    text[g] = text_embeddings(model, g, amp)
                return text[g]

            lam = model.patch["lambda"] if model.patch else 0.0
            pair_logits = combine(v @ t("pairs").float().T, patch, model.pair_disease, lam)[:, trained]
            losses = {"pair": F.cross_entropy(scale * pair_logits, col[y_pair[idx]])}
            if lw.get("within_crop", 0) > 0:
                losses["within_crop"] = within_crop_loss(scale * pair_logits, col[y_pair[idx]], col_crop, y_crop[idx])
            if lw["crop"] > 0:
                losses["crop"] = F.cross_entropy(scale * branch_features(model.branches, "crop", v) @ t("crops").T, y_crop[idx])
            if lw["disease"] > 0:
                dis_logits = patch if patch is not None else branch_features(model.branches, "disease", v) @ t("diseases").T
                losses["disease"] = F.cross_entropy(scale * dis_logits, y_dis[idx])
            if lw.get("image_consistency", 0) > 0:
                losses["image_consistency"] = F.l1_loss(v, frozen_features(model, x, amp))
            if x_local is not None:
                # Local-to-global self-distillation: the small crop's CLS must match the (stop-grad) global view's.
                with torch.autocast(device, enabled=amp):
                    v_local, _ = image_features(model.clip, x_local, None, None)
                losses["ssl"] = (1 - F.cosine_similarity(v_local.float(), v[: len(x_local)].detach().float(), dim=-1)).mean()
            if lw.get("text_anchor", 0) > 0:
                losses["text_anchor"] = sum(F.l1_loss(t(g).float(), text_init[g]) for g in groups) / len(groups)
            loss = sum(lw[k] * l for k, l in losses.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if frozen_vecs is not None:
                frozen_vecs.grad = None
            opt.step()
            sched.step()
            if ema:
                ema.update()
            n_batches += 1
            totals["loss"] += loss.item()
            for k, l in losses.items():
                totals[k] = totals.get(k, 0.0) + l.item()
        rec = {"epoch": epoch, **{k: round(v / n_batches, 4) for k, v in totals.items()}, "sec": round(time.time() - t0)}
        if alt:
            rec["vectors_trained"] = training_set
        if on_epoch_end:
            if ema:
                ema.swap()  # validate with the averaged weights
            rec.update(on_epoch_end(epoch))
            if ema:
                ema.swap()
        if log:
            log(rec)
    if ema:
        ema.swap()  # the fitted model keeps the averaged weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert cfg["method"] == "clip_prompt_lora", f"{args.config} is a '{cfg['method']}' config"
    rt, tcfg, vcfg, ev = cfg["runtime"], cfg["training"], cfg["validation"], cfg["evaluation"]
    device, amp = rt["device"], rt["amp"]
    tta = tuple(ev.get("tta", ["identity"]))
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
    posthoc = "posthoc" in ev
    s_grid, a_grid, b_grid = posthoc_grids(ev)
    fusing = len(a_grid) + len(b_grid) > 2
    hm_grid = np.zeros((n_splits, tcfg["max_epochs"], len(s_grid), len(a_grid), len(b_grid), len(gammas)))
    val_raw, split_info = {}, {}  # (split, epoch, scale index) -> validation cosines; split -> labels, for the bootstrap
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
        split_info[k] = (val_pairs, y_vs, y_vu, n_vs)

        seed_everything(split_seed)
        model, lora_params = build(cfg, pairs, device)
        pc, pdis = model.pair_crop.cpu().numpy(), model.pair_disease.cpu().numpy()

        def validate(epoch, k=k):
            for si, sc in enumerate(s_grid):
                feats = encode_scaled(model, val_dl, device, amp, tta, sc)
                raw = (pair_scores(model, feats, amp), *(primitive_scores(model, feats, amp) if fusing else (None, None)))
                if posthoc:
                    val_raw[k, epoch, si] = raw
                if si == 0:
                    s = raw[0].copy()
                    s[:, hidden] = -1e9
                    res = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,)) for g in gammas]
                for ai, a in enumerate(a_grid):
                    for bi, b in enumerate(b_grid):
                        f = fuse(*raw, pc, pdis, a, b).copy()
                        f[:, hidden] = -1e9
                        hm_grid[k, epoch - 1, si, ai, bi] = hm_top1_over_gammas(f[:n_vs], y_vs, f[n_vs:], y_vu, val_pairs, gammas)
            # Plain scoring (scale 1, weights 0), reported exactly as before evaluation.posthoc existed.
            best = res[int(np.argmax([r["harmonic_mean_top1"] for r in res]))]
            rec = {"val_hm_g0": round(res[0]["harmonic_mean_top1"], 4), "val_hm": round(best["harmonic_mean_top1"], 4),
                   "gamma": best["gamma"], "val_seen": round(best["test_seen"]["top1"], 4),
                   "val_unseen": round(best["test_unseen"]["top1"], 4)}
            if posthoc:
                g5 = hm_grid[k, epoch - 1]
                i = np.unravel_index(g5.argmax(), g5.shape)
                rec.update(val_hm_posthoc=round(float(g5[i]), 4), scale=s_grid[i[0]], crop_w=a_grid[i[1]], dis_w=b_grid[i[2]],
                           gamma_posthoc=float(gammas[i[3]]))
            return rec

        fit(model, lora_params, fit_df, trained_ids, tcfg["max_epochs"], cfg, image_dir, tfs, validate, logger(f"split{k + 1}"))
        del model, lora_params
        torch.cuda.empty_cache()

    mean_grid = hm_grid.mean(axis=0)
    e, si, ai, bi, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    knobs = {"lora_scale": s_grid[si], "crop_weight": a_grid[ai], "disease_weight": b_grid[bi]} if posthoc else {}
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, si, ai, bi, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, si, ai, bi, g]],
            "val_hm_g0_mean": round(float(mean_grid[e, si, ai, bi, 0]), 4)}
    if posthoc:
        base = mean_grid[:, 0, 0, 0]
        eb, gb = np.unravel_index(base.argmax(), base.shape)

        def correct(ep, s_i, a, b, gm):
            # Per-image top-1 correctness on every split's validation images (val-seen, pseudo-unseen), concatenated.
            cs, cu = [], []
            for k, (vp, y_vs, y_vu, n_vs) in split_info.items():
                f = fuse(*val_raw[k, ep, s_i], pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy(), a, b).copy()
                f[:, hidden] = -1e9
                pred = calibrate(f, vp["is_unseen"].to_numpy(), gm).argmax(axis=1)
                cs.append(pred[:n_vs] == y_vs)
                cu.append(pred[n_vs:] == y_vu)
            return np.concatenate(cs), np.concatenate(cu)

        diff = bootstrap_hm_diff(*correct(best_epoch, si, a_grid[ai], b_grid[bi], gamma),
                                 *correct(int(eb) + 1, 0, 0.0, 0.0, float(gammas[gb])))
        best.update(lora_scale=s_grid[si], fusion_crop_weight=a_grid[ai], fusion_disease_weight=b_grid[bi],
                    val_hm_plain_mean=round(float(base[eb, gb]), 4), plain_epoch=int(eb) + 1, plain_gamma=float(gammas[gb]),
                    val_hm_fusion_only_mean=round(float(mean_grid[:, 0].max()), 4),
                    val_hm_scale_only_mean=round(float(mean_grid[:, :, 0, 0].max()), 4),
                    val_hm_gain_vs_plain_bootstrap={"mean": round(diff[0], 4), "ci95": [round(diff[1], 4), round(diff[2], 4)]})
        print(f"post-hoc knobs {knobs}; plain scoring val HM {best['val_hm_plain_mean']:.2%} (epoch {best['plain_epoch']}); "
              f"fusion only {best['val_hm_fusion_only_mean']:.2%}, scale only {best['val_hm_scale_only_mean']:.2%}; "
              f"paired bootstrap gain {diff[0]:+.2%} [{diff[1]:+.2%}, {diff[2]:+.2%}]", flush=True)
        del val_raw
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    seed_everything(cfg["seed"])
    model, lora_params = build(cfg, pairs, device)
    fit(model, lora_params, train, sorted(set(train["pair_id"])), best_epoch, cfg, image_dir, tfs, log=logger("final"))

    scores, labels, files = {}, {}, {}
    for s in ("test_seen", "test_unseen"):
        sub = df[df["split"] == s]
        scores[s] = score_all(model, loader(sub, image_dir, tfs[1], cfg, shuffle=False), device, amp, tta, knobs)
        labels[s], files[s] = sub["pair_id"].to_numpy(), sub["file"].to_numpy()
    top_k = tuple(ev["top_k"])
    result = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, gamma, top_k)
    no_cal = evaluate(scores["test_seen"], labels["test_seen"], scores["test_unseen"], labels["test_unseen"], pairs, 0.0, top_k)
    result.update(version=cfg["version"], selected_epoch=best_epoch, val=best, harmonic_mean_top1_gamma0=no_cal["harmonic_mean_top1"])
    if posthoc:  # read back by knobs_from_metrics (dpd/inference.py, train/ensemble.py)
        result.update(lora_scale=knobs["lora_scale"], fusion_crop_weight=knobs["crop_weight"], fusion_disease_weight=knobs["disease_weight"])
    print("\nTest, uncalibrated:\n" + format_summary(no_cal))
    print("\nTest, gamma from validation:\n" + format_summary(result))

    save_test_outputs(out, result, scores, labels, files, pairs, gamma)
    np.save(out / "val_hm_grid.npy", hm_grid if posthoc else hm_grid[:, :, 0, 0, 0])  # without posthoc: (split, epoch, gamma) as before
    with torch.no_grad():
        np.save(out / "pair_text_embeddings.npy", text_embeddings(model, "pairs", amp).float().cpu().numpy())
        if model.patch or knobs.get("disease_weight"):
            np.save(out / "disease_text_embeddings.npy", text_embeddings(model, "diseases", amp).float().cpu().numpy())
        if knobs.get("crop_weight"):
            np.save(out / "crop_text_embeddings.npy", text_embeddings(model, "crops", amp).float().cpu().numpy())
    torch.save({"crop_vecs": model.crop_vecs.detach().cpu(), "disease_vecs": model.disease_vecs.detach().cpu(), "gamma": gamma},
               out / "prompt_vectors.pt")
    torch.save(lora_state(model.clip.visual), out / "lora_image_encoder.pt")
    if model.branches is not None:
        torch.save(model.branches.state_dict(), out / "primitive_branches.pt")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
