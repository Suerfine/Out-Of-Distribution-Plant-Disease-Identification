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
import copy
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as TV2
from torch.utils.data import DataLoader

from dpd.alt_terms import build_extra, relation_matrices, structured_smoothing_ce
from dpd.cnn import HEAD_SIZES, ImageLabels, build_transforms, compose_pair_scores
from dpd.config import load_config, output_dir
from dpd.data import pair_table, read_metadata
from dpd.new_terms import NewTerms
from dpd.evaluate import evaluate, format_summary, hm_top1_over_gammas
from dpd.report import save_test_outputs
from dpd.resize_cache import ensure_resized
from dpd.splits import make_validation_split
from dpd.supcon import (HEADS, CnCBatchSampler, ContrastiveNet, TwoViewImageLabels, class_prototypes, cnc_loss, embed,
                        fit_linear_heads, linear_log_probs, log_probs, supcon_loss)
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
    cnc = cfg["loss"].get("cnc")  # opt-in: cross-crop positives and same-crop hard negatives for the disease head
    if cnc:
        sampler = CnCBatchSampler(df["crop_id"].to_numpy(), df["disease_id"].to_numpy(), cfg["training"]["batch_size"],
                                  cnc["groups"], cnc["per_crop"], cnc["hard_neg"], cfg["seed"])
        return DataLoader(TwoViewImageLabels(df, image_dir, transform), batch_sampler=sampler, num_workers=rt["num_workers"],
                          pin_memory=True, persistent_workers=rt["num_workers"] > 0)
    if cfg["training"].get("balance_pairs"):  # opt-in: sample images with weight 1 / (pair count) ** alpha
        cnt = df.groupby("pair_id")["pair_id"].transform("count").to_numpy()
        w = torch.as_tensor(cnt ** -float(cfg["training"]["balance_pairs"]), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(df), replacement=True)
        return DataLoader(TwoViewImageLabels(df, image_dir, transform), batch_size=cfg["training"]["batch_size"], sampler=sampler,
                          num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=True)
    return DataLoader(
        TwoViewImageLabels(df, image_dir, transform), batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=rt["num_workers"], pin_memory=True, persistent_workers=rt["num_workers"] > 0, drop_last=True,
    )


def head_ce(logits, y, eps, focal_gamma=None):
    """Cross-entropy with label smoothing; with focal_gamma, each sample is scaled by (1 - p_true) ** gamma (focal loss)."""
    ce = F.cross_entropy(logits, y, label_smoothing=eps, reduction="none")
    if focal_gamma:
        pt = F.softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
        ce = (1 - pt).clamp_min(0.0) ** focal_gamma * ce
    return ce.mean()


def llrd_groups(backbone, base_lr, decay):
    """Layer-wise learning-rate decay for a timm ConvNeXt: the last stage keeps base_lr, each earlier stage is scaled by decay."""
    depth = {}
    for name, p in backbone.named_parameters():
        d = 0 if name.startswith("stem") else 1 + int(name.split(".")[1]) if name.startswith("stages.") else 5
        depth.setdefault(d, []).append(p)
    return [{"params": ps, "lr": base_lr * decay ** (5 - d)} for d, ps in sorted(depth.items())]


@torch.no_grad()
def ema_update(target, source, m):
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def train_network(train_df, epochs, cfg, image_dir, tfs, on_epoch=None, log=None, schedule_epochs=None):
    """Train with SupCon; `on_epoch(model)` runs after every epoch and its results are returned as a list.

    The cosine learning-rate schedule spans `schedule_epochs` (default: `epochs`). A final fit that stops at a chosen
    epoch passes the validation run's full length, so it stops where the validated snapshot stood on the schedule."""
    device, amp = cfg["runtime"]["device"], cfg["runtime"]["amp"]
    mcfg, ocfg, lcfg = cfg["model"], cfg["optimizer"], cfg["loss"]
    model = ContrastiveNet(mcfg["backbone"], mcfg["pretrained"], mcfg["proj_dim"], mcfg.get("proj_hidden"),
                           mcfg.get("ce_dropout"), mcfg.get("drop_path_rate", 0.0), mcfg.get("cosine_scale"),
                           mcfg.get("head_hidden")).to(device)
    model.tta = bool(cfg["scoring"].get("tta"))
    if cfg["training"].get("freeze_stages"):  # opt-in: freeze the stem and the first stages of the backbone
        k = cfg["training"]["freeze_stages"]
        for name, p in model.backbone.named_parameters():
            if name.startswith("stem") or (name.startswith("stages.") and int(name.split(".")[1]) < k):
                p.requires_grad_(False)
    if cfg["runtime"].get("grad_checkpointing") and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)  # same maths, less memory: recompute activations in the backward pass
    model.logit_tau = (lcfg.get("logitnorm") or {}).get("tau")  # opt-in LogitNorm (applies in training and in scoring)
    head_params = list(model.proj.parameters()) + (list(model.cls.parameters()) if model.cls is not None else [])
    extra = None  # opt-in alternative training terms (dpd/alt_terms.py): taxonomy heads or hyperbolic heads
    if lcfg.get("taxonomy"):
        extra = build_extra(lcfg["taxonomy"], model.feat_dim, mcfg["proj_dim"], mcfg.get("ce_dropout") or 0.0, pair_table(read_metadata())).to(device)
    aux = None  # opt-in: compositor / COMO-style outliers / feature synthesis / learned calibration (dpd/new_terms.py)
    if lcfg.get("new"):
        assert lcfg.get("pair_loss"), "the new terms sit on top of the held-out pairs loss"
        pt = pair_table(read_metadata())
        aux = NewTerms(lcfg["new"], model.feat_dim, len(pt), HEAD_SIZES["crop"], HEAD_SIZES["disease"]).to(device)
        model.aux = aux
        head_params += list(aux.parameters())
    rel = relation_matrices(train_df, device) if (lcfg.get("smooth") or {}).get("mode") == "structured" else None
    bb_groups = (llrd_groups(model.backbone, ocfg["lr_backbone"], ocfg["llrd"]["decay"]) if ocfg.get("llrd")
                 else [{"params": model.backbone.parameters(), "lr": ocfg["lr_backbone"]}])
    opt = torch.optim.AdamW(
        bb_groups + [
            {"params": head_params, "lr": ocfg["lr_head"]},
        ] + (extra.param_groups(ocfg["lr_head"]) if extra is not None else []),
        weight_decay=ocfg["weight_decay"],
    )
    sd = lcfg.get("selfdistill")  # opt-in: EMA teacher, KL between student and teacher head outputs
    teacher = None
    if sd:
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if hasattr(teacher.backbone, "set_grad_checkpointing"):
            teacher.backbone.set_grad_checkpointing(False)
    ema_cfg = cfg["training"].get("ema")  # opt-in: evaluate an exponential moving average of the weights
    ema_model = None
    if ema_cfg:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    lp_epochs = cfg["training"].get("lpft_epochs", 0)  # opt-in LP-FT: heads only for the first epochs, then everything
    t_train = tfs[0]
    if cfg["augmentation"].get("trivial"):  # opt-in: TrivialAugment (one random operation per image) before the usual transforms
        t_train = TV2.Compose([TV2.TrivialAugmentWide()] + list(t_train.transforms))
    fm = lcfg.get("distill_fm")  # opt-in: distil a frozen foundation model's image features into the backbone
    fm_teacher, fm_proj = None, None
    if fm:
        import timm
        fm_teacher = timm.create_model(fm["model"], pretrained=True, num_classes=0, img_size=224).to(device).eval()
        for p in fm_teacher.parameters():
            p.requires_grad_(False)
        fm_proj = torch.nn.Linear(model.feat_dim, fm_teacher.num_features).to(device)
        opt.add_param_group({"params": fm_proj.parameters(), "lr": ocfg["lr_head"]})
    loader = make_two_view_loader(train_df, image_dir, t_train, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(schedule_epochs or epochs) * len(loader))
    scaler = torch.amp.GradScaler(device, enabled=amp)
    per_epoch = []
    pl = lcfg.get("pair_loss")  # opt-in: calibrated pair-level loss with pretend-unseen (held-out) pairs
    if pl:
        pairs_all = pair_table(read_metadata())
        pair_of = -torch.ones(HEAD_SIZES["crop"], HEAD_SIZES["disease"], dtype=torch.long)
        pair_of[torch.as_tensor(pairs_all["crop_id"].to_numpy()), torch.as_tensor(pairs_all["disease_id"].to_numpy())] =             torch.as_tensor(pairs_all["pair_id"].to_numpy())
        pair_of = pair_of.to(device)
        pair_crop = torch.as_tensor(pairs_all["crop_id"].to_numpy()).to(device)
        pair_dis = torch.as_tensor(pairs_all["disease_id"].to_numpy()).to(device)
        cand = torch.zeros(len(pairs_all), dtype=torch.bool, device=device)
        cand[torch.as_tensor(train_df["pair_id"].unique()).to(device)] = True
        n_c = torch.bincount(pair_crop[cand], minlength=HEAD_SIZES["crop"])
        n_d = torch.bincount(pair_dis[cand], minlength=HEAD_SIZES["disease"])
        # valid_only: pretend-unseen pairs come only from pairs whose disease stays in >= 2 other pairs and whose crop stays in
        # >= 3 other pairs (as for the real unseen pairs); the share is rescaled so the expected number held out is unchanged
        eligible = (cand & (n_d[pair_dis] >= 3) & (n_c[pair_crop] >= 4)) if pl.get("valid_only") else cand
        hold_p = (min(1.0, pl["holdout_frac"] * float(cand.sum()) / max(float(eligible.sum()), 1.0))
                  if pl.get("valid_only") else pl["holdout_frac"])
    smix = lcfg.get("samecrop_mix")  # opt-in: blend two images of the same crop with different diseases (soft disease target)
    sam = ocfg.get("sam")  # opt-in: sharpness-aware minimisation (second pass at weights moved up the gradient by rho)
    if smix:
        assert pl, "same-crop mixing sits on top of the held-out pairs loss"
        d_cnt = torch.bincount(torch.as_tensor(train_df["disease_id"].to_numpy()), minlength=HEAD_SIZES["disease"]).float()
        w_dis = (1.0 / d_cnt.clamp_min(1.0)).to(device)  # partner chosen with weight 1 / (training images of its disease)
    for epoch in range(1, epochs + 1):
        if lp_epochs:
            for p in model.backbone.parameters():
                p.requires_grad_(epoch > lp_epochs)
            if cfg["runtime"].get("grad_checkpointing") and hasattr(model.backbone, "set_grad_checkpointing"):
                model.backbone.set_grad_checkpointing(epoch > lp_epochs)
        model.train()
        t0, tot, parts, n = time.time(), 0.0, {h: 0.0 for h in HEADS}, 0
        n_mixed = 0
        for v1, v2, y_crop, y_dis in loader:
            x = torch.cat([v1, v2]).to(device, non_blocking=True)
            labels = {"crop": torch.cat([y_crop, y_crop]).to(device), "disease": torch.cat([y_dis, y_dis]).to(device)}
            mixinfo, mix = None, lcfg.get("mix")  # opt-in CutMix over the batch of views; both heads get the mixed targets
            if mix and np.random.rand() < mix["prob"]:
                lam = float(np.random.beta(mix["alpha"], mix["alpha"]))
                perm = torch.randperm(x.size(0), device=device)
                H, W = x.shape[-2:]
                r = math.sqrt(1 - lam)
                ch, cw, cy, cx = int(H * r), int(W * r), np.random.randint(H), np.random.randint(W)
                y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
                x = x.clone()
                x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
                mixinfo = (1 - (y2 - y1) * (x2 - x1) / (H * W), {h: labels[h][perm] for h in HEADS})
            pre = None  # with SAM or same-crop mixing, the held-out pairs and gamma0 of this step are drawn before the forward pass
            if pl and (smix or sam):
                pre = ((torch.rand(len(cand), device=device) < hold_p) & eligible,
                       random.uniform(*pl["gamma0_range"]) if pl.get("gamma0_range") else pl["gamma0"])
            smi = None
            if smix:
                ok = ~pre[0][pair_of[labels["crop"], labels["disease"]]]  # neither image may belong to a held-out pair
                M = ((labels["crop"][:, None] == labels["crop"][None, :]) & (labels["disease"][:, None] != labels["disease"][None, :])
                     & ok[None, :] & ok[:, None])
                has = M.any(1) & (torch.rand(len(x), device=device) < smix["prob"])
                if has.any():
                    jj = torch.zeros(len(x), dtype=torch.long, device=device)
                    jj[has] = torch.multinomial((M.float() * w_dis[labels["disease"]][None, :])[has], 1).squeeze(1)
                    lam = float(np.random.beta(smix["alpha"], smix["alpha"]))
                    x = x.clone()
                    x[has] = lam * x[has] + (1 - lam) * x[jj[has]]
                    smi = (has, jj, lam)
                    n_mixed += int(has.sum())

            def step_loss():
              with torch.autocast(device, enabled=amp):
                  f, z = model.encode(x)
                  logits = model.classify(f) if model.cls is not None else None
              cnc = lcfg.get("cnc")
              losses = {h: supcon_loss(z[h], labels[h], lcfg["temperature"]) for h in HEADS if not (cnc and h == "disease")}
              if cnc:  # Correct-n-Contrast-style disease term instead of the SupCon disease term
                  losses["disease"] = cnc_loss(z["disease"], labels["disease"], labels["crop"], lcfg["temperature"])
              loss = sum(lcfg["weights"][h] * losses[h] for h in HEADS)
              if extra is not None:
                  loss = loss + extra(f.float(), labels)
              if logits is not None and pl:
                  # Each step a random share of the training pairs plays the unseen pairs: their images skip the factor
                  # cross-entropy, and the pair-level cross-entropy over the training pairs shifts every other pair's
                  # score down by gamma0, as calibration does at test time.
                  y_pair = pair_of[labels["crop"], labels["disease"]]
                  if pre is not None:
                      hold, g0 = pre
                  else:
                      hold = (torch.rand(len(cand), device=device) < hold_p) & eligible
                      g0 = random.uniform(*pl["gamma0_range"]) if pl.get("gamma0_range") else pl["gamma0"]
                  keep = (~hold[y_pair]).float() if pl.get("mask_ce_on_holdout", True) else torch.ones_like(y_pair, dtype=torch.float)
                  def hce(h):
                      ce = F.cross_entropy(logits[h].float(), labels[h], label_smoothing=lcfg["label_smoothing"], reduction="none")
                      if smi is not None and h == "disease":  # mixed images: lam on the own disease, 1 - lam on the partner's
                          ce2 = F.cross_entropy(logits[h].float(), labels[h][smi[1]], label_smoothing=lcfg["label_smoothing"], reduction="none")
                          ce = torch.where(smi[0], smi[2] * ce + (1 - smi[2]) * ce2, ce)
                      return (ce * keep).sum() / keep.sum().clamp_min(1.0)

                  loss = loss + lcfg["ce_weight"] * sum(hce(h) for h in HEADS)
                  lp = {h: F.log_softmax(logits[h].float(), dim=-1) for h in HEADS}
                  zraw = lp["crop"][:, pair_crop] + lp["disease"][:, pair_dis]
                  if aux is not None:
                      a_loss, a_extra = aux.step(f.float(), labels, y_pair, cand, hold, keep, zraw, pair_of, pair_crop, pair_dis, model.cls)
                      loss = loss + a_loss
                      zraw = zraw + a_extra
                  zpair = zraw - g0 * (cand & ~hold).float()
                  zpair = zpair.masked_fill(~cand, -1e9)
                  pce = F.cross_entropy(zpair, y_pair, reduction="none")
                  if smi is not None:
                      pce = torch.where(smi[0], smi[2] * pce + (1 - smi[2]) * F.cross_entropy(zpair, y_pair[smi[1]], reduction="none"), pce)
                  loss = loss + pl["weight"] * pce.mean()
              elif logits is not None and rel is not None:  # structured label smoothing instead of uniform
                  loss = loss + lcfg["ce_weight"] * sum(
                      structured_smoothing_ce(logits[h].float(), labels[h], rel[h], lcfg["label_smoothing"]) for h in HEADS
                  )
              elif logits is not None:  # cross-entropy on both views, as in the baselines (optionally focal and/or CutMix targets)
                  fg = (lcfg.get("focal") or {}).get("gamma")
                  eps = lcfg["label_smoothing"]

                  def ce_h(h):
                      l = head_ce(logits[h].float(), labels[h], eps, fg)
                      if mixinfo is not None:
                          l = mixinfo[0] * l + (1 - mixinfo[0]) * head_ce(logits[h].float(), mixinfo[1][h], eps, fg)
                      return l

                  loss = loss + lcfg["ce_weight"] * sum(ce_h(h) for h in HEADS)
                  if teacher is not None:
                      with torch.no_grad(), torch.autocast(device, enabled=amp):
                          tl = teacher.classify(teacher.backbone(x))
                      T = sd["temperature"]
                      loss = loss + sd["weight"] * sum(
                          F.kl_div(F.log_softmax(logits[h].float() / T, -1), F.softmax(tl[h].float() / T, -1), reduction="batchmean") * T * T
                          for h in HEADS
                      )
              cons = lcfg.get("consist")  # opt-in: the two views of an image should get the same prediction (symmetric KL)
              if cons and logits is not None:
                  nb = len(x) // 2
                  for h in HEADS:
                      l1, l2 = F.log_softmax(logits[h][:nb].float(), -1), F.log_softmax(logits[h][nb:].float(), -1)
                      loss = loss + cons["weight"] * 0.5 * (F.kl_div(l1, l2, log_target=True, reduction="batchmean")
                                                             + F.kl_div(l2, l1, log_target=True, reduction="batchmean"))
              if fm_teacher is not None:
                  with torch.no_grad(), torch.autocast(device, enabled=amp):
                      tf = fm_teacher(x).float()
                  loss = loss + fm["weight"] * (1 - F.cosine_similarity(fm_proj(f.float()), tf, dim=-1)).mean()
              return loss, losses

            loss, losses = step_loss()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if sam:
                params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
                inv = 1.0 / scaler.get_scale()
                gn = torch.norm(torch.stack([(p.grad.detach().float() * inv).norm() for p in params]))
                if torch.isfinite(gn):  # otherwise scaler.step skips this step as usual
                    moved = []
                    with torch.no_grad():
                        for p in params:
                            e = p.grad.float() * (inv * sam["rho"] / (gn + 1e-12))
                            p.add_(e.to(p.dtype))
                            moved.append(e)
                    opt.zero_grad(set_to_none=True)
                    loss2, _ = step_loss()
                    scaler.scale(loss2).backward()
                    with torch.no_grad():
                        for p, e in zip(params, moved):
                            p.sub_(e.to(p.dtype))
            scaler.step(opt)
            scaler.update()
            sched.step()
            if teacher is not None:
                ema_update(teacher, model, sd["momentum"])
            if ema_model is not None:
                ema_update(ema_model, model, ema_cfg["momentum"])
            tot += loss.item() * len(v1)
            n += len(v1)
            for h in HEADS:
                parts[h] += losses[h].item() * len(v1)
        rec = {"epoch": epoch, "loss": round(tot / n, 4), **{f"loss_{h}": round(parts[h] / n, 4) for h in HEADS},
               "sec": round(time.time() - t0), **({"mixed_share": round(n_mixed / (2 * n), 3)} if smix else {})}
        if on_epoch is not None:
            per_epoch.append(on_epoch(ema_model if ema_model is not None else model))
        if log:
            log(rec)
    return (ema_model if ema_model is not None else model), per_epoch


def make_scorer(emb_fit, fit_df, cfg, pairs, model=None):
    """Fit the scoring step on the fit images; returns (scorer, classifier). `scorer(embeddings)` gives the
    (images x 333) pair scores. scoring.mode "prototype" (default) uses class prototypes of the projection embeddings;
    "classifier" trains a linear softmax classifier per head on the frozen backbone features (the classifier is None
    for prototypes). Built once per fit, so every target split is scored by the same classifier."""
    scfg = cfg["scoring"]
    labels = {h: fit_df[f"{h}_id"].to_numpy() for h in HEADS}
    if scfg.get("mode", "prototype") == "heads":  # the cross-entropy heads trained with the model
        aux = getattr(model, "aux", None)
        seen_m = np.zeros(len(pairs), bool)
        seen_m[fit_df["pair_id"].unique()] = True

        def heads_scorer(emb):
            S = compose_pair_scores(emb["logp_crop"], emb["logp_disease"], pairs)
            if "logp_comp" in emb:
                S = S + float(emb["fuse_w"][0]) * emb["logp_comp"]
            if aux is not None and aux.kind == "calib":
                dev = next(aux.parameters()).device
                with torch.no_grad():
                    St = torch.from_numpy(S).float().to(dev)
                    sm = torch.from_numpy(seen_m).to(dev)
                    z = St.masked_fill(~sm, -1e9)
                    fe = torch.from_numpy(emb["feat"]).float().to(dev) if aux.fproj is not None else None
                    put = aux.p_unseen(z, fe)
                    pu = put.cpu().numpy()
                    if aux.cfg.get("gate") == "mixture":  # law of total probability (COSMO-style soft gate)
                        ls = F.log_softmax(St.masked_fill(~sm, -1e9), dim=-1)
                        lu = F.log_softmax(St.masked_fill(sm, -1e9), dim=-1)
                        mix = torch.where(sm[None, :], torch.log1p(-put.clamp(max=1 - 1e-6))[:, None] + ls,
                                          torch.log(put.clamp_min(1e-6))[:, None] + lu)
                        heads_scorer.parts = None
                        return mix.cpu().numpy()
                heads_scorer.parts = (S.copy(), pu)  # kept so validation can search the shift scale without rescoring
                S = S - aux.cfg.get("scale", 7.0) * pu[:, None] * seen_m[None, :]
            return S

        return heads_scorer, None
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
    ncfg = cfg["loss"].get("new") or {}
    scale_grid = (np.round(np.arange(*ncfg["scale_grid"]), 6) if ncfg.get("kind") == "calib" and ncfg.get("scale_grid") else None)
    if scale_grid is not None:
        hm_grid4 = np.zeros((n_splits, tcfg["max_epochs"], len(scale_grid), len(gammas)))
    for k in range(0 if (not args.pilot and tcfg.get("reuse_validation_grid")) else n_splits):
        split_seed = cfg["seed"] + k
        fit_df, val_seen, val_unseen, holdout = make_validation_split(
            train, vcfg["n_holdout_pairs"], vcfg["min_holdout_images"], vcfg["val_fraction"], split_seed,
            tuple(vcfg["exclude_holdout_diseases"]), vcfg["max_images_per_holdout_pair"], vcfg.get("min_other_crops", 1),
        )
        print(f"validation split {k + 1}/{n_splits}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}", flush=True)
        val_df = pd.concat([val_seen, val_unseen])
        n_vs = len(val_seen)
        fit_loader, val_loader = eval_loader(fit_df), eval_loader(val_df)
        val_pairs = pairs.assign(is_unseen=~pairs["pair_id"].isin(set(fit_df["pair_id"])))
        y_vs, y_vu = val_seen["pair_id"].to_numpy(), val_unseen["pair_id"].to_numpy()

        vparts = []  # opt-in validation.save_parts: per-epoch crop and disease log-probs of the validation images

        def on_epoch(model):
            scorer, _ = make_scorer(embed(model, fit_loader, device, amp), fit_df, cfg, pairs, model)
            ev_ = embed(model, val_loader, device, amp)
            if vcfg.get("save_parts"):
                vparts.append((ev_["logp_crop"].astype(np.float32), ev_["logp_disease"].astype(np.float32)))
            s = scorer(ev_)
            return scorer.parts if scale_grid is not None else s

        seed_everything(split_seed)
        _, val_scores = train_network(fit_df, tcfg["max_epochs"], cfg, image_dir, tfs, on_epoch, logger(f"split{k + 1}"))
        if vparts:  # lets ensembles and score weightings be chosen on validation later, without retraining
            np.savez_compressed(out / f"val_parts_split{k + 1}.npz", logp_crop=np.stack([a for a, _ in vparts]),
                                logp_disease=np.stack([b for _, b in vparts]), y_val_seen=y_vs, y_val_unseen=y_vu,
                                fit_pairs=np.sort(fit_df["pair_id"].unique()), split_seed=split_seed)
        for e, s in enumerate(val_scores):
            if scale_grid is not None:  # learned detector: search its shift scale together with gamma (fast sweep)
                base, pu = s
                seen_cols = ~val_pairs["is_unseen"].to_numpy()
                for j, sc in enumerate(scale_grid):
                    S = base - sc * pu[:, None] * seen_cols[None, :]
                    S[:, hidden] = -1e9
                    hm_grid4[k, e, j] = hm_top1_over_gammas(S[:n_vs], y_vs, S[n_vs:], y_vu, val_pairs, gammas)
                j = int(hm_grid4[k, e].max(axis=1).argmax())
                print(f"split {k + 1} epoch {e + 1}: best val HM {hm_grid4[k, e].max():.2%} at scale {scale_grid[j]}, "
                      f"gamma {gammas[hm_grid4[k, e, j].argmax()]}", flush=True)
                continue
            s[:, hidden] = -1e9
            hm_grid[k, e] = [evaluate(s[:n_vs], y_vs, s[n_vs:], y_vu, val_pairs, g, (1,))["harmonic_mean_top1"] for g in gammas]
            print(f"split {k + 1} epoch {e + 1}: val HM {hm_grid[k, e, 0]:.2%} at gamma 0, best {hm_grid[k, e].max():.2%} "
                  f"at gamma {gammas[hm_grid[k, e].argmax()]}", flush=True)
    reuse = None if args.pilot else tcfg.get("reuse_validation_grid")
    if reuse:  # stage 1 unchanged from an earlier run: reuse its saved validation grid instead of retraining
        hm_grid = np.load(reuse)
        assert hm_grid.shape == (n_splits, tcfg["max_epochs"], len(gammas)), f"{reuse} does not match this config's validation setup"
        print(f"stage 1 skipped: validation grid loaded from {reuse}", flush=True)
    best_scale = None
    if scale_grid is not None:  # pick (epoch, scale, gamma) on the mean over splits, then keep the 3-D grid at that scale
        _, j, _ = np.unravel_index(hm_grid4.mean(axis=0).argmax(), hm_grid4.shape[1:])
        best_scale = float(scale_grid[j])
        hm_grid = hm_grid4[:, :, j, :]
        cfg["loss"]["new"]["scale"] = best_scale  # the final fit's scorer uses the validation-chosen scale
        np.save(out / "val_hm_grid_scales.npy", hm_grid4)
        print(f"detector shift scale chosen on validation: {best_scale}", flush=True)
    mean_grid = hm_grid.mean(axis=0)
    if tcfg.get("force_epoch") and not args.pilot:  # ablation: stop at a given epoch; gamma is the validation-best one for it
        e = int(tcfg["force_epoch"]) - 1
        g = int(mean_grid[e].argmax())
    else:
        e, g = np.unravel_index(mean_grid.argmax(), mean_grid.shape)
    best_epoch, gamma = int(e) + 1, float(gammas[g])
    best = {"epoch": best_epoch, "gamma": gamma, "val_hm_mean": round(float(mean_grid[e, g]), 4),
            "val_hm_per_split": [round(float(x), 4) for x in hm_grid[:, e, g]], "val_hm_g0_mean": round(float(mean_grid[e, 0]), 4)}
    if best_scale is not None:
        best["scale"] = best_scale
    print(f"selected epoch {best_epoch}, gamma {gamma}: mean val HM {best['val_hm_mean']:.2%}", flush=True)
    if tcfg.get("stage1_only"):  # screening: validation only (training data), no final fit and no test evaluation
        np.save(out / "val_hm_grid.npy", hm_grid)
        best.update(version=cfg["version"], per_epoch_best_mean=[round(float(v), 4) for v in mean_grid.max(axis=1)],
                    per_split_best=[round(float(v), 4) for v in hm_grid.reshape(n_splits, -1).max(axis=1)],
                    gamma0_mean_by_epoch=[round(float(v), 4) for v in mean_grid[:, 0]])
        (out / "stage1.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"screening run finished (validation only); saved to {out}", flush=True)
        return

    # ---- Stage 2: refit on all training images, evaluate once on test ----
    seed_everything(cfg["seed"])
    sched_len = tcfg["max_epochs"] if tcfg.get("final_schedule") == "match_validation" else None
    model, _ = train_network(train, best_epoch, cfg, image_dir, tfs, log=logger("final"), schedule_epochs=sched_len)
    torch.save(model.state_dict(), out / "supcon_net.pt")
    emb_train = embed(model, eval_loader(train), device, amp)
    protos = {}
    for h in HEADS:
        protos[f"{h}_prototypes"], protos[f"{h}_valid"] = class_prototypes(emb_train[h], train[f"{h}_id"].to_numpy(), HEAD_SIZES[h])
    np.savez_compressed(out / "prototypes.npz", temperature=temperature, **protos)
    scorer, clf = make_scorer(emb_train, train, cfg, pairs, model)
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
