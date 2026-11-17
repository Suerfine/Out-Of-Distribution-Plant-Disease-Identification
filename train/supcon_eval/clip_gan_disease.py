"""Disease-only head on FROZEN CLIP features with a crop-anchored feature GAN (2026-11-16 night). The idea is Hein's
(Disease_Transfer_Experiments/clip_anchored_gan_v1: a disease-only CLIP head trained with generated features for the unseen
pairs named 45% of unseen test diseases, against 14.5% for a ConvNeXt disease head); this is a re-implementation for our mix.
Settings fixed BEFORE any result:
  features   OpenAI CLIP ViT-B/16 (frozen), L2-normalised image embeddings of the 256 px cache copies.
  generator  G(noise 64, one-hot disease 175, anchor) = normalise(anchor + MLP(...)), anchor = a real training feature of the
             target pair's CROP (crop anchoring); critic D(feature, one-hot disease); WGAN-GP (penalty 10, 5 critic steps,
             Adam 1e-4, betas 0.5/0.999, batch 256, 3,000 generator steps) + 0.1 x cross-entropy of a disease classifier
             trained on the real features.
  head       linear 512 -> 175 diseases on the features (x 30), trained 2 epochs (Adam 1e-3, weight decay 1e-4, label smoothing
             0.1) on all real training features + 300 generated features per target pair.
  targets    validation: the split's pseudo-unseen pairs; test: the 10 real unseen pairs (their names only, no images).
  splits     the six validation splits of the ConvNeXt runs (seeds 42-47, same images, checked).
Writes runs/clip_gan_disease/{val_split_s<seed>.npz, test.npz}: disease log-probabilities of the validation / test images.
Usage (repo root, DPD_ROOT set): python train/supcon_eval/clip_gan_disease.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.clip_utils import encode_images, load_clip  # noqa: E402
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.resize_cache import cached_name, ensure_resized  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402

DEV = "cuda"
OUT = REPO / "runs" / "clip_gan_disease"
ND, NZ = 175, 64
MODEL = {"library": "open_clip", "arch": "ViT-B-16-quickgelu", "pretrained": "openai"}


def train_classifier(X, y, epochs, seed, extra=None):
    torch.manual_seed(seed)
    if extra is not None:
        X, y = torch.cat([X, extra[0]]), torch.cat([y, extra[1]])
    head = nn.Linear(X.shape[1], ND).to(DEV)
    opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        for b in range(0, len(X), 128):
            i = perm[b:b + 128]
            loss = F.cross_entropy(head(X[i] * 30), y[i], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


class Gen(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NZ + ND + dim, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, dim))

    def forward(self, z, d, anchor):
        return F.normalize(anchor + self.net(torch.cat([z, F.one_hot(d, ND).float(), anchor], 1)), dim=1)


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + ND, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1))

    def forward(self, x, d):
        return self.net(torch.cat([x, F.one_hot(d, ND).float()], 1))


def train_gan(X, crop, dis, seed, steps=3000):
    """X: real features (torch, normalised); crop / dis: labels. Anchor for a real sample = another real feature of its crop."""
    torch.manual_seed(seed)
    dim = X.shape[1]
    G, D = Gen(dim).to(DEV), Critic(dim).to(DEV)
    clf = train_classifier(X, dis, 3, seed)
    for p in clf.parameters():
        p.requires_grad_(False)
    og = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999))
    by_crop = {int(c): torch.nonzero(crop == c).squeeze(1) for c in torch.unique(crop)}
    n = len(X)

    def anchors(c):
        return X[torch.stack([by_crop[int(k)][torch.randint(0, len(by_crop[int(k)]), (1,), device=DEV)] for k in c]).squeeze(1)]

    for step in range(steps):
        for _ in range(5):
            i = torch.randint(0, n, (256,), device=DEV)
            x, d = X[i], dis[i]
            fake = G(torch.randn(256, NZ, device=DEV), d, anchors(crop[i])).detach()
            a = torch.rand(256, 1, device=DEV)
            mix = (a * x + (1 - a) * fake).requires_grad_(True)
            gp = torch.autograd.grad(D(mix, d).sum(), mix, create_graph=True)[0]
            loss_d = D(fake, d).mean() - D(x, d).mean() + 10 * ((gp.norm(dim=1) - 1) ** 2).mean()
            od.zero_grad()
            loss_d.backward()
            od.step()
        i = torch.randint(0, n, (256,), device=DEV)
        fake = G(torch.randn(256, NZ, device=DEV), dis[i], anchors(crop[i]))
        loss_g = -D(fake, dis[i]).mean() + 0.1 * F.cross_entropy(clf(fake * 30), dis[i])
        og.zero_grad()
        loss_g.backward()
        og.step()
    print(f"    gan done: critic {loss_d.item():.3f}, generator {loss_g.item():.3f}", flush=True)
    return G, anchors


def disease_logp(X_tr, crop_tr, dis_tr, target_pairs, pairs, X_eval, seed):
    G, anchors = train_gan(X_tr, crop_tr, dis_tr, seed)
    k = pairs.set_index("pair_id").loc[target_pairs]
    c = torch.as_tensor(np.repeat(k["crop_id"].to_numpy(), 300), device=DEV)
    d = torch.as_tensor(np.repeat(k["disease_id"].to_numpy(), 300), device=DEV)
    with torch.no_grad():
        fake = torch.cat([G(torch.randn(len(cc), NZ, device=DEV), dd, anchors(cc)) for cc, dd in zip(c.split(1000), d.split(1000))])
    head = train_classifier(X_tr, dis_tr, 2, seed + 1, (fake, d))
    with torch.no_grad():
        return F.log_softmax(head(X_eval * 30), 1).cpu().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_metadata()
    pairs = pair_table(df)
    image_dir = ensure_resized(df["file"], 256)
    clip_model, preprocess, _ = load_clip(MODEL, DEV)
    files = [image_dir / cached_name(f) for f in df["file"]]
    feats = encode_images(clip_model, preprocess, [str(f) for f in files], MODEL,
                          {"device": DEV, "amp": True, "batch_size": 128, "num_workers": 2}, cache_name="all_256")
    del clip_model
    torch.cuda.empty_cache()
    F_all = torch.as_tensor(feats, device=DEV).float()
    pos = {f: i for i, f in enumerate(df["file"])}
    T = lambda frame: F_all[[pos[f] for f in frame["file"]]]
    L = lambda frame, col: torch.as_tensor(frame[col].to_numpy(), device=DEV)
    train = df[df["split"] == "train"]
    cfg = json.load(open(REPO / "runs/contrastive/full_res336/config.json", encoding="utf-8"))
    v = cfg["validation"]
    for group, base in (("full_res336", 42), ("full_res336_s45", 45)):
        for k in range(3):
            seed = base + k
            f = OUT / f"val_split_s{seed}.npz"
            if f.exists():
                continue
            fit_df, val_seen, val_unseen, _ = make_validation_split(train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"], seed,
                                                                    tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
            ref = np.load(REPO / "runs/contrastive" / group / f"val_parts_split{k + 1}.npz")
            assert (val_seen["pair_id"].to_numpy() == ref["y_val_seen"]).all() and (val_unseen["pair_id"].to_numpy() == ref["y_val_unseen"]).all()
            target = np.setdiff1d(pairs.loc[~pairs["is_unseen"], "pair_id"].to_numpy(), fit_df["pair_id"].unique())
            print(f"split seed {seed}: fit {len(fit_df)}, {len(target)} pseudo-unseen target pairs", flush=True)
            val_df = pd.concat([val_seen, val_unseen])
            ld = disease_logp(T(fit_df), L(fit_df, "crop_id"), L(fit_df, "disease_id"), target, pairs, T(val_df), seed)
            n = len(val_seen)
            acc_u = (ld[n:].argmax(1) == val_unseen["disease_id"].to_numpy()).mean() * 100
            acc_s = (ld[:n].argmax(1) == val_seen["disease_id"].to_numpy()).mean() * 100
            print(f"  disease named right: pseudo-unseen {acc_u:.1f}%, validation-seen {acc_s:.1f}%", flush=True)
            np.savez_compressed(f, ld=ld, y_val_seen=val_seen["pair_id"].to_numpy(), y_val_unseen=val_unseen["pair_id"].to_numpy())
    if not (OUT / "test.npz").exists():
        ts, tu = df[df["split"] == "test_seen"], df[df["split"] == "test_unseen"]
        ld = disease_logp(T(train), L(train, "crop_id"), L(train, "disease_id"), pairs.loc[pairs["is_unseen"], "pair_id"].to_numpy(), pairs,
                          T(pd.concat([ts, tu])), 7)
        n = len(ts)
        print(f"test: disease named right on unseen images {(ld[n:].argmax(1) == tu['disease_id'].to_numpy()).mean() * 100:.1f}%, "
              f"seen images {(ld[:n].argmax(1) == ts['disease_id'].to_numpy()).mean() * 100:.1f}%", flush=True)
        np.savez_compressed(OUT / "test.npz", ld_ts=ld[:n], ld_tu=ld[n:], y_ts=ts["pair_id"].to_numpy(), y_tu=tu["pair_id"].to_numpy())


if __name__ == "__main__":
    main()
