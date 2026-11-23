"""Disease-only head on frozen CLIP features with a crop-anchored feature GAN, scored on the ensemble's validation split.

The method is Katsuki's re-implementation (train/supcon_eval/clip_gan_disease.py) of Hein's idea
(Disease_Transfer_Experiments/clip_anchored_gan_v1): a WGAN-GP generates CLIP image features for target pairs from a real
feature of the target crop plus the disease label; a linear 175-disease head is trained on real features + 300 generated
features per target pair. Its functions are reused unchanged; only the data differ:
  features  the cached frozen OpenAI CLIP ViT-B/16 embeddings in runs/cache/ (no re-encoding).
  split     the v6 ensembles' validation split (settings and seed read from --config), so the head can be mixed with
            their cached val_scores_*.npy. Targets: the split's pseudo-unseen pairs.
  test      refit on all training images, targets = the 10 real unseen pairs (names only, no images).
Writes <out>/val.npz (disease log-probs of val-seen then pseudo-unseen images) and <out>/test.npz.

--recipe aligned (2026-11-22) uses Hein's own pipeline instead of Katsuki's re-implementation, on the same data:
  generator  Disease_Transfer_Experiments/clip_anchored_gan_v1/study.py train_generator + build_anchor_pool: anchor = a real
             HEALTHY feature of the target crop; condition = CLIP text embedding of the pair prompt (critic sees it too);
             f-CLSWGAN settings of config/vlm/v11.3 (hidden 4096, noise 512, 30 epochs, cls 0.01 from a pair-level cosine
             classifier initialised from the pair text embeddings).
  head       cosine classifier over the 175 diseases (scale 50), initialised from the disease text embeddings, 1 epoch,
             on real features + 300 generated per target pair (as clip_disease_only.py). Output: log-softmax of its logits.
Usage: python -m train.gan_disease_head --config config/vlm/v6.7_three_member_ensemble_config.json --out runs/vlm/gan_disease_head
       [--features lora:v10.3] [--recipe aligned] [--val-only]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train" / "supcon_eval"))
from clip_gan_disease import DEV, disease_logp  # noqa: E402

from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402

CACHE = {"train": "train_2b697aedac", "test_seen": "test_seen_691fade910", "test_unseen": "test_unseen_86b1a782d8"}
GAN_CFG = ROOT / "config/vlm/v11.3_lora_features_config.json"  # Hein's GAN / synthesis / classifier settings


class AlignedRecipe:
    """Hein's recipe (see the module docstring). Text embeddings are computed once; disease_logp has Katsuki's signature
    plus the fitting frame (needed for the healthy anchors)."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / "Disease_Transfer_Experiments" / "clip_anchored_gan_v1"))
        from copy import deepcopy
        from study import build_anchor_pool, train_generator  # noqa: E402
        from dpd.clip_utils import load_clip
        from dpd.feature_gen import fit_classifier
        from train.train_feature_gen import seed_everything, text_conditions
        self.build_anchor_pool, self.train_generator = build_anchor_pool, train_generator
        self.fit_classifier, self.seed_everything = fit_classifier, seed_everything
        self.cfg = json.load(open(GAN_CFG, encoding="utf-8"))
        cfg_d = deepcopy(self.cfg)
        cfg_d["condition"]["parts"] = ["disease"]
        pairs = pair_table(read_metadata())
        net, _, tok = load_clip(self.cfg["model"], DEV)
        self.prompt, self.cond = (torch.as_tensor(a, device=DEV).float() for a in text_conditions(self.cfg, net, tok, pairs, DEV))
        _, disease_by_pair = text_conditions(cfg_d, net, tok, pairs, DEV)
        del net
        torch.cuda.empty_cache()
        _, first = np.unique(pairs["disease_id"].to_numpy(), return_index=True)
        assert len(first) == 175
        self.disease_init = torch.as_tensor(disease_by_pair[first], device=DEV).float()

    def disease_logp(self, fit_df, X_tr, target_pairs, pairs, X_eval, seed):
        y = torch.as_tensor(fit_df["pair_id"].to_numpy(), device=DEV)
        trained = torch.zeros(len(pairs), dtype=torch.bool, device=DEV)
        trained[y.unique()] = True
        assert not trained[torch.as_tensor(target_pairs, device=DEV)].any()
        self.seed_everything(seed)
        aux_cfg = self.cfg["aux_classifier"]
        aux = self.fit_classifier(X_tr, y, trained, self.prompt, aux_cfg, aux_cfg["epochs"], tag="aux", log=lambda s: None)
        pool, audit = self.build_anchor_pool(X_tr, fit_df.reset_index(drop=True), pairs)
        crop_of = pairs.set_index("pair_id")["crop_id"]
        missing = [int(p) for p in target_pairs if audit[int(crop_of[p])] == 0]
        print(f"    target pairs without a healthy anchor (zero anchor): {missing}", flush=True)
        generate = self.train_generator(X_tr, y, self.cond, aux, self.cfg["gan"], pool)
        sy = torch.as_tensor(np.repeat(target_pairs, self.cfg["synthesis"]["n_per_unseen"]), device=DEV)
        with torch.no_grad():
            fake = generate(self.cond[sy], sy)
        assert torch.isfinite(fake).all()
        did = torch.as_tensor(pairs["disease_id"].to_numpy(), device=DEV)
        self.seed_everything(seed)
        head = self.fit_classifier(torch.cat([X_tr, fake]), torch.cat([did[y], did[sy]]), torch.ones(175, dtype=torch.bool, device=DEV),
                                   self.disease_init, self.cfg["classifier"], 1, tag="disease_head")
        with torch.no_grad():
            return torch.log_softmax(head(X_eval), 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="frozen", help="'frozen' (cached OpenAI CLIP) or 'lora:<run>' (e.g. lora:v10.3)")
    ap.add_argument("--recipe", default="katsuki", choices=["katsuki", "aligned"])
    ap.add_argument("--val-only", action="store_true", help="do not compute test.npz")
    args = ap.parse_args()
    if args.recipe == "aligned":
        recipe = AlignedRecipe()
        head_logp = lambda fit, X, crop, dis, target, pairs, X_eval, seed: recipe.disease_logp(fit, X, target, pairs, X_eval, seed)
    else:
        head_logp = lambda fit, X, crop, dis, target, pairs, X_eval, seed: disease_logp(X, crop, dis, target, pairs, X_eval, seed)
    cfg = json.load(open(args.config, encoding="utf-8"))
    v = cfg["validation"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = read_metadata()
    pairs = pair_table(df)
    if args.features == "frozen":
        val_feats = test_feats = {s: np.load(ROOT / "runs/cache" / f"ViT-B-16-quickgelu_openai_{c}.npy") for s, c in CACHE.items()}
    else:  # a LoRA run's adapted encoder, as cached by train_feature_gen.lora_features(): retrained on the split for validation
        run, epochs = args.features.split(":")[1], json.load(open(ROOT / "runs/vlm" / args.features.split(":")[1] / "metrics.json"))["selected_epoch"]
        val_feats = dict(np.load(ROOT / "runs/cache/lora_features" / f"{run}_seed{cfg['seed']}_e{epochs}.npz"))
        final = ROOT / "runs/cache/lora_features" / f"{run}_final.npz"
        test_feats = {} if args.val_only and not final.exists() else dict(np.load(final))  # validation needs train only
        val_feats.update({s: test_feats[s] for s in ("test_seen", "test_unseen") if s in test_feats})  # never used for validation
        for f in (val_feats, test_feats):
            f.update({s: x / np.linalg.norm(x, axis=1, keepdims=True) for s, x in f.items()})
    feats = val_feats
    pos = {}
    for s in (s for s in CACHE if s in feats):
        files = df[df["split"] == s]["file"].to_numpy()
        assert len(files) == len(feats[s]), s
        pos.update({f: (s, i) for i, f in enumerate(files)})
    T = lambda frame: torch.as_tensor(np.stack([feats[pos[f][0]][pos[f][1]] for f in frame["file"]]), device=DEV).float()
    L = lambda frame, col: torch.as_tensor(frame[col].to_numpy(), device=DEV)
    train = df[df["split"] == "train"]

    if not (out / "val.npz").exists():
        t0 = time.time()
        fit_df, val_seen, val_unseen, _ = make_validation_split(train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
                                                                cfg["seed"], tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"])
        target = np.setdiff1d(pairs.loc[~pairs["is_unseen"], "pair_id"].to_numpy(), fit_df["pair_id"].unique())
        print(f"validation split seed {cfg['seed']}: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)} "
              f"images, {len(target)} target pairs", flush=True)
        ld = head_logp(fit_df, T(fit_df), L(fit_df, "crop_id"), L(fit_df, "disease_id"), target, pairs,
                       T(pd.concat([val_seen, val_unseen])), cfg["seed"])
        n = len(val_seen)
        print(f"  disease named right: pseudo-unseen {(ld[n:].argmax(1) == val_unseen['disease_id'].to_numpy()).mean():.1%}, "
              f"val-seen {(ld[:n].argmax(1) == val_seen['disease_id'].to_numpy()).mean():.1%} ({time.time() - t0:.0f} s)", flush=True)
        np.savez_compressed(out / "val.npz", ld=ld, y_val_seen=val_seen["pair_id"].to_numpy(), y_val_unseen=val_unseen["pair_id"].to_numpy())

    if not args.val_only and not (out / "test.npz").exists():
        t0 = time.time()
        feats.clear()
        feats.update(test_feats)  # T() reads `feats`: the encoder trained on all train
        ts, tu = df[df["split"] == "test_seen"], df[df["split"] == "test_unseen"]
        ld = head_logp(train, T(train), L(train, "crop_id"), L(train, "disease_id"), pairs.loc[pairs["is_unseen"], "pair_id"].to_numpy(),
                 pairs, T(pd.concat([ts, tu])), 7)
        n = len(ts)
        # Disease accuracy is printed for the record only; nothing is chosen from it.
        print(f"test: disease named right on unseen images {(ld[n:].argmax(1) == tu['disease_id'].to_numpy()).mean():.1%}, "
              f"seen images {(ld[:n].argmax(1) == ts['disease_id'].to_numpy()).mean():.1%} ({time.time() - t0:.0f} s)", flush=True)
        np.savez_compressed(out / "test.npz", ld_ts=ld[:n], ld_tu=ld[n:], y_ts=ts["pair_id"].to_numpy(), y_tu=tu["pair_id"].to_numpy())


if __name__ == "__main__":
    main()
