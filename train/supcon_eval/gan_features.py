"""GAN test, step 1 (GPU, 2026-11-16 night). Backbone features (1024-d) and head weights for the feature-generation test.
  (a) validation network: the seed-42 control config trained on validation split 1's fit part (split seed 42, the same split the
      control used) for 17 epochs on its 20-epoch schedule = the control's validation snapshot at its selected epoch.
      Features of the fit images, the validation-seen images and the pseudo-unseen images.
  (b) the finished seed-42 control (runs/contrastive/supcon_ce_convnext_base_control_no_supcon): features of all 8,000 training
      images and of both test sets.
Writes queue/gan/val_split1.npz, val_split1_cls.pt, final_s42.npz, final_s42_cls.pt."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[3] / "queue"
REPO = HERE.parent / "Out-Of-Distribution-Plant-Disease-Identification"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from dpd.cnn import ImageLabels, build_transforms  # noqa: E402
from dpd.data import read_metadata  # noqa: E402
from dpd.resize_cache import ensure_resized  # noqa: E402
from dpd.splits import make_validation_split  # noqa: E402
from dpd.supcon import ContrastiveNet, embed  # noqa: E402
from train.train_cnn import seed_everything  # noqa: E402
from train.train_supcon import train_network  # noqa: E402

OUT = HERE / "gan"
RUN = REPO / "runs" / "contrastive" / "supcon_ce_convnext_base_control_no_supcon"


def heads_state(model):
    return {h: {"weight": model.cls[h][1].weight.detach().cpu(), "bias": model.cls[h][1].bias.detach().cpu()} for h in ("crop", "disease")}


def feats(model, frame, image_dir, tf):
    e = embed(model, DataLoader(ImageLabels(frame, image_dir, tf), batch_size=128, shuffle=False, num_workers=2), "cuda", True)
    return e["feat"].astype(np.float32), e["logp_crop"], e["logp_disease"]


def main():
    OUT.mkdir(exist_ok=True)
    cfg = json.load(open(RUN / "config.json", encoding="utf-8"))
    df = read_metadata()
    train = df[df["split"] == "train"]
    image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
    tfs = build_transforms(cfg["input"], cfg["augmentation"])
    if not (OUT / "val_split1.npz").exists():
        v = cfg["validation"]
        fit_df, val_seen, val_unseen, _ = make_validation_split(train, v["n_holdout_pairs"], v["min_holdout_images"], v["val_fraction"],
                                                                cfg["seed"], tuple(v["exclude_holdout_diseases"]), v["max_images_per_holdout_pair"],
                                                                v.get("min_other_crops", 1))
        ep = json.load(open(RUN / "metrics.json"))["selected_epoch"]
        print(f"split 1: fit {len(fit_df)}, val-seen {len(val_seen)}, pseudo-unseen {len(val_unseen)}; training {ep} of {cfg['training']['max_epochs']} epochs", flush=True)
        seed_everything(cfg["seed"])
        model, _ = train_network(fit_df, ep, cfg, image_dir, tfs, log=lambda r: print(r, flush=True), schedule_epochs=cfg["training"]["max_epochs"])
        model.eval()
        out = {}
        for name, frame in (("fit", fit_df), ("vs", val_seen), ("vu", val_unseen)):
            out[f"feat_{name}"], out[f"lc_{name}"], out[f"ld_{name}"] = feats(model, frame, image_dir, tfs[1])
            out[f"pair_{name}"] = frame["pair_id"].to_numpy()
            out[f"crop_{name}"], out[f"dis_{name}"] = frame["crop_id"].to_numpy(), frame["disease_id"].to_numpy()
        out["fit_pairs"] = np.sort(fit_df["pair_id"].unique())
        np.savez_compressed(OUT / "val_split1.npz", **out)
        torch.save(heads_state(model), OUT / "val_split1_cls.pt")
        del model
        torch.cuda.empty_cache()
        print("saved validation features", flush=True)
    if not (OUT / "final_s42.npz").exists():
        m = cfg["model"]
        model = ContrastiveNet(m["backbone"], False, m["proj_dim"], m.get("proj_hidden"), m.get("ce_dropout"), m.get("drop_path_rate", 0.0),
                               m.get("cosine_scale"), m.get("head_hidden"))
        model.load_state_dict(torch.load(RUN / "supcon_net.pt", map_location="cpu"))
        model = model.to("cuda").eval()
        out = {}
        for name, frame in (("train", train), ("ts", df[df["split"] == "test_seen"]), ("tu", df[df["split"] == "test_unseen"])):
            out[f"feat_{name}"], out[f"lc_{name}"], out[f"ld_{name}"] = feats(model, frame, image_dir, tfs[1])
            out[f"pair_{name}"] = frame["pair_id"].to_numpy()
            out[f"crop_{name}"], out[f"dis_{name}"] = frame["crop_id"].to_numpy(), frame["disease_id"].to_numpy()
        np.savez_compressed(OUT / "final_s42.npz", **out)
        torch.save(heads_state(model), OUT / "final_s42_cls.pt")
        print("saved final-model features", flush=True)


if __name__ == "__main__":
    main()
