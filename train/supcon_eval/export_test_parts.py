"""Re-embed the test sets with a finished run's saved final model and save the crop and disease log-probs separately
(test_parts.npz), so crop/disease weightings chosen on validation can be applied to test. Checks that the sum
reproduces the saved pair scores. GPU, a few minutes per run. Run from anywhere: python export_test_parts.py <run> ..."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import os  # noqa: E402

os.chdir(REPO)
from dpd.data import pair_table, read_metadata  # noqa: E402
from dpd.supcon import ContrastiveNet, embed  # noqa: E402
from dpd.cnn import ImageLabels, build_transforms  # noqa: E402
from dpd.resize_cache import ensure_resized  # noqa: E402


def main():
    df = read_metadata()
    pairs = pair_table(df)
    cid, did = pairs["crop_id"].to_numpy(), pairs["disease_id"].to_numpy()
    for name in sys.argv[1:]:
        d = REPO / "runs" / "contrastive" / name
        cfg = json.load(open(d / "config.json", encoding="utf-8"))
        m = cfg["model"]
        model = ContrastiveNet(m["backbone"], False, m["proj_dim"], m.get("proj_hidden"), m.get("ce_dropout"),
                               m.get("drop_path_rate", 0.0), m.get("cosine_scale"), m.get("head_hidden"))
        model.load_state_dict(torch.load(d / "supcon_net.pt", map_location="cpu"))
        model = model.to("cuda").eval()
        model.tta = bool(cfg["scoring"].get("tta"))
        image_dir = ensure_resized(df["file"], cfg["input"]["resize_cache"])
        tf = build_transforms(cfg["input"], cfg["augmentation"])[1]
        z = np.load(d / "scores.npz")
        out = {}
        for s in ("test_seen", "test_unseen"):
            sub = df[df["split"] == s]
            e = embed(model, DataLoader(ImageLabels(sub, image_dir, tf), batch_size=128, num_workers=2), "cuda", True)
            out[f"{s}_logp_crop"], out[f"{s}_logp_disease"] = e["logp_crop"], e["logp_disease"]
            out[f"{s}_labels"] = sub["pair_id"].to_numpy()
            S = e["logp_crop"][:, cid] + e["logp_disease"][:, did]
            gap = np.abs(S - z[f"{s}_scores"]).max()
            agree = (S.argmax(1) == z[f"{s}_scores"].argmax(1)).mean()
            print(f"{name} {s}: max |diff| vs saved scores {gap:.4f}, top-1 agreement {agree:.4f}", flush=True)
        np.savez_compressed(d / "test_parts.npz", **out)


if __name__ == "__main__":
    main()
