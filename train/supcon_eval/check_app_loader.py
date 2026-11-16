"""Check that the app's loaders (dpd/inference.py) reproduce the evaluated test scores for the contrastive-family runs and the mix.
For each run: load it exactly as the Streamlit app does, score N test images from their ORIGINAL files (the app's path:
downscale + JPEG round trip, then the test transform) and compare with the rows saved in the run's scores.npz.
Usage (from the repo root, DPD_ROOT set):  python train/supcon_eval/check_app_loader.py [N] [run_version ...]"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dpd.data import IMAGE_DIR, read_metadata  # noqa: E402
from dpd.inference import list_runs, load_run_model  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    wanted = sys.argv[2:] or ["full_res336", "mix_res336_v103"]
    df = read_metadata()
    runs = {r["version"]: r for r in list_runs()}
    print(f"{len(runs)} runs listed by the app; contrastive family: {sum(r['family'] == 'contrastive' for r in runs.values())}")
    rng = np.random.default_rng(0)
    for v in wanted:
        run = runs[v]
        model = load_run_model(run, df)
        z = np.load(run["dir"] / "scores.npz")
        gaps, agree, total = [], 0, 0
        for split in ("test_seen", "test_unseen"):
            sub = df[df["split"] == split].reset_index(drop=True)
            for i in rng.choice(len(sub), n // 2, replace=False):
                emb, s = model.predict(IMAGE_DIR / sub["file"][i])
                saved = z[f"{split}_scores"][i]
                gaps.append(np.abs(s - saved).max())
                agree += int(s.argmax() == saved.argmax())
                total += 1
        bank_dim = emb.shape
        print(f"{v} ({run['method']}): {total} images, max |score difference| {max(gaps):.4f}, median {np.median(gaps):.4f}, "
              f"same top-1 pair on {agree}/{total}; retrieval embedding {bank_dim}, gamma {model.gamma}")
        del model


if __name__ == "__main__":
    main()
