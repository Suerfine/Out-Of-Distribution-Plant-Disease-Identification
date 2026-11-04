# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Personal notes (e.g. one member's pipeline details) go in `CLAUDE.local.md`, which is gitignored.

## Project

COS30082 Applied Machine Learning (Sem 2, 2026) group project: **zero-shot / out-of-distribution plant disease identification**. The spec is `docs/Out of Distributon Plant Disease Identification v2.pdf`; it is the source of truth, so re-read it when in doubt. Due 27/11/2026, 11:59pm.

The repo holds a dataset explorer / inference app, shared evaluation code, the required CNN baselines, and several proposed models: CLIP-based (zero-shot, prompt learning, LoRA, ensembles; `config/vlm/`), a disentangled multi-task CNN (`config/disentangle/`), and a multi-task CNN (`config/multitask/`). Each config's `_result` records its outcome.

## Commands

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/streamlit run streamlit_app.py
.venv/Scripts/python -m train.<script> --config <config.json>
```

Training scripts live in `train/` and are run as modules from the project root so `dpd` imports resolve. Each script asserts the config's `method` matches:

| Script | Method(s) | Configs |
|---|---|---|
| `train.train_cnn` | `cnn_independent`, `cnn_multitask` | `config/baseline/` |
| `train.train_multi_task_cnn` | `multi_task_cnn` | `config/multitask/` |
| `train.train_disentangle` | `cnn_disentangle` | `config/disentangle/` |
| `train.zeroshot`, `train.train`, `train.train_lora`, `train.ensemble` | CLIP methods | `config/vlm/` |

## Configs and runs

- Every script takes `--config` and reads every decision from it (no hard-coded hyperparameters). To try a change, add a new config rather than editing an old one.
- Keys starting with `_` are documentation: `_note` explains what changed and why, `_result` records the outcome after the run (filled in by hand). Each section can carry its own `_note`.
- `config/baseline/` holds the required baselines (`aniq_baseline_independent_cnn`, `aniq_baseline_multitask_cnn`). Keep them unchanged.
- `config/vlm/vN[.M]_<title>_config.json`: CLIP versions; the main number is a branch (one idea), the decimal a variant.
- `config/multitask/<name>.json`: multi-task CNN experiments (`nat_multitask_v1`, ...). Outputs go to `runs/multitask/<name>/`. Supports `--set` CLI overrides, backbone selection, custom scheduler/loss weighting, and checkpointing.
- Outputs go to `runs/<family>/<name>/` (`output_dir` must match the family folder), with a copy of the config. `runs/` is gitignored; `runs/cache/` holds shared embeddings and resized images.
- Choose epochs, thresholds and gamma on a validation split carved from train (`dpd/splits.py`), never on the test sets.
- **Recording results (`docs/all_results.csv`)**: After experiments have run, agents MUST update `docs/all_results.csv` with the experiment results. Agents must NOT modify, add, delete, or rename columns in `docs/all_results.csv`—only fill in the existing columns (`datetime`, `type`, `experiment_name`, `harmonic_mean_top1`, `harmonic_mean_top5`, `seen_top_1`, `seen_top1_class_mean`, `seen_top5`, `seen_top5_class_mean`, `seen_crop_top1`, `seen_disease_top1`, `unseen_top_1`, `unseen_top1_class_mean`, `unseen_top5`, `unseen_top5_class_mean`, `unseen_crop_top1`, `unseen_disease_top1`, `config`, `notes`). Include both uncalibrated ($\gamma = 0$) and calibrated ($\gamma > 0$) evaluations where applicable, clearly highlighting in the `notes` column that gamma was applied (e.g. `Calibrated (gamma=X applied)` vs `Uncalibrated (gamma=0.0)`).

## Code layout

- `dpd/data.py`: the single place that knows the dataset layout. `read_metadata()` returns all splits in one DataFrame (`split`, `source`, `is_unseen` columns); `pair_table()` maps pair id 0..332 to crop/disease ids and names. Dataset root defaults to `./DPD_Small`, override with `DPD_ROOT`.
- `dpd/evaluate.py`: the shared protocol. Every model produces a score matrix (images × 333 pairs) per test split and calls `evaluate()`, which ranks against all 333 pairs (generalised zero-shot) and returns Top-1/Top-5, per-class means, crop/disease accuracy, harmonic mean, and an unseen-only diagnostic. `calibrate()` subtracts `gamma` from seen-pair scores. `seen_confusion_matrix()` is 323 seen rows × 333 predicted columns. `hm_top1_over_gammas()` is a fast gamma sweep.
- `dpd/clip_utils.py`: open_clip loading, prompt building from the config's `names`/`prompts` sections, and image/text embedding. Image embeddings from a frozen encoder are cached in `runs/cache/`.
- `dpd/report.py`: writes `metrics.json`, `confusion_test_seen.csv`, `predictions_<split>.csv`, `scores.npz` identically for every method. Use it so the app and dashboard can read any run.
- `dpd/resize_cache.py`: one-off copies at shorter side 256 in `runs/cache/resized_256/`, shared by the image-based models.
- `dpd/inference.py`: loads any finished run for the app. `list_runs` scans `runs/<family>/<name>/`; `load_run_model` returns a model with `.predict(path or file-like)` -> (retrieval embedding, uncalibrated pair scores), `.train_bank(train_df)`, `.pairs`, `.gamma`, `.logit_scale`. Models take the image source rather than a decoded image, because inference must reproduce the training cache's downscale + JPEG round trip (`dpd/resize_cache.standardise`); fed full-size originals, predictions disagreed with the evaluated scores on 10-20% of images.
- `train_cnn.py` + `dpd/cnn.py`: the CNN baselines (torchvision ResNet-50, one linear head per target). Pair scores = log P(crop) + log P(disease), so they go through the same 333-pair evaluation and gamma selection (gamma in log-prob units, hence a different `gamma_grid`). Images come from `dpd/resize_cache.py` (one-off copies at shorter side 256 in `runs/cache/resized_256/`). Do not use `channels_last`: on this machine (RTX 4050 Laptop, torch 2.11) it made ResNet-50 training 4.5x slower.
- `train/train_multi_task_cnn.py`: tunable multi-task CNN trainer (method `multi_task_cnn`). One shared backbone (resnet, efficientnet, convnext, vit) with joint crop and disease heads, early stopping, periodic checkpointing every N epochs (`runtime.checkpoint_every_n_epochs`) with atomic `last.pt` and `--resume`. Notebook at `notebooks/train_multi_task_cnn_colab.ipynb`.
- `dpd/disentangle.py` + `train/train_disentangle.py`: disentangled multi-task CNN (crop and disease branches, adversarial "spy" heads behind gradient reversal), evaluated with the same protocol as the baselines.
- `streamlit_app.py` + `app_pages/`: multipage Streamlit app (`st.navigation`; pages are plain scripts). `dpd/ui.py` holds the Streamlit-only helpers (cached metadata, thumbnails, models, training banks, per-image test outcomes read from `scores.npz`). `app_pages/inference.py` is the required UI and the default page; `app_pages/dashboard.py` compares every run.

## Gotchas

- After changing anything under `dpd/`, restart the Streamlit server: a running server can keep the old module and cached objects (seen: `'dict' object has no attribute 'pairs'`).
- The app caches CLIP models on the GPU (about 3.5 GB with two runs loaded); stop it before training on the 6 GB GPU. Don't run two training jobs at once (Windows DataLoader workers each commit ~1.4 GB of memory).
- On Windows, never run DataLoader-with-workers code from a stdin script (`python -`): spawned workers cannot re-import it and hang; use a file.
- On Windows the Python process can segfault (exit 139) at shutdown after torch is imported; runs finish and write outputs before that.

## Task formulation

- Full category set `C = S ∪ U`. Train only on seen categories `S`; must recognise unseen categories `U`.
- Each category is a **crop–disease composition**: 55 crops, 175 diseases, 323 seen compositions, 10 unseen. All label indices start at 0.
- The natural framing is to predict crop and disease separately and compose them, so unseen pairs can be scored. Every unseen pair's crop and disease each appear in training (paired with other partners), though some only weakly (e.g. disease `aphis_spp` has 7 train images; crop `cucumber` has 36).
- Composition IDs share one 0–332 space; the 10 unseen IDs are interleaved (39, 84, 107, 124, 133, 136, 144, 166, 276, 303), not 323–332. Never assume unseen = last indices.

## Dataset: DPD-Small (required; do not use the full DPD dataset)

| Split | Images |
|---|---|
| Train | 8,000 |
| Test seen | 2,000 |
| Test unseen | 1,448 |

Local layout (gitignored, downloaded separately, never commit it): `DPD_Small/DPD_Small_dataset/` (all 11,448 images in one flat folder) and `DPD_Small/DPD_Small_{train,test_seen,test_unseen}.csv`. CSVs are comma-separated with **no header row** (`header=None`). Columns: image file name, crop label, disease label, composition label, crop name, disease name, composition name.

- Real-world field images (varied lighting, backgrounds, viewpoints, quality), not lab images.
- Very long-tailed: median 7 train images per seen pair, 139 pairs have ≤5, minimum 1 (`cassava_mosaic` has 544). The seen test set has a median of 2 images per pair.
- The file-name prefix (e.g. `CDDM`, `PD`) is the source dataset (46 sources, CDDM ≈ 52%). It correlates strongly with labels, so never feed it to a model. 4 sources (`PDL`, `PDL3`, `BLD`, `BLD3`) appear only in test seen.
- Some disease classes are near-duplicate names (e.g. `banana_segatoka` / `banana_sigatoka`); they are separate classes, never merge them.
- Mixed `.jpg`/`.jpeg`/`.png` files, so convert to RGB on load.

## Required components

1. **Baselines** (seen-domain training only), any modern backbone with transfer learning and augmentation:
   - *Independent*: two separate CNNs, one predicting crop (`N_c` classes), one predicting disease (`N_d` classes).
   - *Multi-task*: one shared backbone with two heads (crop + disease) trained jointly.
2. **Proposed approach**: at least one new deep-learning method that improves unseen generalisation and field robustness, compared against the baselines with the same evaluation protocol.
3. **User interface**: input is a plant image; output is the predicted crop–disease class plus the matching/relevant **training images** (retrieval over training-set embeddings).

## Evaluation protocol (must report all)

- Top-1 and Top-5 accuracy.
- Harmonic mean of seen and unseen accuracy: `H = 2·seen·unseen / (seen + unseen)`.
- Confusion matrix for the **seen** test set.

Keep evaluation code shared across all models (`dpd/evaluate.py`, `dpd/report.py`) so results are directly comparable. After runs finish, always log results into `docs/all_results.csv` without modifying the existing columns.

## Deliverables beyond code

8-page max report (Methodology; Results & Discussion, covering training scheme, losses, hyperparameters, and model comparison), ≤8-minute recorded presentation, and git repo read access for the lecturer.
