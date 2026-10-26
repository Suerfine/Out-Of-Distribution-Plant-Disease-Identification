# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

COS30082 Applied Machine Learning (Sem 2, 2026) group project: **zero-shot / out-of-distribution plant disease identification**. The assignment spec is at `docs/Out of Distributon Plant Disease Identification v2.pdf`. Current state: dataset explorer app, shared evaluation code, zero-shot CLIP versions (v1-v3, v7), the proposed model (compositional prompt learning on frozen CLIP: v4-v6, v8), and the required CNN baselines (`aniq_baseline_independent_cnn`, `aniq_baseline_multitask_cnn`). Each config's `_result` records its outcome.

## Commands

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/streamlit run streamlit_app.py
.venv/Scripts/python zeroshot.py --config config/vlm/v3_config.json   # method clip_zeroshot
.venv/Scripts/python train.py --config config/vlm/v6_config.json      # method clip_prompt_learning
.venv/Scripts/python train_cnn.py --config config/baseline/aniq_baseline_independent_cnn.json   # method cnn_independent / cnn_multitask
```

Each script asserts the config's `method` matches. The Streamlit app caches CLIP models on the GPU (about 3.5 GB once two runs are loaded); stop it while training CNNs on the 6 GB GPU. On Windows, never run DataLoader-with-workers code from a stdin script (`python -`): spawned workers cannot re-import it and hang; use a file. On Windows the Python process can segfault (exit 139) at interpreter shutdown after torch is imported; runs finish and write outputs before that.

## Versioned configs

Configs are split by family, and every script takes `--config` and reads every decision from it (no hard-coded hyperparameters):

- `config/vlm/vN_config.json`: the vision-language (CLIP) versions, numbered v1, v2, ... Outputs go to `runs/vlm/vN/`. To try a change, copy the latest config to the next version number rather than editing an old one.
- `config/baseline/<name>.json`: the CNN baselines, named rather than numbered (`aniq_baseline_independent_cnn`, `aniq_baseline_multitask_cnn`). Outputs go to `runs/baseline/<name>/`. Their `version` field is the name.

Keys starting with `_` are documentation: `_note` explains what changed and why, `_result` records the outcome after the run (fill it in by hand). Each section can carry its own `_note`. `output_dir` must match the family folder; each run also keeps a copy of its config as `runs/<family>/<name>/config.json`. `runs/` is gitignored; `runs/cache/` holds shared embeddings and resized images.

## Code layout

- `dpd/data.py`: the single place that knows the dataset layout (no Streamlit imports). `read_metadata()` returns all splits in one DataFrame with `split`, `source`, `is_unseen` columns; `pair_table()` maps pair id 0..332 to crop/disease ids and names. Dataset root defaults to `./DPD_Small`, override with the `DPD_ROOT` env var.
- `dpd/evaluate.py`: the shared protocol. Every model produces a score matrix (images × 333 pairs) for each test split and calls `evaluate()`, which ranks against all 333 pairs (generalised zero-shot) and returns Top-1/Top-5, per-class means, crop/disease accuracy, harmonic mean, and an unseen-only diagnostic. `calibrate()` subtracts `gamma` from seen-pair scores; gamma must be tuned without the test sets. `seen_confusion_matrix()` is 323 seen rows × 333 predicted columns.
- `dpd/clip_utils.py`: open_clip loading, prompt building from the config's `names`/`prompts` sections, and image/text embedding. Image embeddings from a frozen encoder are cached in `runs/cache/`.
- `zeroshot.py`: zero-shot CLIP runner (method `clip_zeroshot`). Fails on duplicate prompts across pairs (e.g. `banana_segatoka` and `banana_sigatoka` are separate classes, so they must not be renamed to the same text) and on prompts over 77 tokens.
- `train.py` + `dpd/prompt_learner.py`: the proposed model. `CompositionalPrompts` keeps CLIP frozen and learns one vector per crop and per disease, substituted for a placeholder token (`x`) in fixed prompt sentences; text sequences are truncated after EOT (safe under the causal mask) to save memory. Training runs on cached image embeddings. Two stages: select epoch + gamma on a validation split (`dpd/splits.py`: whole seen pairs held out as pseudo-unseen, plus a fraction of images as val-seen), then refit on all train and test once.
- `dpd/inference.py`: loads a finished run (`list_runs` scans `runs/<family>/<name>/`, VLM versions first; `load_run_model`) and scores single images. Uses the run's `pair_text_embeddings.npy` (saved by `train.py` since v5) or rebuilds the embeddings from the config (zero-shot) or `prompt_vectors.pt` (v4); verified to reproduce the saved test scores.
- `train_cnn.py` + `dpd/cnn.py`: the CNN baselines (torchvision ResNet-50, one linear head per target). Pair scores = log P(crop) + log P(disease), so they go through the same 333-pair evaluation and gamma selection (gamma in log-prob units, hence a different `gamma_grid`). Images come from `dpd/resize_cache.py` (one-off copies at shorter side 256 in `runs/cache/resized_256/`). Do not use `channels_last`: on this machine (RTX 4050 Laptop, torch 2.11) it made ResNet-50 training 4.5x slower.
- `dpd/report.py`: writes `metrics.json`, `confusion_test_seen.csv`, `predictions_<split>.csv`, `scores.npz` identically for every method.
- `knowledge/disease_descriptions_v1.json`: one short visual symptom description per raw disease name, used via `prompts.descriptions_file`. Written from general knowledge, unverified.
- `streamlit_app.py` + `app_pages/`: multipage Streamlit app (`st.navigation`; pages are plain scripts). `dpd/ui.py` holds the Streamlit-only helpers (cached `load_metadata`, thumbnails, image grid). `app_pages/inference.py` is the required UI: pick any finished run from `runs/`, upload or pick a test image, and it shows the calibrated top-5 pairs, the nearest training images, and for the predicted pair the training images of the same crop and the same disease (how an unseen pair is composed).
- Streamlit caches thumbnails with `st.cache_data`. After changing `dpd/ui.py` or `dpd/data.py`, restart the server, because cached entries from the old code can be served otherwise.

The spec PDF is the source of truth for requirements; re-read it when in doubt. Due 27/11/2026, 11:59pm.

## Task formulation

- Full category set `C = S ∪ U`. Train only on seen categories `S`; must recognise unseen categories `U`.
- Each category is a **crop–disease composition**. 55 crop classes, 175 disease classes, 323 seen compositions, 10 unseen compositions. All label indices start at 0.
- Because labels are compositional, the natural framing is to predict crop and disease separately and compose them so unseen pairs can be scored. Verified: every unseen pair's crop and disease each appear in training (paired with other partners), though some only weakly (e.g. disease `aphis_spp` has 7 train images; crop `cucumber` has 36).
- Composition IDs share one 0–332 space; the 10 unseen IDs are interleaved (39, 84, 107, 124, 133, 136, 144, 166, 276, 303), not 323–332. Never assume unseen = last indices.

## Dataset: DPD-Small (required; do not use the full DPD dataset)

| Split | Images |
|---|---|
| Train | 8,000 |
| Test seen | 2,000 |
| Test unseen | 1,448 |

Local layout (gitignored): `DPD_Small/DPD_Small_dataset/` (all 11,448 images in one flat folder) and `DPD_Small/DPD_Small_{train,test_seen,test_unseen}.csv`. CSVs are comma-separated with **no header row** (use `header=None`). Column order:

1. image file name
2. crop label (int)
3. disease label (int)
4. crop–disease composition label (int)
5. crop name
6. disease name
7. crop–disease composition name

The images are real-world field conditions (varied lighting, backgrounds, viewpoints, quality), not lab images. The dataset is downloaded separately. Don't commit it to git.

Data characteristics that affect modelling:
- Very long-tailed: median 7 train images per seen pair, 139 pairs have ≤5, minimum 1 (`cassava_mosaic` has 544). The seen test set has a median of 2 images per pair.
- The file-name prefix (e.g. `CDDM`, `PD`) is the source dataset (46 sources, CDDM ≈ 52%). It correlates strongly with labels, so never feed it to a model. 4 sources (`PDL`, `PDL3`, `BLD`, `BLD3`) appear only in test seen.
- Mixed `.jpg`/`.jpeg`/`.png` files, so convert to RGB on load.

## Required components

1. **Baselines** (seen-domain training only), any modern backbone (ResNet, EfficientNet, ConvNeXt, ViT, …) with transfer learning and augmentation:
   - *Independent*: two separate CNNs, one predicting crop (`N_c` classes), one predicting disease (`N_d` classes).
   - *Multi-task*: one shared backbone with two heads (crop + disease) trained jointly.
2. **Proposed approach**: at least one new deep-learning method that improves unseen generalisation and field robustness (e.g. vision-language/CLIP-style, contrastive, self-supervised, compositional, feature disentanglement, domain adaptation, generative augmentation). It must be compared against the baselines using the same evaluation protocol.
3. **User interface**: input is a plant image. Output is the predicted crop–disease class plus the matching/relevant **training images** (so retrieval over training-set embeddings is needed).

## Evaluation protocol (must report all)

- Top-1 and Top-5 accuracy.
- Harmonic mean of seen and unseen accuracy on DPD-Small: `H = 2·seen·unseen / (seen + unseen)`.
- Confusion matrix for the **seen** test set.

Keep evaluation code shared across all models so baseline and proposed results are directly comparable.

## Deliverables beyond code

8-page max report (Methodology; Results & Discussion, covering training scheme, losses, hyperparameters, and model comparison), ≤8-minute recorded presentation, and git repo read access for the lecturer.
