# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

COS30082 Applied Machine Learning (Sem 2, 2026) group project: **zero-shot / out-of-distribution plant disease identification**. The assignment spec is at `docs/Out of Distributon Plant Disease Identification v2.pdf`. Current state: dataset explorer app, shared evaluation code, zero-shot CLIP versions (v1-v3.9), the proposed model (compositional prompt learning on frozen CLIP: v4-v4.3; plus LoRA on the image encoder: v5-v5.7; plus a CNN ensemble: v6-v6.2), and the required CNN baselines (`aniq_baseline_independent_cnn`, `aniq_baseline_multitask_cnn`). Each config's `_result` records its outcome.

## Commands

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/streamlit run streamlit_app.py
.venv/Scripts/python -m train.zeroshot --config config/vlm/v3_knowledge_base_config.json      # method clip_zeroshot
.venv/Scripts/python -m train.train --config config/vlm/v4.2_multi_split_val_config.json         # method clip_prompt_learning
.venv/Scripts/python -m train.train_lora --config config/vlm/v5_lora_config.json    # method clip_prompt_lora
.venv/Scripts/python -m train.ensemble --config config/vlm/v6_cnn_ensemble_config.json     # method ensemble
.venv/Scripts/python -m train.train_cnn --config config/baseline/aniq_baseline_independent_cnn.json   # cnn_independent / cnn_multitask
.venv/Scripts/python -m train.train_cnn --config config/cnn/aniq_cnn_v1.json   # improved CNN (same script)
.venv/Scripts/python -m train.train_disentangle --config config/disentangle/victor_disentangle_v2.json   # cnn_disentangle
```

Training scripts live in `train/` and are run as modules from the project root (`python -m train.<script>`) so `dpd` imports resolve. Each script asserts the config's `method` matches. The Streamlit app caches CLIP models on the GPU (about 3.5 GB once two runs are loaded); stop it while training CNNs on the 6 GB GPU. On Windows, never run DataLoader-with-workers code from a stdin script (`python -`): spawned workers cannot re-import it and hang; use a file. On Windows the Python process can segfault (exit 139) at interpreter shutdown after torch is imported; runs finish and write outputs before that.

## Versioned configs

Configs are split by family, and every script takes `--config` and reads every decision from it (no hard-coded hyperparameters):

- `config/vlm/vN[.M]_<title>_config.json`: the vision-language (CLIP) versions. The main number is a branch (one idea), the decimal a variant within it: v1 base (zero-shot CLIP), v2 clean_names, v3 knowledge_base (symptom descriptions; v3.1-v3.9 prompt-text and backbone variants, zero-shot), v4 prompt_learning (learned crop/disease vectors; v4.1-v4.3), v5 lora (v4 + LoRA on the image encoder; v5.1-v5.7 ablations), v6 cnn_ensemble (v5-family run + the independent CNN; v6.1-v6.2). The title is also stored in the config's `title` field; `version` (e.g. `v5.3`) stays short and names the run folder `runs/vlm/vN.M/`. `list_runs` sorts variants next to their branch. To try a change, add the next variant number in its branch (or a new branch for a new idea) rather than editing an old config. Versions were renumbered into these branches on 2026-10-31; every config, metrics file and decision log was rewritten to the new numbers, but raw `train.log` files and anything written outside the repo (e.g. report drafts) still use the old flat numbers (v5-v19).
- `config/baseline/<name>.json`: the CNN baselines, named rather than numbered (`aniq_baseline_independent_cnn`, `aniq_baseline_multitask_cnn`). Outputs go to `runs/baseline/<name>/`. Their `version` field is the name. Keep these unchanged as the required baselines.
- `config/cnn/<name>.json`: improved CNNs built on the baselines (`aniq_cnn_v1`, ...), same `train/train_cnn.py`. Outputs go to `runs/cnn/<name>/`; the dashboard labels this family "proposed (CNN)", not baseline.

Keys starting with `_` are documentation: `_note` explains what changed and why, `_result` records the outcome after the run (fill it in by hand). Each section can carry its own `_note`. `output_dir` must match the family folder; each run also keeps a copy of its config as `runs/<family>/<name>/config.json`. `runs/` is gitignored; `runs/cache/` holds shared embeddings and resized images.

## Code layout

- `dpd/data.py`: the single place that knows the dataset layout (no Streamlit imports). `read_metadata()` returns all splits in one DataFrame with `split`, `source`, `is_unseen` columns; `pair_table()` maps pair id 0..332 to crop/disease ids and names. Dataset root defaults to `./DPD_Small`, override with the `DPD_ROOT` env var.
- `dpd/evaluate.py`: the shared protocol. Every model produces a score matrix (images × 333 pairs) for each test split and calls `evaluate()`, which ranks against all 333 pairs (generalised zero-shot) and returns Top-1/Top-5, per-class means, crop/disease accuracy, harmonic mean, and an unseen-only diagnostic. `calibrate()` subtracts `gamma` from seen-pair scores; gamma must be tuned without the test sets. `seen_confusion_matrix()` is 323 seen rows × 333 predicted columns. `hm_top1_over_gammas()` gives the top-1 HM for a whole gamma grid at once (checked identical to `evaluate()` on the baseline's test scores); use it for large validation sweeps.
- `dpd/clip_utils.py`: open_clip loading, prompt building from the config's `names`/`prompts` sections, and image/text embedding. Image embeddings from a frozen encoder are cached in `runs/cache/`.
- `train/zeroshot.py`: zero-shot CLIP runner (method `clip_zeroshot`). Fails on duplicate prompts across pairs (e.g. `banana_segatoka` and `banana_sigatoka` are separate classes, so they must not be renamed to the same text) and on prompts over 77 tokens.
- `train/train.py` + `dpd/prompt_learner.py`: the proposed model (v4-v4.3). `CompositionalPrompts` keeps CLIP frozen and learns one vector per crop and per disease, substituted for a placeholder token (`x`) in fixed prompt sentences; text sequences are truncated after EOT (safe under the causal mask) to save memory. Training runs on cached image embeddings. Two stages: select epoch + gamma on a validation split (`dpd/splits.py`: whole seen pairs held out as pseudo-unseen, plus a fraction of images as val-seen), then refit on all train and test once.
- `dpd/inference.py`: loads any finished run for the app. `list_runs` scans `runs/<family>/<name>/` (VLM versions first, then baselines); `load_run_model` returns a `ClipRunModel` or `CnnRunModel` with the same interface: `.predict(path or file-like)` -> (retrieval embedding, uncalibrated pair scores), `.train_bank(train_df)`, `.pairs`, `.gamma`, `.logit_scale`. CLIP runs use `pair_text_embeddings.npy` (or rebuild it); CNN runs load `<network>.pt` and save `train_features.npy` (backbone features for retrieval) in the run folder. Models take the image source rather than a decoded image because CNN inference must reproduce the training cache's downscale + JPEG round trip (`dpd/resize_cache.standardise`); fed full-size originals instead, they disagreed with the evaluated scores on 10-20% of top-1 predictions. Both paths are verified to reproduce the saved test scores.
- `train/train_lora.py` + `dpd/lora.py`: v5+. Same prompt model as `train/train.py`, plus LoRA adapters (rank/alpha/targets in the config's `image_encoder.lora`) on the last blocks of the CLIP image encoder, trained jointly with the prompt vectors. LoRA is attached with `torch.nn.utils.parametrize` because `nn.MultiheadAttention` reads `in_proj_weight`/`out_proj.weight` directly. Image embeddings change during training, so there is no embedding cache: images are read from the 256 px resize cache with augmentation (`clip_transforms` in `dpd/clip_utils.py`). A run saves `lora_image_encoder.pt` (adapters only) next to the prompt vectors; `ClipLoraRunModel` in `dpd/inference.py` rebuilds it for the app. Roughly 100-130 s per epoch on the RTX 4050.
- v5.1-v5.4 are single-change ablations of v5 on the same validation split (seed 42), each enabled by one config option of `train/train_lora.py`: `model.image_size` (v5.1, 336 px; needs `input.resize_cache` >= it and `runtime.grad_checkpointing_image`), `evaluation.tta` (v5.2, flip views averaged at validation and test; `ClipLoraRunModel` applies the same views), `image_encoder.lora` size (v5.3), `training.sampling` class-balanced (v5.4). Keep a change only if its validation HM beats v5's (51.2%) by more than 2 points: run-to-run noise is about 3 points (v5 retrained on the same split, seed and settings scored 48.3% vs 51.2%). Never decide on test numbers.
- `train/combine_ablations.py`: builds a combined config (v5.5) from the ablations whose validation HM beat the base by more than a margin (`--margin`, default 1 point; the v5.5 queue uses 2); reads only validation numbers, copies each winning variant's changed settings, and logs every include/exclude decision in the config's `_note` (and `runs/v5.5_decision.txt`). If the ensemble ablation also won (or with `--force-ensemble`, which the v5.5 queue uses), it writes an ensemble config on top (v6.1), recording why in its `_note`. `--alias-if-identical` copies the single qualifying variant's finished run instead of retraining an identical config (identical configs reproduce bit-for-bit here). Exits with code 2 if nothing qualified.
- `train/ensemble.py` (v6): mixes two finished runs on a log-probability scale (`w * A + (1 - w) * B`), choosing `w` and gamma on validation. Validation scores come from retraining each member on the validation split for its selected epoch (cached as `val_scores_<run>.npy`); test scores are the members' saved `scores.npz`. CNN members are scored with the crop weight and tau they chose (`crop_weight`/`tau` in their `metrics.json`, defaults otherwise), so validation and test scores match; the validation-score cache name includes those settings.
- Long queues: background tool calls are capped at 2 hours, so multi-run queues are launched as a detached PowerShell script (`runs/queue_*.ps1`, status in `runs/queue_status.txt`). In PowerShell 5.1 logs, `NativeCommandError` lines are only stderr warnings wrapped as error records, not failures. Those logs are UTF-16 (`iconv -f UTF-16LE -t UTF-8` to grep them). Memory: on Windows every DataLoader worker is a separate process committing ~1.4 GB, and `train_lora.py` keeps a training and a validation loader alive (2 x num_workers processes); with 4 workers one run commits ~20 GB, so a second job alongside can exhaust virtual memory (error 1455, 'paging file is too small', which crashed the first v5.1 attempt). Don't run two training jobs at once, or use 2 workers.
- `train/train_disentangle.py` + `dpd/disentangle.py` (configs in `config/disentangle/`): a teammate's disentangled multi-task CNN (crop and disease branches, adversarial "spy" heads behind gradient reversal), evaluated with the same protocol as the baselines (method `cnn_disentangle`).
- `train/train_cnn.py` + `dpd/cnn.py`: the CNN baselines (torchvision ResNet-50, one linear head per target). Pair scores = log P(crop) + log P(disease), so they go through the same 333-pair evaluation and gamma selection (gamma in log-prob units, hence a different `gamma_grid`). Images come from `dpd/resize_cache.py` (one-off copies at shorter side 256 in `runs/cache/resized_256/`). Do not use `channels_last`: on this machine (RTX 4050 Laptop, torch 2.11) it made ResNet-50 training 4.5x slower. An optional `composition` config section (used by `config/cnn/`) adds `crop_weight_grid` (score = 2·(w·log P(crop) + (1−w)·log P(disease)); w 0.5 is exactly the baseline sum) and `tau_grid` (post-hoc logit adjustment of each head by its training log prior), both chosen on validation with epoch and gamma; the run stores `crop_weight`/`tau` in `metrics.json` and `CnnRunModel` applies them, and `val.plain_scoring` records the same networks scored the plain way.
- `dpd/report.py`: writes `metrics.json`, `confusion_test_seen.csv`, `predictions_<split>.csv`, `scores.npz` identically for every method.
- `knowledge/disease_descriptions_v1.json`: one short visual symptom description per raw disease name, used via `prompts.descriptions_file`. Written from general knowledge, unverified.
- `streamlit_app.py` + `app_pages/`: multipage Streamlit app (`st.navigation`; pages are plain scripts). `dpd/ui.py` holds the Streamlit-only helpers (cached `load_metadata`, thumbnails, image grid, cached `get_model` / `get_train_bank`, and `test_predictions`: per-image top-k and outcome — correct / wrong crop / wrong disease / both wrong — read from a run's `scores.npz`, no GPU). `app_pages/inference.py` is the required UI and the default page: pick any finished run, then browse a grid of many test images (from saved scores) or several uploads (run live), filter by outcome, sort by confidence; *Details* opens a dialog that re-runs the model and shows the top-5 pairs, the nearest training images, and the training images of the same crop and the same disease (how an unseen pair is composed). `app_pages/dashboard.py` compares every run (charts, outcome breakdown, full metrics table, per-model details and top confusions).
- Streamlit caches thumbnails and models (`st.cache_data` / `st.cache_resource`). After changing anything under `dpd/`, restart the server: a running server can keep the old module and old cached objects (seen in practice: `'dict' object has no attribute 'pairs'`).

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
