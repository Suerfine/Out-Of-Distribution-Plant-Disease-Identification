# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

COS30082 Applied Machine Learning (Sem 2, 2026) group project: **zero-shot / out-of-distribution plant disease identification**. The repository currently contains only the README, a Python `.gitignore`, and the assignment spec at `docs/Out of Distributon Plant Disease Identification v2.pdf`. There is no source code, dependency manifest, build, lint, or test setup yet — when these are added, update this file with the real commands rather than assuming any.

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
