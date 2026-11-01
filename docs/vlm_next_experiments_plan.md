# Plan: next three VLM experiments

Base: v5.5, validation HM 56.89%; v8, contrastive descriptions, scored 56.76% on validation, so it is not kept. Operation
ids refer to `docs/diagrams/vlm_pipeline.drawio`. The same rules apply as always: decide on validation only, run one
job at a time, and add a new config for each change.

## v9: post-hoc scoring knobs (primitive fusion + LoRA scale)

**Idea.** Prediction (H5) uses only the pair score, although the crop (E4) and disease (E5) prompts are trained every
step. The LoRA update (C6) is applied at full strength, although a partial update may trade seen against unseen better.
Neither knob changes training, so both can be swept per epoch on validation.

**Scoring.** For pair p with crop c(p) and disease d(p):

    s(p) = [pair(p) + a·crop(c(p)) + b·disease(d(p))] / (1 + a + b)        (then minus gamma if p is seen)

All three terms are cosines, and dividing by (1 + a + b) keeps the scores on the cosine scale, so the existing gamma grid
still fits. The image embedding is computed with the LoRA delta scaled by s_lora: W + s_lora·(alpha/r)·BA.

**Code changes** (each one off by default, so old configs behave exactly as before):

- `dpd/lora.py`: a `set_lora_scale(visual, s)` helper that sets a multiplier on every `LoRADelta`. Parametrizations
  recompute W on every forward, so changing the multiplier takes effect immediately.
- `train/train_lora.py`:
  - a new `evaluation.posthoc` section: `{"crop_weight_grid": [...], "disease_weight_grid": [...], "lora_scale_grid": [...]}`.
  - In `validate`: for each `lora_scale`, re-encode the validation images (~5 s each). Compute the pair, crop and
    disease cosines, and sweep (a, b, gamma) with `hm_top1_over_gammas`.
  - Keep the full grid in `val_hm_grid.npy` (epoch x s x a x b x gamma) and select the best overall.
  - Stage 2 uses the chosen (epoch, s, a, b, gamma).
  - Also save `crop_text_embeddings.npy` and `disease_text_embeddings.npy`, and write `lora_scale`, `crop_weight` and
    `disease_weight` into `metrics.json`.
- `dpd/inference.py` (`ClipLoraRunModel`): apply the same scale and fusion, so the app reproduces the saved test scores
  (verify on the test images, as before).
- `train/ensemble.py` (`member_val_scores`): score the CLIP members the same way, so the v6-style ensembles stay
  consistent.

**Grids.** a, b in {0, 0.25, 0.5, 1}; s in {0.5, 0.75, 1.0}. That is 48 combinations x 41 gammas per epoch, cheap with
the vectorised sweep.

**Decision.**

- The grid point a = b = 0, s = 1 *is* v5.5 (the same seed reproduces bit-for-bit), so the comparison is paired:
  - the same model and validation images, with and without the knobs;
  - report the HM gain, plus a bootstrap 95% CI over validation images.
- Keep the knobs if the gain is > 2 points **and** the CI excludes 0. Report "fusion only" (s = 1) and "scale only"
  (a = b = 0) separately, so we know which one helped.
- **Risk:** choosing 4 knobs on one split can overfit validation. If the gain is borderline, rerun stage 1 with
  `n_splits: 2`.

**Cost.** One run, about 50 min (training is unchanged; the extra validation encodes add ~15 s per epoch).

## v10.x: regularise toward frozen CLIP (single-change ablations)

Base: v9 if kept, otherwise v5.5. One change per config, judged against the base's validation HM; the winners are
combined afterwards with `train/combine_ablations.py`.

| Version | Change | Implementation | Extra cost |
|---|---|---|---|
| v10 | Image consistency (PromptSRC / CoPrompt) | `training.loss_weights.image_consistency`: L1 between the LoRA CLS embedding and the frozen-CLIP embedding of the same augmented image. The frozen embedding comes from a no_grad forward with the LoRA scale set to 0 (the helper from v9). | ~+40% time per epoch (one extra forward) |
| v10.1 | Text anchor (KgCoOp) | `training.loss_weights.text_anchor`: L1 between the current T_pair / T_crop / T_dis and their values at initialisation (computed once in `build`). | negligible |
| v10.2 | Weight averaging (PromptSRC self-ensemble) | `training.weight_average: {"type": "ema", "decay": 0.999}`: an EMA of the vectors and LoRA weights, updated every step. Validation and the final model use the EMA weights. | negligible |

- **Weights:** start from the papers' defaults (check PromptSRC's config for its L1 weights; KgCoOp uses 8 on its text
  term). The loss here is a CE scaled by 100, so the weights may need one rescale. Pick a single value first rather
  than a grid.
- **Smoke test:** before each long run, a tiny fit on 256 images, as for v7. Check the new loss is finite and its size
  is comparable to the CE.
- **Cost:** three runs, about 50-70 min each. They can go in one detached queue overnight (`runs/queue_v10.ps1`, same
  pattern as `queue_v7.ps1`).

## SCOLD zero-shot check (filter only)

**What it is** (from its Hugging Face repo `enalis/scold`, CC-BY-4.0):

- a timm Swin-B image encoder (`swin_base_patch4_window7_224`; the 7x7x1024 final map is flattened, then projected to
  512);
- a RoBERTa-base text encoder (CLS + mean pooling, projected to 512);
- 950 MB of weights in `scold.pth`, plus a custom `model.py`.

So it is **not** a CLIP architecture. None of the current prompt-learning or LoRA code applies to it directly.

**Check:**

1. Needs `transformers` (missing from `.venv`; timm 1.0.30 is installed) and the 950 MB download. I will ask before
   installing or downloading.
2. Add `dpd/scold.py`: rebuild `LVL` from its `model.py` (pin the NHWC/NCHW flatten order of timm's Swin output), load
   `scold.pth`, and expose `encode_image` / `encode_text` / tokenizer the way `train/zeroshot.py` expects. Route it via
   `model.library: "scold"`.
3. Use its own preprocessing: the README shows resize 224 + ToTensor; confirm whether it normalises.
4. Config v3.10_scold: v3's prompts on SCOLD, through `train.zeroshot` (a few minutes).

**Gate.**

- Worth pursuing only if zero-shot HM clearly beats OpenAI CLIP's v3 score (10.2%).
- If it passes, *training* it is a separate, larger task: a RoBERTa prompt learner (vectors substituted via
  `inputs_embeds`; no causal mask, so no EOT truncation) and LoRA targets for Swin.

**Leakage risk (must check before trusting any number).** SCOLD was pretrained on 186k leaf images, probably drawn from
public sets such as PlantVillage and PlantDoc, which may also be DPD-Small's sources (e.g. the `PD` prefix). If its
pretraining saw our test images, its scores are not comparable. Read its paper's data section. If there is overlap,
report SCOLD only with that caveat, or drop it.

## Order and time

1. v8 stage 2 finishes (running now); fill in its `_result`.
2. v9 (~50 min).
3. SCOLD check (~30 min of setup, after approval for the install and download). It can run while waiting on nothing
   else; it is not a training job.
4. The v10 queue (~3 h, can run overnight).
