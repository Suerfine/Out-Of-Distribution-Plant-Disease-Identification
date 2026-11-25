# vlm2: controlled, gamma-free ablation of the CLIP improvements

**How to run each stage, and the rules for choosing the final model: [RUNNING.md](RUNNING.md).**

Status (2026-11-23): **shelved, not run.** The report uses the baseline ladder built from existing runs instead (`docs/vlm/vlm_baseline_ladder.md`). Stage 0–2 configs are written. The queue scripts are `runs/queue_vlm2.ps1` (this laptop) and `config/vlm2/run_queue.sh` (Git Bash, Linux or Colab). Stage 3 (interactions) and stage 4 (final) configs are written once stage 2 results are in, because they depend on which improvements pass.

## Why a second round

The `config/vlm/` history is a chain, not a controlled experiment:

- **Parents differ.** Children branch off v5, v5.5, v5.8, v10.3 or v13, so two improvements are rarely measured against the same base.
- **Validation designs differ.** Some runs use 1 validation split and some use 3.
- **Changes are confounded.** v4.2 → v5 ("LoRA") also changed augmentation, batch size and epoch count.
- **Decisions mixed two effects.** They used calibrated HM, so a change in recognition and a change in seen bias could not be told apart (`docs/vlm/gamma0_audit.md`).
- **Validation and test disagree.** For CoOp, validation said +2.8 and gamma-free test said −0.7, and there was no gamma-free validation number to settle it.

vlm2 re-measures every improvement as a single change from **one common base**, with **gamma-free validation metrics** as the decision rule. Improvements that pass are then tested **in pairs**, to measure whether they add up (complementary) or replace each other (overlapping).

## Principles (fixed before any run)

1. **One base.** Every screening run is the base plus exactly one change, on the same 3 validation splits (seeds 42, 43, 44).
2. **Gamma-free decisions, HM selection.** Within a run, the epoch, post-hoc knobs and γ are chosen by mean validation HM, as in `config/vlm`. This was your choice (2026-11-23); `validation.select_by: "ausuc"` exists but is not used. Between runs, the decision metric is validation AUSUC at that selected epoch, the mean over the 3 splits. Secondary metrics are validation closed-set seen Top-1 and closed-set unseen Top-1. The protocol HM is reported next to them.
3. **A measured noise floor.** The base is trained twice: same splits, different initialisation seed. A change is real only if both hold:
   - its paired-bootstrap 95% interval on validation excludes 0;
   - |Δ AUSUC| is at least 2 × the base replicate's difference, and at least 1.5 pts.
4. **Complementary vs overlapping is tested, not assumed.** For improvements A and B, the interaction is I = Δ(A+B) − ΔA − ΔB.
   - **Complementary:** I is within noise (the gains add up).
   - **Overlapping:** Δ(A+B) ≈ max(ΔA, ΔB), i.e. I ≈ −min(ΔA, ΔB).
   - **Antagonistic:** Δ(A+B) < max(ΔA, ΔB).
5. **Test once.** Only the final combination and the base are evaluated on test. A fresh replicate of each guards against winner's curse.

## The base (B0)

B0 is v5.8's recipe. Every later improvement was built on it, and it holds the two components with by far the largest proven gains (gamma-free test: CSP +35 AUSUC, LoRA +17).

- Clean class names plus one symptom description per disease.
- CSP: learned crop and disease vectors.
- Fixed context "a photo of".
- LoRA on the last 8 image blocks, rank 16.
- v5's augmentation, 3 validation splits.

Effects are **conditional on this base**, and that is deliberate. The GAN shows why: it adds +4.2 AUSUC on frozen CLIP but −3.5 on LoRA features, so an add-on measured from zero-shot would mislead.

## Mechanism groups and pre-registered hypotheses

Improvements that act on the **same lever** are hypothesised to overlap. Improvements on **different levers** are hypothesised to be complementary. Existing evidence (test, gamma-free AUSUC; see `docs/vlm/gamma0_audit.md`) is listed for each group.

| Group | Lever | Improvements (config switch) | Prior evidence |
|---|---|---|---|
| **F: foundations** | Removed from B0 one at a time (leave-one-out) | no LoRA (`image_encoder.lora.last_n_blocks` 0), no CSP (`prompts.learn_primitives` false), no descriptions (`prompts.descriptions_file` none), raw class names | v3→v4 CSP +35.0. v4.2→v5 LoRA +16.9, confounded with augmentation; leave-one-out fixes this |
| **T: text representation** | What the class text embedding is | T1 CoOp context (`prompts.learned_context`), T2 multi-description (`descriptions_file` multi), T3 text anchor (`loss_weights.text_anchor`) | Mixed. CSP only 56.0, CoOp only 56.5, both 55.4: **likely overlapping** |
| **I: image adaptation** | What the image embedding is | I1 SSL local-global (`loss_weights.ssl`), I2 image consistency (`loss_weights.image_consistency`), I3 LoRA all 12 blocks, rank 32, I4 weight EMA | I1 +3.5 (best single gain); I2 +0.4; I3 +1.2, seen only |
| **C: composition** | How crop, disease and pair scores combine | C1 primitive branches (`model.primitive_branches`), C2 post-hoc primitive fusion (`evaluation.posthoc`; scoring only, no retrain), C3 DRPT alternating (`training.alternating_primitives`), C4 within-crop negatives | C1 +1.8, C2 +1.7, C3 +1.1, all from unseen gains; **likely overlapping with each other** |
| **G: classifier replacement** | Replaces the prompt scorer with a GAN-trained classifier on B0's features (`train_feature_gen`, `features.run` = B0) | G1 feature generation | Frozen CLIP +4.2, on LoRA −3.5: **overlapping or antagonistic with LoRA** (your intuition, already visible in the 2×2) |

Cross-group hypotheses: **T, I and C are complementary.** One data point supports this: primitive branches added +1.8 on CSP and +2.3 on CoOp + CSP. **G overlaps with LoRA and with the prompt classifier.**

Left out of the design:

- **Bias-only changes:** γ, the gated router, class-balanced sampling. They move the operating point and not recognition, so gamma-free metrics are blind to them by construction.
- **Large cost for no gain:** 336 px, TTA, patch disease.
- **Clearly worse:** BioCLIP.
- **The CNN ensemble.** It is orthogonal by construction (a different model), so it is added after the VLM is fixed.

## Run plan

| Stage | Runs | Purpose |
|---|---|---|
| 0. Base | B0, B0-r2 (same splits, new init seed) | Reference point and noise floor |
| 1. Foundations, leave-one-out | F1 no LoRA, F2 no CSP, F3 no descriptions, F4 raw names | What each core component contributes in context (your table) |
| 2. Screening, add-one | T1, T2, T3, I1, I2, I3, C1, C3, G1 (C2 is free: scored from B0's saved validation scores; I4 and C4 are optional) | Main effect of each improvement on B0 |
| 3. Interactions | Pairs among the stage-2 winners: one within-group pair per group (to confirm overlap), and every cross-group pair of group winners | Complementary vs overlapping, measured |
| 4. Final | Best of each group combined where complementary, plus a fresh replicate; B0 replicate on the same seeds; one test evaluation each | Final model and its honest gain |

Expected cost: about 100 min per 3-split LoRA run on this GPU (v5.11 86 min, v5.9 94, v5.13 123). The GAN arm takes about 70 min once features are cached.

| Variant | Runs | GPU time |
|---|---|---|
| Full (stages 0–4, about 3 winners) | about 20 | about 33 h |
| Lean: B0, B0-r2, F1, F2; T1, I1, C1, G1 (C2 free); 3 cross-group pairs; final plus replicate | about 13 | about 21 h |

Screening runs can skip the stage-2 test refit, saving about 10–15 min each. With the 27/11 deadline, only the lean variant fits alongside writing the report.

## Files and how to run

**Code changes:**
- `train/train_lora.py`:
  - every epoch now records validation AUSUC and closed-set seen/unseen Top-1 per split, in `val_gamma_free_grid.npz`, plus `val_gamma_free` and `test_gamma_free` in `metrics.json`;
  - `validation.train_seed_offset` sets a new initialisation on the same splits;
  - `validation.only` stops after stage 1 and writes `val_metrics.json`, not `metrics.json`, so the app ignores screening runs;
  - `validation.select_by` defaults to `"hm"`.
- `train/train_feature_gen.py`: the same γ-free metrics, plus `validation.save_scores`.
- `dpd/evaluate.py`: `closed_set_top1` and `gamma_free_metrics`.
- `train/vlm2_report.py`: the paired bootstrap against `s0_base` on the same validation images, the noise floor from `s0_base_r2`, the decision rule, C2 (fusion) scored from `s0_base`'s saved scores, and the interaction test for any run with two or more `_vlm2.factors`. It writes `runs/vlm2/vlm2_report.csv` and `docs/vlm/vlm2_report.md`.

**Configs.** Each config is `config/vlm2/<name>_config.json`, with output in `runs/vlm2/<name>/`. Each carries `_vlm2` (stage, group, factors), and every one except `s2_G1_gan` was checked to differ from `s0_base` by exactly its change. Stage-1/2 LoRA runs are validation-only. `s0_base` and `s2_G1_gan` are full runs.

**Queue.** `runs/queue_vlm2.ps1`, detached, with progress in `runs/queue_status.txt`. It runs in this priority order and regenerates the report after every run:

1. `s0_base`, `s0_base_r2`
2. `s1_no_lora`, `s1_no_csp`
3. `s2_T1_coop`, `s2_I1_ssl`, `s2_C1_primitive_branches`, `s2_G1_gan`
4. `s2_C3_drpt`, `s2_I3_lora_all_blocks`, `s2_T3_text_anchor`, `s2_I2_image_consistency`, `s2_T2_multi_description`
5. `s1_no_descriptions`, `s1_raw_names`
6. `s2_C4_within_crop`, `s2_I4_ema`

**Stage 3 configs.** Copy the single-factor configs being combined, merge their changes, and set `_vlm2.factors` to both factor ids, e.g. `["T1", "C1"]`.

## Limits worth stating in the report

- **Effects are conditional on B0.** A different base could reorder them, as the GAN does.
- **Validation has 20 pseudo-unseen pairs per split.** These are real seen pairs held out, so validation AUSUC estimates the real 10 unseen pairs only loosely.
- **Pairwise interactions do not rule out 3-way effects.** The final replicate is the check on the combination as a whole.
