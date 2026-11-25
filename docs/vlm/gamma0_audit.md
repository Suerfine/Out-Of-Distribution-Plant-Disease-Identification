# Gamma-0 audit: what each iteration changed with calibration off

Audit date 2026-11-23. Every finished run with saved scores (66 runs: VLM v1–v13.1 and v15, ensembles, CNN baselines, crop-norm) was re-scored from its saved raw score matrices with no retraining. Script: `scratch/gamma0_audit.py`. Full numbers: `docs/vlm/gamma0_audit.csv` (one row per run), `docs/vlm/gamma0_audit_deltas.csv` (one row per iteration step), `docs/vlm/gamma0_audit_val_members.csv` (gamma-free validation metrics).

## Bottom line

1. **Test HM at γ = 0 does not measure what the model learned. It measures how seen-biased the scores are.** Across the 25 LoRA runs (v5–v13), HM at γ=0 correlates **1.00** with "share of unseen-test images whose best unseen pair beats all 323 seen pairs". Its correlation with how well the model actually tells pairs apart is about zero: closed-set unseen accuracy −0.08, closed-set seen accuracy +0.10, AUSUC +0.13. It also swings more than any other metric. Its standard deviation across those runs is 4.9 pts, against 2.0 for calibrated HM and 1.3 for AUSUC.
2. **So γ is not hiding recognition gains. It is removing the seen-vs-unseen score offset**, and that offset is the only thing HM at γ=0 picks up. Calibrated HM tracks the gamma-free AUSUC reasonably well (r = 0.74 within LoRA; 0.95 over all trained CLIP runs). It sits on average 2.1 pts below the best HM any γ could reach on test, so the γ choice itself costs little.
3. **To explain mechanisms, report gamma-free metrics instead of turning γ off.** These are closed-set seen Top-1, closed-set unseen Top-1 and AUSUC. Report the bias gap as a separate number (how seen-biased the model is). These separate "recognises better" from "is less seen-biased", and γ=0 HM mixes the two. Several mechanisms only show up this way (see "Cases where a single HM misleads").
4. **Turning γ off for model selection does not work either.** At γ=0 the validation HM is 5–17% for every LoRA run, because pseudo-unseen images are almost never predicted as unseen. Selecting the epoch on it picks **epoch 1** for 12 of the 31 trained CLIP runs that have a validation grid, because the least-trained model is the least seen-biased.

## How the audit was computed

- `γ` (calibrated stacking) subtracts a constant from every seen pair's score before ranking (`dpd/evaluate.calibrate`). All `scores.npz` files hold the raw, uncalibrated scores, so the γ=0 metrics are exact. The calibrated HM was recomputed too, and it matches every `metrics.json`, with one exception: `aniq_baseline_multitask_cnn` differs by one seen-test image (50.65% vs 50.70%), which is probably a tie-break difference.
- **Closed-set seen Top-1**: seen-test images ranked among the 323 seen pairs only. **Closed-set unseen Top-1**: unseen-test images ranked among the 10 unseen pairs only (the existing `diagnostic_unseen_only_top1`). Neither depends on γ.
- **AUSUC**: area under the seen/unseen accuracy curve as γ sweeps from −∞ to +∞ (`dpd/evaluate.ausuc`). It does not depend on γ. It is 1.0 for a model perfect on both splits at once.
- **Bias gap**: median, over unseen-test images, of (best seen score − best unseen score). This is the offset γ has to cancel. It is in cosine units for CLIP runs and log-probability units for CNNs and ensembles.
- Test numbers at γ=0 come from the model refit at the epoch chosen *with* γ. There is no saved test model for the γ=0 epoch choice.
- Validation numbers come from `val_hm_grid.npy` at the γ=0 column, at the selected epoch and knobs. 1-split runs (v5–v5.7, v7–v10.2, v13) and 3-split runs (v4.2, v4.3, v5.8+, v10.3, v11, v12) are **not directly comparable**. Steps that cross the two (v4.2→v5, v5.5→v5.8) mostly show split noise in the validation column.
- Gamma-free metrics are mostly **test** numbers. Per the decision rule they explain results, but they must not be used to choose between models. The only gamma-free **validation** numbers available are for the members whose validation scores the ensembles cached (all on the seed-42 split); they are in the last table.

## Why HM at γ=0 behaves this way

At γ=0, under 1% of seen-test images go to an unseen pair in any LoRA run (at most 3% for v4 and v11), so seen accuracy at γ=0 equals closed-set seen accuracy to within 0.9 pts in every trained CLIP run. Unseen accuracy at γ=0 is, to within a few points, the share of unseen images routed to an unseen pair at all (median ratio 0.96). The HM is therefore roughly `2·S·r/(S+r)`, where `r` is the routing rate, and `r` depends on the seen/unseen score offset. That offset changes with things unrelated to recognition: the epoch count, the LoRA scale, prompt wording, loss weights and EMA. Calibrated HM cancels the offset per model, which is why it is the more stable number.

Examples of a change in HM at γ=0 that is only a change in offset:

- **v5 → v5.3/v5.5 (bigger LoRA)**: test HM at γ=0 went up 14.2 pts, from 26.1 to 40.3. Closed-set seen +0.4, closed-set unseen −0.6, AUSUC +0.6, so test recognition was unchanged. The bias gap shrank from 0.027 to 0.019. On **validation**, the opposite holds. HM at γ=0 was identical (10.88 vs 10.87), while the gamma-free numbers show a real gain: closed-set unseen 48.0 → 57.0, AUSUC 34.6 → 42.6. Here γ=0 hid the improvement that calibrated validation HM (+5.7) correctly reported.
- **v13 → v13.1 (SSL weight 5 → 20)**: −12.3 HM at γ=0. AUSUC −1.6, so part of this is a real regression and most of it is offset.
- **v15 (gated router)**: HM at γ=0 is 75.0, the same as v6.3's calibrated HM, and its AUSUC is 1.5 lower. The gate has learned to do γ's job. It adds no recognition.
- **v11.3 vs v10.3 (same LoRA features, GAN classifier vs learned prompts)**: v11.3 has the highest HM at γ=0 of any single model (44.0 vs 33.5), because its bias gap is 2.5× smaller. Its test AUSUC is 3.5 lower (52.7 vs 56.2). On validation they are about level (AUSUC 45.2 vs 44.2).

## Cases where a single HM misleads (mechanism notes)

These are the steps where the gamma-free metrics tell a different story from calibrated HM, from HM at γ=0, or from both:

| Step | What a single HM says | What the gamma-free metrics show |
|---|---|---|
| v4 → v4.1 (gamma_fix) | Calibrated HM +9.3 | The whole gain is the better γ choice. Closed-set unseen fell 6.8 pts and AUSUC is flat (+0.5). Unseen recognition got **worse**. |
| v4.2 → v5 (LoRA) | HM at γ=0 +0.1 (looks like nothing) | Closed-set seen +10.8, closed-set unseen +11.1, AUSUC +16.9, the largest single gain after v4. HM at γ=0 hides it completely. |
| v3 → v3.3 / v3.4 (taxonomy text, zero-shot, no γ at all) | HM −2.2 / −1.7, rejected | Closed-set unseen +8.2 / +7.0, AUSUC +1.3 / +0.9. Taxonomy text helps tell the unseen pairs apart. HM fell only because fewer unseen images were routed to unseen pairs, which is the same offset problem with γ fixed at 0. |
| v5.8 → v5.11 (primitive branches) | HM at γ=0 −4.8 | Closed-set unseen +5.7, AUSUC +1.8, closed-set seen −2.3. Separate crop and disease branches compose better to unseen pairs and give up some seen accuracy. |
| v5.13 → v5.14 (+ primitive branches) | Calibrated HM +2.0, validation +0.5 | Closed-set unseen +3.7, AUSUC +2.3, the same mechanism as v5.11, reproduced. |
| v5.5 → v9 (post-hoc fusion) | HM at γ=0 −8.1 | Closed-set unseen +3.9, AUSUC +1.7. Fusing in the primitive scores helps unseen composition, as in v5.11. |
| v5.8 → v5.9 (DRPT alternating) | HM at γ=0 −7.3 | Closed-set unseen +3.5, AUSUC +1.1. |
| v5.5 → v13 (SSL local-global) | Validation −0.4, rejected | Test AUSUC +3.5 (largest among v5.5's single-change children), closed-set seen +1.7, closed-set unseen +3.0. This is test-only evidence on a 1-split validation, so it **cannot** reverse the decision. It is the strongest candidate for a 3-split re-validation. |
| v10.3 → v12 (LoRA all blocks) | HM at γ=0 −6.5 | Closed-set seen +2.5, closed-set unseen −0.9, AUSUC +1.2. More capacity helps seen pairs only. |
| v11 → v11.2 (no generation) | HM at γ=0 = 0 | Closed-set unseen −24.2. The generated features are what make unseen pairs recognisable at all. |
| v6.x ensembles | HM at γ=0 20–30% | The ensembles need γ ≈ 5–7 log-prob units because the CNN member's bias gap (4.9–5.3) is larger than CLIP's (3.3–4.0). v6.7 has the best AUSUC of any run (69.6). |
| CLIP vs ConvNeXt CNN | CLIP's HM at γ=0 is far higher (26–44 vs 10–25) | On gamma-free numbers the ConvNeXt-336 CNN is ahead. Validation AUSUC is 53.2 vs 42–45 for CLIP, and test closed-set unseen is 91.9 vs 84–91 for the LoRA models. CLIP's advantage at γ=0 is a smaller seen bias, not better unseen recognition. This matters for how the report frames the proposed method. |

## Data issues found

- **`val_hm_g0_mean` was wrong for v11–v11.4 (fixed 2026-11-23).** `train/train_feature_gen.py` stored `mean_grid[e, 0]`, but these configs' γ grid starts at −0.1, so index 0 was γ=−0.1 and the stored value was 0.0. The true γ=0 validation HM is v11 15.1, v11.1 5.0, v11.2 0.0, v11.3 28.6 and v11.4 18.1 (at the selected epoch). The trainer now uses `dpd.evaluate.gamma0_index`. These runs' `metrics.json` and `history.csv` were corrected from `val_hm_grid.npy`, and each `metrics.json` records the correction in `val_hm_g0_fix`. The same `[..., 0]` assumption exists in the other trainers, and is only safe while their grids start at 0. v6.8's grid also starts at −5, but ensembles do not store this field.
- `docs/all_results.csv` has no rows (calibrated or uncalibrated) for v5.9–v5.14, the v6.x ensembles, the `aniq_*` baselines or crop-norm. The audit CSV has every number those rows need.
- `v5.5` is a copy of `v5.3` (`copied_from`), so their rows are identical by construction.

## Recommendation for the write-up

For each iteration, report three numbers alongside the calibrated HM:

- **AUSUC** (overall quality, γ-free)
- **Closed-set unseen Top-1** (composition to new pairs)
- **Closed-set seen Top-1** (seen recognition)

Then describe the bias gap separately, as the reason calibration is needed. This explains each mechanism without the offset noise, and keeps the protocol-required calibrated HM as the headline. HM at γ=0 can stay in the results table (AGENTS.md asks for it), labelled as a measure of seen bias, not of model quality. To make this decision-grade rather than test-only, save validation scores for future runs (`validation.save_scores: true` in `train_lora.py`) and compute validation AUSUC.

## Every iteration step (Δ in points; parent = the run the config's `_note` says it changed one thing from)

| Step | Change | Δ Val HM (γ) | Δ Test HM (γ) | Δ Test HM (γ=0) | Δ Closed seen | Δ Closed unseen | Δ AUSUC |
|---|---|---|---|---|---|---|---|
| v1 → v2 | clean_names | – | +0.3 | +0.3 | +0.9 | +4.1 | +0.6 |
| v2 → v3 | knowledge_base | – | +4.5 | +4.5 | +0.7 | +1.2 | +0.8 |
| v3 → v3.1 | random_text | – | -4.7 | -4.7 | -3.5 | -5.5 | -1.9 |
| v3 → v3.2 | descriptor_avg | – | -6.3 | -6.3 | -0.7 | +0.5 | -0.6 |
| v3 → v3.3 | taxonomy | – | -2.2 | -2.2 | +2.1 | +8.2 | +1.3 |
| v3 → v3.4 | taxonomy_symptoms | – | -1.7 | -1.7 | +1.7 | +7.0 | +0.9 |
| v3.4 → v3.5 | symptoms_first | – | +0.1 | +0.1 | -0.1 | -2.3 | -0.3 |
| v3.4 → v3.6 | separate_prompts | – | +3.5 | +3.5 | +0.9 | +3.2 | +1.0 |
| v3.6 → v3.7 | long_separate | – | -1.8 | -1.8 | +1.7 | -0.8 | +0.7 |
| v3 → v3.8 | long_symptoms | – | -2.3 | -2.3 | +1.0 | -1.2 | +0.2 |
| v3 → v3.9 | bioclip | – | -6.9 | -6.9 | -4.8 | -35.0 | -3.1 |
| v3 → v4 | prompt_learning | – | +33.4 | +24.7 | +43.7 | +32.2 | +35.0 |
| v4 → v4.1 | gamma_fix | – | +9.3 | -8.8 | +6.7 | -6.8 | +0.5 |
| v4.1 → v4.2 | multi_split_val | – | -1.1 | -0.1 | -1.9 | +1.8 | -0.5 |
| v4.2 → v4.3 | bioclip | +1.5 | -3.2 | -9.2 | +0.8 | -16.0 | -6.8 |
| v4.2 → v5 | lora | +4.7 | +13.3 | +0.1 | +10.8 | +11.1 | +16.9 |
| v5 → v5.1 | 336px | -0.2 | -1.4 | +1.8 | -0.1 | -2.5 | -1.6 |
| v5 → v5.2 | tta | +0.4 | -1.2 | +2.1 | -0.5 | +1.5 | +0.6 |
| v5 → v5.3 | lora_large | +5.7 | -0.6 | +14.2 | +0.4 | -0.6 | +0.6 |
| v5 → v5.4 | class_balanced | -2.2 | -2.8 | +6.8 | -1.8 | -1.1 | -1.6 |
| v5 → v5.5 | combined | +5.7 | -0.6 | +14.2 | +0.4 | -0.6 | +0.6 |
| v5.5 → v5.6 | prompt_swap | – | -3.0 | -8.1 | -3.6 | +3.0 | -1.2 |
| v5.5 → v5.7 | two_sentence | -1.3 | +2.0 | -7.5 | +0.4 | -0.8 | -0.1 |
| v5.5 → v5.8 | three_splits | +2.9 | -0.3 | -0.5 | +0.8 | -0.9 | +0.1 |
| v5.8 → v5.9 | drpt_alternating | +0.8 | +3.2 | -7.3 | -1.2 | +3.5 | +1.1 |
| v5.8 → v5.10 | multi_description | +0.9 | +2.8 | -0.4 | +0.0 | +1.2 | +1.0 |
| v5.8 → v5.11 | primitive_branches | +2.5 | +4.1 | -4.8 | -2.3 | +5.7 | +1.8 |
| v5.8 → v5.12 | coop | -0.6 | +2.8 | -11.2 | -1.8 | +2.8 | +0.4 |
| v5.8 → v5.13 | coop_csp | +2.8 | +2.1 | -2.7 | -2.0 | +1.1 | -0.7 |
| v5.13 → v5.14 | coop_csp_primitive_branches | +0.5 | +2.0 | -2.1 | +0.1 | +3.7 | +2.3 |
| v5.5 → v7 | patch_disease | +1.5 | -2.3 | -1.9 | -0.5 | +0.8 | -0.7 |
| v5.5 → v7.1 | zoom_crops | -1.6 | +1.9 | -2.7 | -0.4 | +1.1 | -0.0 |
| v5.5 → v7.2 | within_crop_negatives | -0.5 | +0.2 | +0.1 | -0.6 | -0.1 | -0.4 |
| v5.5 → v9 | posthoc_fusion_lora_scale | +1.4 | +3.2 | -8.1 | -0.8 | +3.9 | +1.7 |
| v5.5 → v10 | image_consistency | +0.5 | +0.0 | +0.9 | -0.5 | +1.0 | +0.4 |
| v5.5 → v10.1 | text_anchor | +1.4 | +0.9 | -4.4 | -1.0 | +0.1 | -1.0 |
| v5.5 → v10.2 | weight_ema | +0.4 | +0.1 | -1.1 | +0.6 | -1.0 | -0.1 |
| v5.8 → v10.3 | text_anchor_three_splits | +1.4 | +2.2 | -6.3 | -0.9 | +1.7 | +0.2 |
| v10.3 → v12 | lora_all_blocks | +0.7 | +1.6 | -6.5 | +2.5 | -0.9 | +1.2 |
| v5.5 → v13 | ssl_local_global | -0.4 | +1.7 | +0.9 | +1.7 | +3.0 | +3.5 |
| v13 → v13.1 | ssl_weight20 | -0.9 | +2.2 | -12.3 | -0.8 | -1.3 | -1.6 |
| v11 → v11.1 | compositional_condition | +1.5 | +1.9 | -18.0 | -1.7 | +3.7 | +1.6 |
| v11 → v11.2 | no_generation | -12.2 | -15.8 | -24.2 | -3.8 | -24.2 | -17.7 |
| v11 → v11.3 | lora_features | +11.0 | +7.1 | +19.9 | +3.2 | +11.2 | +10.1 |
| v11.1 → v11.4 | lora_compositional | +11.1 | +6.5 | +25.2 | +7.1 | +6.8 | +10.0 |
| v5 → v6 | cnn_ensemble | +1.8 | +5.1 | -6.4 | +4.1 | +2.6 | +5.7 |
| v6 → v6.1 | lora_large_ensemble | +5.8 | +0.6 | +5.9 | -0.1 | -0.4 | +0.3 |
| v6.1 → v6.2 | two_sentence_ensemble | -0.7 | -0.3 | -3.3 | +0.5 | -0.7 | -0.1 |
| v6.1 → v6.3 | convnext_v10.3_ensemble | +7.1 | +4.2 | +1.3 | +3.3 | +6.3 | +7.2 |
| v6.3 → v6.4 | convnext336_ensemble | +1.6 | -0.4 | -7.4 | +0.1 | -1.4 | -0.9 |
| v6.3 → v6.5 | convnext_v5.13_ensemble | +1.1 | +1.0 | -0.4 | +0.0 | -0.2 | +0.1 |
| v6.4 → v6.6 | convnext336_v5.13_ensemble | +0.6 | +0.9 | +0.5 | +0.3 | +0.1 | +0.6 |
| v6.6 → v6.7 | three_member_ensemble | +0.4 | +1.6 | +3.3 | +0.8 | +1.2 | +1.5 |
| v6.3 → v6.8 | convnext_v11.3_ensemble | +2.8 | -0.8 | +2.5 | +0.2 | +0.5 | +0.1 |
| v6.3 → v15 | gated_router | – | -0.0 | +48.0 | +0.0 | +0.0 | -1.5 |
| aniq_baseline_independent_cnn → aniq_baseline_independent_convnext_base |  | +9.2 | +8.9 | +10.2 | +7.0 | +15.4 | +15.4 |
| aniq_baseline_independent_convnext_base → aniq_independent_convnext_base_336_1split |  | +3.5 | +0.6 | -7.3 | +0.2 | +0.8 | +1.5 |
| aniq_cropnorm_v0 → aniq_cropnorm_v1 |  | +2.2 | -2.4 | -8.9 | -2.8 | -1.5 | -3.4 |
| aniq_cropnorm_v1 → aniq_cropnorm_v2 |  | +0.4 | +1.3 | -5.0 | -0.7 | +3.5 | +1.9 |
| aniq_cropnorm_v2 → aniq_cropnorm_v3 |  | -1.9 | -1.6 | +1.2 | +1.7 | -3.7 | -1.7 |
| aniq_cropnorm_v3 → aniq_cropnorm_v4 |  | +0.5 | +2.1 | +16.5 | -4.6 | +3.7 | +0.0 |

Validation Δ crosses 1-split and 3-split runs for v4.2 → v5 and v5.5 → v5.8, and v4/v4.1 have no validation grid. The baseline and crop-norm rows use the same columns.

## Every run (%, test unless marked Val)

| Run | Title | γ | Test HM (γ) | Test HM (γ=0) | Seen γ=0 | Unseen γ=0 | Closed seen | Closed unseen | AUSUC | Val HM (γ) | Val HM (γ=0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | base | 0 | 5.4 | 5.4 | 6.4 | 4.6 | 6.5 | 43.1 | 2.0 | – | – |
| v2 | clean_names | 0 | 5.6 | 5.6 | 7.2 | 4.6 | 7.3 | 47.2 | 2.6 | – | – |
| v3 | knowledge_base | 0 | 10.2 | 10.2 | 7.8 | 14.6 | 8.0 | 48.5 | 3.4 | – | – |
| v3.1 | random_text | 0 | 5.5 | 5.5 | 4.3 | 7.4 | 4.5 | 43.0 | 1.5 | – | – |
| v3.2 | descriptor_avg | 0 | 3.8 | 3.8 | 7.1 | 2.6 | 7.3 | 49.0 | 2.8 | – | – |
| v3.3 | taxonomy | 0 | 8.0 | 8.0 | 9.8 | 6.7 | 10.1 | 56.6 | 4.7 | – | – |
| v3.4 | taxonomy_symptoms | 0 | 8.4 | 8.4 | 9.5 | 7.6 | 9.7 | 55.5 | 4.3 | – | – |
| v3.5 | symptoms_first | 0 | 8.5 | 8.5 | 9.4 | 7.7 | 9.6 | 53.2 | 4.1 | – | – |
| v3.6 | separate_prompts | 0 | 11.9 | 11.9 | 10.3 | 14.1 | 10.5 | 58.7 | 5.3 | – | – |
| v3.7 | long_separate | 0 | 10.1 | 10.1 | 12.0 | 8.7 | 12.2 | 57.9 | 6.0 | – | – |
| v3.8 | long_symptoms | 0 | 7.9 | 7.9 | 8.9 | 7.1 | 9.0 | 47.2 | 3.6 | – | – |
| v3.9 | bioclip | 0 | 3.3 | 3.3 | 3.1 | 3.5 | 3.2 | 13.5 | 0.3 | – | – |
| v4 | prompt_learning | 0.045 | 43.5 | 34.9 | 50.8 | 26.6 | 51.7 | 80.7 | 38.4 | – | – |
| v4.1 | gamma_fix | 0.0375 | 52.9 | 26.1 | 57.6 | 16.9 | 58.4 | 73.8 | 38.9 | – | – |
| v4.2 | multi_split_val | 0.0375 | 51.8 | 26.0 | 55.5 | 17.0 | 56.4 | 75.6 | 38.4 | 46.6 | 15.3 |
| v4.3 | bioclip | 0.07 | 48.5 | 16.8 | 56.9 | 9.9 | 57.2 | 59.6 | 31.6 | 48.1 | 10.1 |
| v5 | lora | 0.0725 | 65.1 | 26.1 | 67.0 | 16.2 | 67.2 | 86.7 | 55.3 | 51.2 | 10.9 |
| v5.1 | 336px | 0.0725 | 63.7 | 27.9 | 66.9 | 17.6 | 67.2 | 84.2 | 53.8 | 51.0 | 9.6 |
| v5.2 | tta | 0.0725 | 63.9 | 28.3 | 66.5 | 18.0 | 66.8 | 88.1 | 55.9 | 51.6 | 9.6 |
| v5.3 | lora_large | 0.075 | 64.5 | 40.3 | 67.6 | 28.7 | 67.7 | 86.1 | 56.0 | 56.9 | 9.2 |
| v5.4 | class_balanced | 0.0725 | 62.3 | 32.9 | 65.2 | 22.0 | 65.4 | 85.6 | 53.7 | 49.0 | 9.6 |
| v5.5 | combined | 0.075 | 64.5 | 40.3 | 67.6 | 28.7 | 67.7 | 86.1 | 56.0 | 56.9 | 9.2 |
| v5.6 | prompt_swap | 0.075 | 61.5 | 32.2 | 63.7 | 21.6 | 64.0 | 89.0 | 54.7 | – | – |
| v5.7 | two_sentence | 0.075 | 66.5 | 32.8 | 68.0 | 21.6 | 68.0 | 85.2 | 55.8 | 55.6 | 9.9 |
| v5.8 | three_splits | 0.085 | 64.2 | 39.8 | 68.3 | 28.1 | 68.4 | 85.2 | 56.0 | 59.8 | 14.8 |
| v5.9 | drpt_alternating | 0.075 | 67.3 | 32.5 | 67.0 | 21.5 | 67.2 | 88.6 | 57.2 | 60.6 | 11.7 |
| v5.10 | multi_description | 0.075 | 66.9 | 39.5 | 68.3 | 27.8 | 68.4 | 86.4 | 57.0 | 60.7 | 16.3 |
| v5.11 | primitive_branches | 0.0325 | 68.2 | 35.0 | 66.0 | 23.8 | 66.1 | 90.9 | 57.8 | 62.2 | 13.7 |
| v5.12 | coop | 0.07 | 67.0 | 28.6 | 66.3 | 18.2 | 66.5 | 88.0 | 56.5 | 59.1 | 10.6 |
| v5.13 | coop_csp | 0.065 | 66.3 | 37.1 | 66.2 | 25.8 | 66.4 | 86.3 | 55.4 | 62.6 | 16.6 |
| v5.14 | coop_csp_primitive_branches | 0.045 | 68.3 | 35.0 | 66.5 | 23.8 | 66.5 | 89.9 | 57.7 | 63.1 | 13.2 |
| v7 | patch_disease | 0.0925 | 62.2 | 38.4 | 67.0 | 26.9 | 67.2 | 86.9 | 55.3 | 58.4 | 7.2 |
| v7.1 | zoom_crops | 0.06 | 66.4 | 37.7 | 67.1 | 26.2 | 67.2 | 87.2 | 55.9 | 55.2 | 10.9 |
| v7.2 | within_crop_negatives | 0.075 | 64.6 | 40.4 | 67.0 | 28.9 | 67.0 | 85.9 | 55.5 | 56.4 | 9.9 |
| v9 | posthoc_fusion_lora_scale | 0.0275 | 67.7 | 32.2 | 66.8 | 21.2 | 66.8 | 90.0 | 57.6 | 58.3 | 5.4 |
| v10 | image_consistency | 0.0675 | 64.5 | 41.2 | 67.0 | 29.8 | 67.1 | 87.0 | 56.4 | 57.4 | 10.9 |
| v10.1 | text_anchor | 0.065 | 65.4 | 35.9 | 66.5 | 24.6 | 66.6 | 86.1 | 54.9 | 58.3 | 7.5 |
| v10.2 | weight_ema | 0.085 | 64.6 | 39.2 | 68.2 | 27.5 | 68.2 | 85.1 | 55.9 | 57.2 | 9.9 |
| v10.3 | text_anchor_three_splits | 0.0725 | 66.3 | 33.5 | 67.4 | 22.3 | 67.5 | 86.8 | 56.2 | 61.2 | 15.3 |
| v12 | lora_all_blocks | 0.08 | 68.0 | 27.0 | 70.0 | 16.7 | 70.1 | 85.9 | 57.4 | 61.8 | 15.2 |
| v13 | ssl_local_global | 0.0725 | 66.1 | 41.2 | 69.2 | 29.3 | 69.3 | 89.1 | 59.5 | 56.5 | 8.2 |
| v13.1 | ssl_weight20 | 0.0625 | 68.3 | 28.9 | 68.4 | 18.3 | 68.5 | 87.8 | 57.9 | 55.6 | 8.2 |
| v11 | feature_gen | 0.03 | 57.8 | 24.2 | 63.7 | 14.9 | 64.0 | 71.3 | 42.6 | 49.2 | 15.1 |
| v11.1 | compositional_condition | 0.0375 | 59.7 | 6.2 | 62.4 | 3.2 | 62.4 | 75.0 | 44.2 | 50.7 | 5.0 |
| v11.2 | no_generation | 0.0975 | 41.9 | 0.0 | 60.2 | 0.0 | 60.2 | 47.0 | 24.9 | 37.0 | 0.0 |
| v11.3 | lora_features | 0.03 | 64.9 | 44.0 | 66.6 | 32.9 | 67.3 | 82.5 | 52.7 | 60.2 | 28.6 |
| v11.4 | lora_compositional | 0.0425 | 66.1 | 31.4 | 69.3 | 20.3 | 69.5 | 81.8 | 54.2 | 61.9 | 18.1 |
| v6 | cnn_ensemble | 6.6 | 70.1 | 19.8 | 71.2 | 11.5 | 71.4 | 89.2 | 61.0 | 53.1 | 2.6 |
| v6.1 | lora_large_ensemble | 6.3 | 70.8 | 25.7 | 71.2 | 15.7 | 71.3 | 88.8 | 61.2 | 58.9 | 4.8 |
| v6.2 | two_sentence_ensemble | 7.1 | 70.5 | 22.4 | 71.8 | 13.3 | 71.9 | 88.1 | 61.2 | 58.2 | 6.5 |
| v6.3 | convnext_v10.3_ensemble | 6.7 | 75.0 | 27.0 | 74.6 | 16.5 | 74.6 | 95.1 | 68.4 | 66.0 | 3.7 |
| v6.4 | convnext336_ensemble | 6.8 | 74.6 | 19.7 | 74.7 | 11.3 | 74.8 | 93.7 | 67.5 | 67.6 | 6.9 |
| v6.5 | convnext_v5.13_ensemble | 6.4 | 76.0 | 26.7 | 74.6 | 16.2 | 74.6 | 94.9 | 68.6 | 67.1 | 2.2 |
| v6.6 | convnext336_v5.13_ensemble | 6.7 | 75.5 | 20.2 | 75.0 | 11.7 | 75.0 | 93.8 | 68.1 | 68.2 | 5.8 |
| v6.7 | three_member_ensemble | 6.4 | 77.1 | 23.5 | 75.8 | 13.9 | 75.8 | 95.0 | 69.6 | 68.7 | 5.8 |
| v6.8 | convnext_v11.3_ensemble | 5.4 | 74.2 | 29.6 | 74.7 | 18.4 | 74.8 | 95.7 | 68.5 | 68.8 | 4.8 |
| v15 | gated_router | 0 | 75.0 | 75.0 | 64.2 | 90.1 | 74.6 | 95.1 | 67.0 | – | – |
| aniq_baseline_independent_cnn |  | 6.5 | 59.7 | 14.5 | 67.2 | 8.2 | 67.2 | 75.8 | 47.2 | 50.9 | 4.8 |
| aniq_baseline_multitask_cnn |  | 6.9 | 56.7 | 10.0 | 67.6 | 5.4 | 67.7 | 73.2 | 44.8 | 51.6 | 2.4 |
| aniq_baseline_independent_convnext_base |  | 6.4 | 68.6 | 24.7 | 74.2 | 14.8 | 74.2 | 91.2 | 62.6 | 60.1 | 10.9 |
| aniq_independent_convnext_base_336_1split |  | 6.6 | 69.3 | 17.4 | 74.4 | 9.9 | 74.5 | 91.9 | 64.0 | 63.6 | 6.5 |
| aniq_cropnorm_v0 |  | 7 | 56.6 | 14.4 | 68.2 | 8.1 | 68.2 | 74.9 | 45.9 | 50.9 | 4.3 |
| aniq_cropnorm_v1 |  | 7 | 54.2 | 5.6 | 65.5 | 2.9 | 65.5 | 73.3 | 42.5 | 53.2 | 0.9 |
| aniq_cropnorm_v2 |  | 6.7 | 55.5 | 0.5 | 64.8 | 0.3 | 64.8 | 76.8 | 44.5 | 53.6 | 0.9 |
| aniq_cropnorm_v3 |  | 7.2 | 53.9 | 1.8 | 66.5 | 0.9 | 66.5 | 73.1 | 42.8 | 51.7 | 0.6 |
| aniq_cropnorm_v4 |  | 5.9 | 56.0 | 18.2 | 61.7 | 10.7 | 61.9 | 76.7 | 42.8 | 52.1 | 7.0 |

Zero-shot runs (v1–v3.9) never used γ, so both HM columns are the same. v15 (the gated router) routes with a learned gate, so γ is 0 by design. "Val HM (γ=0)" is at the epoch selected with γ.

## Gamma-free validation metrics (seed-42 split, 862 val-seen / 533 pseudo-unseen images)

These come from the validation scores that the ensembles cached for their members. They are the only gamma-free numbers that may be used for decisions. The members were retrained on the split by `train/ensemble.py`, so their best-γ HM differs by about 1 pt from each run's own recorded validation HM (v5.5: 55.6 here vs 56.9 recorded).

| Member | Val HM (best γ) | Val HM (γ=0) | Closed seen | Closed unseen | AUSUC | Bias gap (log-prob) |
|---|---|---|---|---|---|---|
| aniq_baseline_independent_cnn | 44.5 | 2.2 | 86.2 | 45.4 | 32.6 | 5.06 |
| v5 | 48.4 | 10.9 | 84.2 | 48.0 | 34.6 | 4.20 |
| v5.5 | 55.6 | 10.9 | 83.3 | 57.0 | 42.6 | 3.29 |
| v5.7 | 55.4 | 8.6 | 85.4 | 56.7 | 42.4 | 3.87 |
| aniq_baseline_independent_convnext_base | 57.7 | 2.9 | 90.3 | 56.7 | 46.3 | 5.27 |
| v10.3 | 57.1 | 9.6 | 85.2 | 58.2 | 44.2 | 3.95 |
| aniq_independent_convnext_base_336_1split | 63.6 | 6.5 | 90.8 | 64.0 | 53.2 | 4.94 |
| v5.13 | 58.4 | 6.1 | 83.8 | 59.1 | 44.8 | 3.36 |
| v11.3 | 58.5 | 22.7 | 84.3 | 59.5 | 45.2 | 1.44 |

## Which changes are real: paired bootstrap on test (added 2026-11-23)

`scratch/gamma0_bootstrap.py` resamples the seen-test and unseen-test images 1,000 times, using the same draw for both runs of a step. Full output: `docs/vlm/gamma0_audit_bootstrap.csv`, which also has the zero-shot, ensemble and CNN steps not repeated here. The bootstrap only covers test-sampling noise. Retraining noise has to be added on top, and it can be measured from two same-recipe pairs:

| Same recipe, retrained | Δ AUSUC | Δ closed seen | Δ closed unseen |
|---|---|---|---|
| v5.5 → v5.8 | +0.1 | +0.8 | −0.9 |
| v10.1 → v10.3 | +1.3 | +0.9 | +0.7 |

So a trained-model change counts as **real** only if its 95% interval excludes 0 **and** it moves by at least 2 pts. Zero-shot runs (v1–v3.9) have no training noise, so for them the interval alone decides.

**The table's components (clean names, symptom descriptions, CSP, CoOp, LoRA), gamma-free:**

| Component | Step | Δ AUSUC | Δ closed seen | Δ closed unseen | Verdict |
|---|---|---|---|---|---|
| Clean class names | v1 → v2 | +0.6 | +0.9 | **+4.1** | Real, small: unseen pairs only |
| Symptom description | v2 → v3 | +0.8 | +0.7 | +1.2 | Weak. Its +4.5 HM is mostly more unseen routing |
| CSP prompt learning | v3 → v4 | **+35.0** | **+43.7** | **+32.2** | Real, the largest gain |
| LoRA image encoder | v4.2 → v5 | **+16.9** | **+10.8** | **+11.1** | Real |
| LoRA larger (8 blocks, rank 16) | v5 → v5.5 | +0.6 | +0.4 | −0.6 | No γ-free gain on test (validation γ-free +8 AUSUC) |
| CoOp context | v5.8 → v5.13 | −0.7 | **−2.0** | +1.1 | Not real on test γ-free. The calibrated +2.1 is a better operating point, not better recognition (validation γ-free +2.2 AUSUC vs v5.5) |

**Other changes on the LoRA line that are real (γ-free):**

| Experiment | Step | Δ AUSUC | Δ closed seen | Δ closed unseen | Notes |
|---|---|---|---|---|---|
| SSL local-global consistency (v13) | v5.5 → v13 | **+3.5** | +1.7 | **+3.0** | Largest gain, and the only one that helps both sides. Rejected on 1-split validation (−0.4). Weight 20 (v13.1) lost it again (−1.6). |
| Primitive branches (Troika) | v5.8 → v5.11 | +1.8 | **−2.3** | **+5.7** | Trades seen for unseen |
| Primitive branches on CoOp | v5.13 → v5.14 | **+2.3** | +0.1 | **+3.7** | Same mechanism, replicated, with no seen cost here |
| Post-hoc primitive fusion (v9) | v5.5 → v9 | +1.7 | −0.8 | **+3.9** | Same mechanism as primitive branches, applied without retraining |
| DRPT alternating (v5.9) | v5.8 → v5.9 | +1.1 | −1.2 | **+3.5** | Unseen only |
| CoOp without CSP (v5.12) | v5.8 → v5.12 | +0.4 | −1.8 | **+2.8** | Unseen gain, seen cost |
| LoRA all 12 blocks (v12) | v10.3 → v12 | +1.2 | **+2.5** | −0.9 | Seen only |

**Not real on test γ-free** (|Δ| < 2 pts on every γ-free metric, or the interval includes 0):

- **Trained-model changes:** 336 px, TTA, class-balanced sampling, image consistency (v10), text anchor (v10.1/v10.3), EMA, patch disease (v7), zoom crops, within-crop negatives, multi-description (v5.10), two-sentence prompts (v5.7) and the gated router (v15, −1.5 AUSUC).
- **BioCLIP** is significantly worse, both zero-shot (v3.9) and trained (v4.3).
- **The v11 feature-generation branch** is a real gain over its own frozen-CLIP baseline (generation adds +24 closed unseen). On the same LoRA features it is worse than the prompt model it replaces (v10.3 → v11.3: AUSUC −3.5, closed unseen −4.3).

**Zero-shot text findings** that did not carry over to the trained model:

- Taxonomy text adds +8.2 closed unseen (v3.3) and +7.0 (v3.4). Separate taxonomy and symptom prompts add +3.2 (v3.6).
- Once trained, the same text gives nothing (v5.7: AUSUC −0.1).

**Ensembles** give the largest single gain over the VLM line: v5.13 → v6.5 (+ConvNeXt) is +13.2 AUSUC, +8.2 closed seen and +8.6 closed unseen. Adding a third member (v6.6 → v6.7) gives +1.5 AUSUC.

All of this is test evidence. It explains the results; it does not license picking models on test. The two changes worth re-deciding properly are v13 (SSL) and primitive branches combined with it, re-run on the 3-split validation (v5.8 recipe).

### The v11 feature-generation branch against the prompt model on the same image features (paired bootstrap, test)

| Step | Δ AUSUC | Δ closed seen | Δ closed unseen | Δ calibrated HM | Δ validation HM (3 splits) |
|---|---|---|---|---|---|
| Frozen CLIP: v4.2 (CSP) → v11 (GAN) | **+4.2** | **+7.7** | **−4.4** | +6.0 | +2.7 |
| Frozen CLIP: v4.2 → v11.1 (compositional condition) | **+5.8** | **+6.0** | −0.6 | +7.9 | +4.2 |
| LoRA features: v10.3 (CSP + LoRA) → v11.3 | **−3.5** | −0.2 | **−4.3** | −1.4 | −0.9 |
| LoRA features: v10.3 → v11.4 | **−2.1** | +1.9 | **−5.0** | −0.2 | +0.7 |
| CoOp + CSP LoRA features: v5.13 → v11.5 | −0.4 | **+1.9** | **−2.3** | +0.4 | −3.2 |
| Feature source swap: v11.3 → v11.5 | **+2.2** | +1.0 | +1.5 | +1.8 | −0.8 |
| Generation off: v11 → v11.2 | **−17.7** | −3.8 | **−24.2** | −15.8 | −12.2 |

Reading:

- **On frozen CLIP**, the GAN classifier beats CSP, because a classifier trained per seen pair recognises seen pairs better.
- **On LoRA features**, where the LoRA and prompts already fit the seen pairs, that advantage is gone. The synthetic unseen features are then worse than CSP's learned text compositions.
- **The GAN classifiers have a much smaller seen bias** (median gap 0.009 for v11.3 vs 0.023 for v10.3). That is why v11.3 has the best γ=0 HM of any single model (44.0). This is a bias effect, not better recognition.
- **v11.5** (the GAN on v5.13's CoOp + CSP features, added 2026-11-23) confirms this. It is level with v5.13 on AUSUC, trades unseen for seen, and is 3.2 pts below it on validation (59.43 vs 62.59). Against v11.3 its +2.2 test AUSUC is just past noise, but its validation HM is 0.8 lower, so it is not kept.
- **In an ensemble** it adds nothing over v10.3 (v6.3 → v6.8: AUSUC +0.1, calibrated −0.8).
