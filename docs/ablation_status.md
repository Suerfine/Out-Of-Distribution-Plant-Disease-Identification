# Ablation status for the final combined model (no calibration)

Report scope: Independent CNN, MTL CNN, CLIP, LoRA, CSP, CoOp, GAN, GATE. Calibration (gamma / routing offset) is **excluded, final decision**.
All numbers below are official-test, gamma = 0, top-1 %. Sources: `docs/all_results.csv`, `Disease_Transfer_Experiments/clip_anchored_gan_v1/README.md`, `config/vlm/*_config.json` `_result`.

## 1. Evidence at gamma = 0 today

| Component | Comparison | Seen | Unseen | HM | Status |
|---|---|---:|---:|---:|---|
| Independent CNN (ConvNeXt-B) | baseline | 73.8 | 14.0 | 23.6 | have |
| MTL CNN (ConvNeXt-B v3) | baseline | - | - | 11.7 | have (many variants) |
| CLIP zero-shot | best prompt (v3.6) | - | - | 11.9 | have |
| + LoRA | v5 / v5.3 large | 67.0 / 67.6 | 16.2 / 28.7 | 26.1 / 40.3 | have |
| + CoOp (v5.12) | fixed names | - | - | - | only calibrated test (HM 66.95) + val; **no gamma=0 row** |
| + CSP + CoOp (v5.13) | learned context + CSP | - | - | - | only calibrated test (HM 66.29) + val; **no gamma=0 row** |
| + GAN (no anchor) | original heads | 66.0 | 35.6 | 46.3 | have |
| + GAN crop-anchored | anchored heads | 66.1 | 39.0 | 49.0 | have |
| + ConvNeXt crop scores | pair-based CLIP+GAN | 68.0 | 40.7 | 50.9 | have (README) |
| + disease-only CLIP head | two-expert baseline 52.3 -> | 72.9 | 48.3 | 58.1 | have |
| + GATE (hard routing) | at gamma=0 | 72.9 | 48.3 | 58.1 | have, but the routing offset is the gamma-style calibration; see section 3 |
| Gate-dependent blend | | 72.3 | 47.6 | 57.4 | have, worse than hard routing at gamma=0 |
| Fixed blend | | 69.7 | 43.7 | 53.7 | have, worse than hard routing at gamma=0 |

## 2. Gaps to close for a final-model ablation

1. **CoOp and CSP have no gamma=0 test row.** v5.12 (CoOp only) and v5.13 (CoOp + CSP) were run and judged on validation HM with calibration. Re-evaluate their saved `scores.npz` at gamma=0 and log them. If the scores were not saved, re-run.
2. **CoOp / CSP are not yet in the final pipeline.** The final pipeline (disease-only CLIP LoRA + GAN + ConvNeXt + gate) does not use CoOp or CSP text prompts. If the final model is the combination of all, the combined row (LoRA + CoOp + CSP + GAN + GATE) does not exist yet. This is the main missing experiment.
3. **No single leave-one-out chain.** The README rows use different bases. Produce one table from the full final model, removing one component at a time: GAN, crop anchoring, ConvNeXt handoff, disease-only head, CoOp, CSP, LoRA, GATE.
4. **Gate with no offset.** At gamma=0 the gate is tuned only by its form (hard vs blend). Decide what the gate is without an offset and ablate that form only. Drop the "routing offset" ablation from the report.
5. **Seeds.** Decision: single seed (42) for everything, including the final model and all ablations; no time for 3 seeds before the 27/11 deadline. State this as a limitation in the report (10 unseen pairs, one official test set, differences under about 1 HM are noise).
6. **Baselines.** MTL CNN: v4_1_cutmix, v4_2_balsoftmax, v4_3_droppath have configs with no logged results that I found.

## 3. Constraint to resolve

The gate's seen/unseen decision uses a threshold, which is a form of the offset we excluded. Without it, unseen accuracy is 48.3% (vs 82.8% at offset 3.0). Confirm that the report's gate is allowed to learn or fix its threshold without a gamma search, or that we report 58.1 HM as the headline.
