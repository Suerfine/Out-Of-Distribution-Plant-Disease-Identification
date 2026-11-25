# VLM results as a baseline ladder (no retraining)

Written 2026-11-23 from existing runs only. Sources: `docs/vlm/gamma0_audit.csv` (validation and test numbers per run), `docs/vlm/gamma0_audit_bootstrap.csv` (paired test bootstrap), `docs/vlm/gamma0_audit_val_members.csv` (ensemble members on validation split 42). The controlled re-run design in `config/vlm2/` is shelved. Diagram: `docs/vlm/vlm_baseline_ladder.drawio` (page 1 the ladder, page 2 the interactions).

## Rule

1. Each target model is compared with the **current baseline**.
2. Promotion needs **validation HM (calibrated, the protocol metric) above the baseline's by more than 2 points**. If so, the target becomes the new baseline.
3. If several targets beat the same baseline, the one with the highest validation HM becomes the baseline. The others are candidates for combination (see the interaction section).
4. Test numbers never decide. They are reported next to each step, gamma-free (AUSUC, closed-set seen/unseen Top-1), to explain what the change did. Significance (sig+ / sig− / n.s.) comes from the paired bootstrap on test images, 95% interval.

Validation designs differ across the history, so each comparison uses the same images on both sides:

| Run family | Validation |
|---|---|
| v4.2, v4.3, v5.8+, v10.3, v11.x, v12 | mean of 3 splits (seeds 42, 43, 44) |
| v5–v5.7, v7.x, v9, v10–v10.2, v13.x | 1 split (seed 42) |
| Ensembles v6.x, router v15 | split 42, members rescored on it |

A 1-split run is compared with a 3-split baseline's split-42 value. Noise is about 3 points between same-recipe reruns on 1 split (v5: 48.3 vs 51.2), so the 2-point threshold is lenient for 1-split runs.

## The ladder

**Stage 0: zero-shot CLIP (v1 → v2 → v3).** No training, so there is no validation number, and nothing is promoted on test. v3 (clean names + symptom descriptions) is the base that prompt learning was built on. For reference, on test: HM 5.4 → 5.6 → 10.2; AUSUC 2.0 → 2.6 → 3.4, both steps sig+.

**Stage 1: prompt learning (CSP).** v4.2 is the first run with a saved validation number: **46.57** (3 splits). v4 and v4.1 were protocol fixes (gamma grid, split count) without stored validation. Test, gamma-free, v3 → v4: AUSUC +35.0, sig+.

| Target (change) | vs | Val HM | Base | Δ | Verdict | Test γ-free Δ (AUSUC; closed unseen) |
|---|---|---|---|---|---|---|
| v4.3 BioCLIP backbone | v4.2 | 48.06 | 46.57 | +1.49 | ✗ | −6.8 sig−; −16.0 sig− |
| **v5 LoRA (last 4 blocks)** | v4.2 split 42 | 51.22 | 44.53 | **+6.69** | **✓ new baseline** | +16.9 sig+; +11.1 sig+ |

**Stage 2: on v5 (1 split, 51.22).** These are the single-change ablations v5.1–v5.4.

| Target (change) | Val HM | Δ | Verdict | Test γ-free Δ (AUSUC) |
|---|---|---|---|---|
| v5.1 336 px | 51.03 | −0.19 | ✗ | −1.6 n.s. |
| v5.2 flip TTA | 51.64 | +0.42 | ✗ | +0.6 n.s. |
| **v5.3 larger LoRA (8 blocks, rank 16)** | 56.89 | **+5.67** | **✓ new baseline (= v5.5)** | +0.6 n.s. |
| v5.4 class-balanced sampling | 49.03 | −2.19 | ✗ | −1.6 sig− |

**Stage 3: on v5.5 (1 split, 56.89).**

| Target (change) | Val HM | Δ | Verdict | Test γ-free Δ (AUSUC; closed unseen) |
|---|---|---|---|---|
| v5.7 two-sentence prompts | 55.60 | −1.29 | ✗ | −0.1 n.s. |
| v7 patch-level disease scores | 58.39 | +1.50 | ✗ | −0.7 n.s. |
| v7.1 zoom crops | 55.25 | −1.64 | ✗ | 0.0 n.s. |
| v7.2 within-crop negatives | 56.36 | −0.53 | ✗ | −0.4 n.s. |
| v9 post-hoc primitive fusion | 58.33 | +1.44 | ✗ | **+1.7 sig+; +3.9 sig+** |
| v10 image consistency | 57.43 | +0.54 | ✗ | +0.4 n.s. |
| v10.1 text anchor | 58.30 | +1.41 | ✗ | −1.0 sig− |
| v10.2 weight EMA | 57.25 | +0.36 | ✗ | −0.1 n.s. |
| v13 SSL local-global | 56.53 | −0.36 | ✗ | **+3.5 sig+; +3.0 sig+** |
| v13.1 SSL weight 20 (vs v13) | 55.65 | −0.88 | ✗ | −1.6 sig− |

v5.8 is v5.5's recipe re-measured on 3 splits: **59.75**. Its split-42 value is 56.48, consistent with v5.5's 56.89, and test AUSUC moves by only +0.1. It is not a change. It becomes the reference for the 3-split runs that follow.

**Stage 4: on v5.8 (3 splits, 59.75).**

| Target (change) | Val HM | Δ | Verdict | Test γ-free Δ (AUSUC; closed unseen) |
|---|---|---|---|---|
| v5.9 DRPT alternating primitives | 60.57 | +0.82 | ✗ | +1.2 n.s.; +3.5 sig+ |
| v5.10 multiple descriptions | 60.69 | +0.94 | ✗ | +1.0 n.s. |
| **v5.11 primitive branches** | 62.23 | **+2.48** | ✓ passes | **+1.8 sig+; +5.7 sig+** |
| v5.12 CoOp context, no CSP | 59.13 | −0.62 | ✗ | +0.4 n.s. |
| **v5.13 CoOp context + CSP** | 62.59 | **+2.84** | **✓ highest → new baseline** | −0.7 n.s.; +1.1 n.s. |
| v10.3 text anchor | 61.15 | +1.40 | ✗ | +0.2 n.s. |
| v12 LoRA on all 12 blocks (on v10.3: 61.15) | 61.80 | +0.65 | ✗ | +1.2 n.s. |
| v11.3 feature-GAN on v10.3 features (vs v10.3) | 60.24 | −0.91 | ✗ | −3.5 sig− |

**Stage 5: on v5.13 (3 splits, 62.59).**

| Target (change) | Val HM | Δ | Verdict | Test γ-free Δ (AUSUC; closed unseen) |
|---|---|---|---|---|
| v5.14 + primitive branches | 63.07 | +0.48 | ✗ | **+2.3 sig+; +3.7 sig+** |
| v11.5 feature-GAN on v5.13 features | 59.43 | −3.16 | ✗ | −0.4 n.s. |

**Best single VLM under the rule: v5.13.** Validation HM 62.59. Test: HM 66.29, AUSUC 55.35.

**Stage 6: ensemble with a CNN (split 42; v5.13 as a member scores 58.42).**

| Target (change) | Val HM | Base | Δ | Verdict | Test γ-free Δ (AUSUC) |
|---|---|---|---|---|---|
| **v6.5 v5.13 + ConvNeXt-B** | 67.12 | 58.42 | **+8.70** | **✓ new baseline** | +13.2 sig+ |
| v6.6 v5.13 + ConvNeXt-B 336 px (vs v6.5) | 68.25 | 67.12 | +1.13 | ✗ | −0.5 |
| v6.7 v5.13 + both ConvNeXts (vs v6.5) | 68.66 | 67.12 | +1.54 | ✗ | +1.0 |
| v6.8 v11.3 + ConvNeXt-B (vs v6.5) | 68.81 | 67.12 | +1.69 | ✗ | −0.1 |
| v15 gated seen/unseen router (on v6.3: 66.02) | 65.61 | 66.02 | −0.41 | ✗ | −1.5 sig− |

**Final model under the rule: v6.5.** Validation HM 67.12. Test: HM 76.03, AUSUC 68.58. v6.7 has the highest test HM (77.07), but it did not clear +2 on validation, so it is not selected.

Promotion path: **v3 → v4.2 (CSP) → v5 (LoRA) → v5.5 (larger LoRA) → [v5.8 re-measure] → v5.13 (CoOp + CSP) → v6.5 (+ ConvNeXt ensemble)**.

### Where the HM rule and the gamma-free evidence disagree

Calibrated validation HM mixes recognition with seen/unseen bias (`docs/vlm/gamma0_audit.md`). Five steps should be reported with that caveat.

| Run | HM rule | Gamma-free test | Reading |
|---|---|---|---|
| v5.3 larger LoRA | ✓ +5.7 | AUSUC +0.6 n.s.; HM at γ=0 +14.2 | The gain is mostly a smaller seen bias, not better recognition |
| v5.13 CoOp + CSP | ✓ +2.8 | AUSUC −0.7 n.s. | The validation gain did not carry over to test recognition |
| v13 SSL | ✗ −0.4 | AUSUC +3.5 sig+ | The largest real recognition gain, rejected by HM |
| v9 post-hoc fusion | ✗ +1.4 | AUSUC +1.7 sig+; unseen +3.9 | Real, unseen-side only |
| v5.14 + primitive branches | ✗ +0.5 over v5.13 | AUSUC +2.3 sig+; unseen +3.7 | Real over v5.13, below the HM threshold |

## Complementary and overlapping changes

For changes A and B measured from a common base: interaction **I = Δ(A+B) − ΔA − ΔB**.

| Label | Condition |
|---|---|
| Complementary | Δ(A+B) ≈ ΔA + ΔB (I near 0) |
| Overlapping | Δ(A+B) ≈ max(ΔA, ΔB) |
| Antagonistic | Δ(A+B) < max(ΔA, ΔB) |

Only pairs whose A, B and A+B runs already exist can be classified. Validation HM uses the same images throughout each row. Test AUSUC is shown as the gamma-free check.

| Pair (base) | ΔA | ΔB | Δ(A+B) | I | Class (val HM) | Test AUSUC: ΔA / ΔB / Δ(A+B) | Class (γ-free) |
|---|---|---|---|---|---|---|---|
| Primitive branches (v5.11) × CoOp context (v5.13), base v5.8; A+B = v5.14 | +2.48 | +2.84 | +3.32 | −2.00 | **Overlapping** | +1.8 / −0.7 / +1.6 | CoOp adds nothing; the gain is the branches |
| LoRA path (v10.3) × feature-GAN (v11), base v4.2 frozen CLIP; A+B = v11.3 | +14.58 | +2.68 | +13.67 | −3.59 | **Antagonistic** | +17.8 / +4.2 / +14.3 | Antagonistic (I = −7.7) |
| Feature-GAN × CoOp + CSP features (v5.13 → v11.5) | — | — | GAN −3.16 on v5.13 (−0.91 on v10.3) | — | **Antagonistic** (the GAN hurts on every LoRA base) | −0.4 / −3.5 | Same |
| VLM (v5.13, 58.42) × ConvNeXt-B (57.67), v6.5 | — | — | 67.12, +8.70 over the best member | — | **Complementary** | members 55.4 / 62.6, ensemble 68.6 | Complementary |
| VLM improvement × ensemble: v10.3 → v5.13 as a member (+1.27) vs v6.3 → v6.5 (+1.10) | — | — | gain kept | ≈ 0 | **Complementary** | — | — |
| VLM improvement × ensemble: v10.3 → v11.3 as a member (+1.36) vs v6.3 → v6.8 (+2.79) | — | — | gain kept | +1.4 (within noise) | Complementary | — | — |
| ConvNeXt 336 px × ensemble: member +5.91 (57.67 → 63.58); in the ensemble with v5.13 +1.13, with v10.3 +1.62 | +5.91 | — | +1.13 | −4.8 | **Overlapping** | member +1.5, ensemble −0.5 | Overlapping |
| Second ConvNeXt in the ensemble (v6.7 vs v6.6) | — | — | +0.41 | — | **Overlapping** (same model family) | +1.5 | — |

One more pair has no "neither" run, so I cannot be computed: **CSP × CoOp on LoRA**. CSP only (v5.8) scores 59.75, CoOp only (v5.12) 59.13, and both (v5.13) 62.59. That is +2.84 over the better single, which looks additive on validation. On test AUSUC the three are 56.0, 56.5 and 55.4: substitutes. Both act on the text-side class representation, so **overlapping** is the better-supported reading.

### Summary

- **Complementary (combine):**
  - the VLM and the CNN (different models, different errors);
  - VLM improvements survive ensembling, so a better VLM still gives a better ensemble.
- **Overlapping (keep the best one):**
  - text-side changes among themselves: CSP, CoOp, and the branches/CoOp pair on HM;
  - CNN variants inside the ensemble.
- **Antagonistic (do not combine):** the feature-GAN classifier with LoRA. LoRA already supplies the adapted image features the GAN was compensating for, and the GAN replaces the prompt scorer that CSP/CoOp train.

## Not classifiable from existing runs

These have only single-change runs, so their combinations are untested:
- SSL (v13);
- post-hoc fusion (v9);
- DRPT (v5.9);
- text anchor (v10.1 / v10.3).

The pair most worth one extra run is **SSL × primitive branches**, because:
- both are real gamma-free gains on different levers (image vs composition);
- the hypothesis is complementary.

## Uncalibrated ladder (γ = 0 everywhere)

Diagram: `docs/vlm/vlm_tree_uncalibrated.drawio`. It shows the full lineage tree of every VLM config, with each run's verdict.

**Rule.** It is the same as above, except every validation HM is taken at γ = 0. The score is the saved model (epoch as trained) with no seen-pair penalty. Runs are checked in version order against the current baseline, on the same validation images.

**Exclusions.**
- Zero-shot runs (v1–v3.9), v4, v4.1 and v5.6 have no validation number, so they cannot be promoted.
- The v15 gated router is excluded: a learned seen/unseen gate is calibration by another means.

**Promotion path: v4.2 (15.30) → v11.3 (28.60, +13.30).** Nothing else clears +2.

| Run | Val HM γ=0 | Δ vs baseline | Verdict | Test HM γ=0 | Test AUSUC |
|---|---|---|---|---|---|
| v4.2 CSP, 3 splits | 15.30 | first validated | baseline | 26.0 | 38.4 |
| v5 LoRA | 10.88 | −1.52 (split 42) | ✗ | 26.1 | 55.3 |
| v5.5 larger LoRA | 9.22 | −3.18 (split 42) | ✗ | 40.3 | 56.0 |
| v5.8 | 14.81 | −0.49 | ✗ | 39.8 | 56.0 |
| v5.10 multi-description | 16.31 | +1.01 | ✗ | 39.5 | 57.0 |
| v5.13 CoOp + CSP | 16.58 | +1.28 | ✗ | 37.1 | 55.4 |
| v10.3 text anchor | 15.28 | −0.02 | ✗ | 33.5 | 56.3 |
| v11 feature-GAN, frozen CLIP | 15.13 | −0.17 | ✗ | 24.2 | 42.6 |
| **v11.3 feature-GAN on v10.3's LoRA features** | **28.60** | **+13.30** | **✓ final** | 44.0 | 52.7 |
| v11.4 compositional GAN on LoRA | 18.07 | −10.53 vs v11.3 | ✗ | 31.4 | 54.2 |
| v11.5 GAN on v5.13 features | 26.71 | −1.89 vs v11.3 | ✗ | 37.7 | 54.9 |
| v13 SSL | 8.21 | −14.49 vs v11.3 (split 42) | ✗ | 41.2 | 59.5 |
| v6.x ensembles with a CNN | 2.2–6.9 | −5.5 to −10.2 (split 42) | ✗ | 19.7–29.6 | 60.9–69.6 |

**Reading.**
- Without calibration, HM rewards a model for having little seen bias, not for recognising better. The audit established this (`gamma0_audit.md`).
- The winner, v11.3, has the highest γ = 0 HM on both validation and test. But its test AUSUC is 3.5 points *below* the LoRA run whose features it uses (v10.3, sig−).
- LoRA itself (v5) is rejected, although it is the second-largest recognition gain in the history (+16.9 AUSUC). It made the model more confident on seen pairs.
- The CNN ensembles score lowest of all at γ = 0, despite the best AUSUC (68–70). The CNN member is heavily seen-biased.
- So the uncalibrated ladder picks the least-biased model, not the best recogniser. The calibrated ladder (v6.5) and test AUSUC (v6.7 / v6.5 for ensembles, v13 for a single VLM) answer the recognition question.
