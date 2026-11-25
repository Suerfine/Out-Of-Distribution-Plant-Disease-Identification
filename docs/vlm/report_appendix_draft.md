# Report appendices (draft): unreported configurations and negative results

Draft generated on 2026-11-25 from `config/vlm/*_config.json` (each config's `_note` and `_result`), the run folders in
`runs/vlm/`, and `docs/all_results.csv`. Every number is on the official test set or the validation split stated.
Letters A–I refer to the sections of the main results.

## Appendix D: Every VLM configuration

All configurations in `config/vlm/`, grouped by branch.
- **Val HM:** mean over the stated number of validation splits; "(1)" is the single seed-42 split, "(3)" the mean of seeds 42–44.
- **Δ val:** against the parent, only when both use the same split design.
- **Test HM, calibrated:** with the γ chosen on validation. **Test HM, γ = 0:** without calibration.
- **AUSUC:** area under the seen–unseen accuracy curve, which does not depend on γ.
- Zero-shot runs have no validation and no γ.
- **"Kept":** the change passed the rule (> 2 points over its parent on validation). **"Reference":** the same model re-validated on three splits. **"Probe":** scored without retraining.

Notes on individual rows:
- v5.5 is a copy of v5.3: the combination step kept only v5.3's change.
- v8's run folder was later overwritten; its numbers come from its config's result note.
- v14 and v14.1 (SigLIP) were run by a teammate on Colab; their numbers come from `docs/all_results.csv`.
- v15 is the gated router (renumbered from v14). It uses a learned gate instead of γ, so its γ = 0 value is its result, and AUSUC is not defined for it.

### D.1 Zero-shot CLIP (no training)

| Version | Change from parent | Parent | Val HM (splits) | Δ val vs parent | Test HM, calibrated | Test HM, γ = 0 | Test AUSUC | Decision |
|---|---|---|---|---|---|---|---|---|
| v1 | Zero-shot CLIP ViT-B/16, one name-only prompt per pair | - | – | – | – | 5.4 | 2.0 | Baseline (A) |
| v2 | Cleaned class names + 5 prompt templates | v1 | – | – | – | 5.6 | 2.6 | Kept as preprocessing |
| v3 | Visual symptom description appended to each prompt | v2 | – | – | – | 10.2 | 3.4 | Kept (B) |
| v3.1 | Random letter strings instead of descriptions (control) | v3 | – | – | – | 5.5 | 1.5 | Control |
| v3.2 | Descriptors scored separately and averaged | v3 | – | – | – | 3.8 | 2.8 | Not kept |
| v3.3 | Official taxonomic descriptions instead of visual ones | v3 | – | – | – | 8.0 | 4.7 | Not kept |
| v3.4 | Taxonomy + symptoms in one sentence | v3 | – | – | – | 8.4 | 4.3 | Not kept |
| v3.5 | Symptoms first, then taxonomy | v3.4 | – | – | – | 8.5 | 4.1 | Not kept |
| v3.6 | Taxonomy and symptoms as two separate prompts, averaged | v3.4 | – | – | – | 11.9 | 5.3 | Best zero-shot |
| v3.7 | Longer visual descriptions (21 vs 8 words) | v3.6 | – | – | – | 10.1 | 6.0 | Not kept |
| v3.8 | Longer visual descriptions alone | v3 | – | – | – | 7.9 | 3.6 | Not kept |
| v3.9 | BioCLIP backbone instead of OpenAI CLIP | v3 | – | – | – | 3.3 | 0.3 | Not kept |
| v14 | SigLIP backbone, zero-shot | v3 | – | – | – | 14.3 | – | Not compared |

### D.2 Prompt learning, frozen encoders

| Version | Change from parent | Parent | Val HM (splits) | Δ val vs parent | Test HM, calibrated | Test HM, γ = 0 | Test AUSUC | Decision |
|---|---|---|---|---|---|---|---|---|
| v4 | CSP: learned crop and disease vectors (encoders frozen) | v3 | 35.8 (1) | – | 43.6 | 34.9 | 38.4 | Kept |
| v4.1 | Fixed validation split (hide real unseen), lower lr | v4 | 45.8 (1) | +10.0 | 52.9 | 26.1 | 38.9 | Kept |
| v4.2 | Validation on 3 splits | v4.1 | 46.6 (3) | – | 51.8 | 26.0 | 38.4 | Kept (C) |
| v4.3 | BioCLIP backbone | v4.2 | 48.1 (3) | +1.5 | 48.6 | 16.8 | 31.6 | Not kept |

### D.3 LoRA on the image encoder

| Version | Change from parent | Parent | Val HM (splits) | Δ val vs parent | Test HM, calibrated | Test HM, γ = 0 | Test AUSUC | Decision |
|---|---|---|---|---|---|---|---|---|
| v5 | LoRA on image encoder (rank 8, last 4 blocks) | v4.2 | 51.2 (1) | – | 65.1 | 26.1 | 55.3 | Kept |
| v5.1 | 336 px input | v5 | 51.0 (1) | -0.2 | 63.7 | 27.9 | 53.8 | Not kept |
| v5.2 | Test-time augmentation (4 flips) | v5 | 51.6 (1) | +0.4 | 63.9 | 28.3 | 55.9 | Not kept |
| v5.3 | Larger LoRA (rank 16, last 8 blocks) | v5 | 56.9 (1) | +5.7 | 64.5 | 40.3 | 55.9 | Kept |
| v5.4 | Class-balanced sampling | v5 | 49.0 (1) | -2.2 | 62.3 | 32.9 | 53.7 | Not kept |
| v5.5 | Combination of kept ablations (= v5.3) | v5 | 56.9 (1) | +5.7 | 64.5 | 40.3 | 55.9 | Kept (D) |
| v5.6 | Swap in v3.6 two-sentence prompts, no retraining | v5.5 | – | – | 61.5 | 32.2 | 54.7 | Probe |
| v5.7 | Two-sentence prompts (taxonomy + symptoms), trained | v5.5 | 55.6 (1) | -1.3 | 66.5 | 32.8 | 55.8 | Not kept |
| v5.8 | Same model, validated on 3 splits | v5.5 | 59.8 (3) | – | 64.2 | 39.8 | 56.0 | Reference |
| v5.9 | DRPT alternating crop/disease updates | v5.8 | 60.6 (3) | +0.8 | 67.3 | 32.5 | 57.2 | Not kept |
| v5.10 | Several descriptions per disease (PLID) | v5.8 | 60.7 (3) | +0.9 | 66.9 | 39.5 | 57.0 | Not kept |
| v5.11 | Separate crop/disease visual branches + fusion (Troika) | v5.8 | 62.2 (3) | +2.5 | 68.2 | 35.0 | 57.8 | Kept by rule |
| v5.12 | CoOp (learned context, fixed names) instead of CSP | v5.8 | 59.1 (3) | -0.6 | 67.0 | 28.6 | 56.5 | Not kept |
| v5.13 | CoOp + CSP (learned context + learned names) | v5.8 | 62.6 (3) | +2.8 | 66.3 | 37.1 | 55.4 | Kept |
| v5.14 | + primitive branches (v5.11) | v5.13 | 63.1 (3) | +0.5 | 68.3 | 35.0 | 57.7 | Not kept |
| v7 | Patch-level disease scoring | v5.5 | 58.4 (1) | +1.5 | 62.2 | 38.4 | 55.3 | Not kept |
| v7.1 | Zoomed crops (scale 0.25-1) | v5.5 | 55.2 (1) | -1.6 | 66.4 | 37.7 | 55.9 | Not kept |
| v7.2 | Within-crop hard negatives | v5.5 | 56.4 (1) | -0.5 | 64.6 | 40.4 | 55.5 | Not kept |
| v8 | Rewritten descriptions for near-duplicate diseases | v5.5 | 56.8 (1) | -0.1 | 66.4 | – | – | Not kept |
| v9 | Post-hoc crop/disease fusion + LoRA scale | v5.5 | 58.3 (1) | +1.4 | 67.7 | 32.2 | 57.6 | Not kept |
| v10 | Image consistency with frozen CLIP | v5.5 | 57.4 (1) | +0.5 | 64.5 | 41.2 | 56.4 | Not kept |
| v10.1 | Text anchor (KgCoOp / PromptSRC) | v5.5 | 58.3 (1) | +1.4 | 65.4 | 35.9 | 54.9 | Not kept |
| v10.2 | EMA of prompt and LoRA weights | v5.5 | 57.2 (1) | +0.4 | 64.6 | 39.2 | 55.9 | Not kept |
| v10.3 | Same model, validated on 3 splits | v10.1 | 61.2 (3) | – | 66.3 | 33.5 | 56.2 | Reference |
| v10.4 | Text anchor on CoOp + CSP | v5.13 | – | – | – | – | – | Not run |
| v12 | LoRA on all 12 blocks, rank 32 | v10.3 | 61.8 (3) | +0.6 | 67.9 | 27.0 | 57.4 | Not kept |
| v13 | Self-supervised local-global consistency loss | v5.5 | 56.5 (1) | -0.4 | 66.1 | 41.2 | 59.5 | Not kept |
| v13.1 | SSL loss weight 5 -> 20 | v13 | 55.6 (1) | -0.9 | 68.3 | 28.9 | 57.9 | Not kept |
| v14.1 | v10.3 recipe on SigLIP | v10.3 | 59.7 (1) | – | 69.2 | 29.1 | – | Not compared |

### D.4 Feature generation (GAN)

| Version | Change from parent | Parent | Val HM (splits) | Δ val vs parent | Test HM, calibrated | Test HM, γ = 0 | Test AUSUC | Decision |
|---|---|---|---|---|---|---|---|---|
| v11 | f-CLSWGAN on frozen CLIP features | v4.2 | 49.2 (3) | +2.7 | 57.8 | 24.2 | 42.6 | Kept |
| v11.1 | GAN conditioned on pair + crop + disease text | v11 | 50.7 (3) | +1.5 | 59.7 | 6.2 | 44.2 | Not kept |
| v11.2 | No generated features (ablation) | v11 | 37.0 (3) | -12.2 | 41.9 | 0.0 | 24.9 | Ablation |
| v11.3 | LoRA-adapted features (v10.3 encoder) | v11 | 60.2 (3) | +11.0 | 64.9 | 44.0 | 52.7 | Kept (E) |
| v11.4 | Compositional condition on LoRA features | v11.1 | 61.9 (3) | +11.1 | 66.1 | 31.4 | 54.2 | Not kept |
| v11.5 | v5.13 encoder features | v11.3 | 59.4 (3) | -0.8 | 66.7 | 37.7 | 54.9 | Not kept |
| v11.6 | v5.8 encoder features (no text anchor) | v11.3 | 59.8 (3) | -0.4 | 64.1 | 41.7 | 51.8 | Not kept |

### D.5 Ensembles and routing with a CNN

| Version | Change from parent | Parent | Val HM (splits) | Δ val vs parent | Test HM, calibrated | Test HM, γ = 0 | Test AUSUC | Decision |
|---|---|---|---|---|---|---|---|---|
| v6 | Ensemble with ResNet-50 CNN baseline | v5 | 53.1 (1) | +1.8 | 70.1 | 19.8 | 61.0 | Kept |
| v6.1 | CLIP member v5.5 | v6 | 58.9 (1) | +5.8 | 70.8 | 25.7 | 61.2 | Kept |
| v6.2 | CLIP member v5.7 | v6.1 | 58.2 (1) | -0.7 | 70.5 | 22.4 | 61.2 | Not kept |
| v6.3 | Both members upgraded: v10.3 + ConvNeXt-Base | v6.1 | 66.0 (1) | – | 75.0 | 27.0 | 68.4 | Kept |
| v6.4 | ConvNeXt-Base at 336 px | v6.3 | 67.6 (1) | +1.6 | 74.6 | 19.7 | 67.5 | Not kept |
| v6.5 | CLIP member v5.13 | v6.3 | 67.1 (1) | +1.1 | 76.0 | 26.7 | 68.6 | Not kept |
| v6.6 | CLIP member v5.13 | v6.4 | 68.2 (1) | +0.6 | 75.5 | 20.2 | 68.1 | Not kept |
| v6.7 | Three members (+ ConvNeXt-Base 224) | v6.6 | 68.7 (1) | +0.4 | 77.1 | 23.5 | 69.6 | Not kept |
| v6.8 | CLIP member v11.3 | v6.3 | 68.8 (1) | +2.8 | 74.2 | 29.6 | 68.5 | Kept |
| v15 | Learned seen/unseen gate instead of gamma | v6.3 | 65.6 (1) | -0.4 | 75.0 | 75.0 | – | Not kept |

## Appendix E: Ablations and negative results

Each entry gives the hypothesis, the validation result against its parent, and the decision. Test numbers are reported for completeness only and were never used to decide. A change was kept only if its validation HM beat its parent by more than 2 points. The measured run-to-run noise is about 3 points: v5 retrained with the same split, seed and settings scored 48.3% against 51.2%.

### E.1 Text side: what the prompts say

**Do the descriptions help because of what they say, or just because they add text?** (v3.1) Replacing each description with random letters of the same length gives HM 5.5%, the same as no description (v2: 5.6%) and far below v3 (10.2%). The gain in section B comes from the meaning of the symptom words, not from longer prompts.

**Expert taxonomy instead of visual symptoms** (v3.3–v3.6). The official taxonomic descriptions helped seen pairs and ranking among the unseen pairs, but hurt unseen Top-1 (HM 8.0%).
- **One sentence:** combining taxonomy and symptoms in one sentence let the taxonomy dominate whatever the word order (v3.4: 8.4%, v3.5: 8.5%).
- **Two prompts:** keeping them as two separate prompts and averaging gave the best zero-shot result, HM 11.9% (v3.6).
- **After training:** the same two-sentence setup did not help once the model was trained. v5.7 scored 1.3 points below v5.5 on validation, and swapping it into v5.5 without retraining (v5.6) lowered test HM.

**Longer descriptions** (v3.7, v3.8). Descriptions of about 21 words instead of 8 lowered HM in both setups (7.9% and 10.1%). Longer texts share more generic phrases ("leaves yellowing", "drying"), so diseases become less distinct. Short, specific symptom words work best.

**Scoring descriptors separately** (v3.2). Averaging the scores of separate descriptor sentences dropped HM to 3.8%. Ranking among the unseen pairs was unchanged (49.0%), but without renormalisation the scores became unbalanced between classes.

**Rewritten descriptions for near-duplicate diseases** (v8). Rewriting the 33 descriptions that were nearly identical in CLIP's text space had no effect on validation (−0.1).

**Learned context words** (v5.12, v5.13). CoOp, which learns the words around a fixed class name, was no better than CSP (−0.6). Learning both the context and the crop/disease vectors (CoOp + CSP, v5.13) was kept: +2.8 points, higher on all three splits.

**Several descriptions per disease** (v5.10, PLID). +0.9 points, within noise, at twice the training time. Not kept.

**Keeping the learned vectors close to their names** (v10.1, v10.3, text anchor). +1.4 points on one split and +1.4 on three splits, higher on every split. Consistent, but below the 2-point bar. v11.6 later showed the anchor also makes no difference to the GAN's features.

### E.2 Training the crop and disease vectors

**Alternating crop and disease updates** (v5.9, DRPT). +0.8 points, within noise. Not kept.

**Separate visual branches for crop and disease** (v5.11, Troika). +2.5 points, kept by the rule, but mostly through post-hoc fusion of the crop and disease scores: without fusion the branches gave only +1.0. Adding them to v5.13 (v5.14) gave +0.5, so the simpler v5.13 was kept.

**Post-hoc fusion alone** (v9). Fusing the crop and disease prompts into the pair score at test time gave +1.6 points, but the confidence interval included zero. Not kept.

**Same-crop hard negatives** (v7.2). An extra loss that ranks each image only against other pairs of its crop changed nothing (−0.5).

### E.3 Image side: the encoder and its inputs

**LoRA capacity** (v5.3, v12). Going from rank 8 on 4 blocks to rank 16 on 8 blocks gave +5.7 points, the clearest image-side gain. Going further, to rank 32 on all 12 blocks (v12), gave +0.6: it raised validation-seen accuracy but not pseudo-unseen accuracy.

**Input resolution and augmentation** (v5.1, v5.2, v7.1). 336 px input (−0.2), test-time flips (+0.4) and zoomed training crops (−1.6) did not help. Resolution and framing are not the bottleneck.

**Lesion-level scoring** (v7). Matching each disease against image patches instead of the whole-image embedding gave +1.5 on validation, but lowered test HM (62.2% vs 64.5%). Not kept.

**Class-balanced sampling** (v5.4). −2.2 points, consistently below v5 from epoch 10 on. Square-root balancing draws pairs with five or fewer training images about four times as often, which likely overweights the rarest pairs.

**Keeping the image features close to frozen CLIP** (v10, image consistency) and **weight averaging** (v10.2, EMA). +0.5 and +0.4. Not kept.

**Self-supervised local–global consistency** (v13, v13.1). −0.4 at weight 5, and −1.2 at weight 20, where the loss was actually optimised. Not kept, although its test AUSUC was the highest of v5.5's single-change variants. Re-validating it on three splits would settle this.

**Other backbones** (v3.9, v4.3, v14, v14.1).
- **BioCLIP:** worse than OpenAI CLIP, both zero-shot (HM 3.3% vs 10.2%) and with prompt learning (48.6% vs 51.8%). It recognised crops in only 23% of seen images.
- **SigLIP:** better zero-shot (14.3%) and promising with the v10.3 recipe (test 69.2%). It was run on one validation split, on another machine, so it isn't comparable with the three-split results.

### E.4 Feature generation

**Do the generated features matter?** (v11.2) Training the same classifier without generated unseen features drops validation HM by 12.2 points (49.2% → 37.0%). Generation is essential to the GAN branch.

**Which features the GAN runs on** (v11 → v11.3, v11.5, v11.6). Moving from frozen CLIP to LoRA-adapted features gives +11.0, the GAN branch's largest gain. Which LoRA encoder is used makes little difference: v10.3 (text anchor), v5.13 (CoOp + CSP) and v5.8 (no anchor) all fall within 1 point.

**Conditioning on crop and disease text** (v11.1, v11.4). Adding the crop and disease embeddings to the generator's condition gives about +1.5 on both feature types, consistent but below the bar. On LoRA features (v11.4) it reaches 61.9% validation HM.

**GAN against learned prompts on the same features.** On the same LoRA features, the GAN classifier never beats the prompt model (v11.3 60.2% vs v10.3 61.2%). It trades unseen recognition for a smaller seen bias. Its value is as a different ensemble member (v6.8).

**Crop-anchored GAN as an extra disease head in the ensemble** (validation only, `train/gan_head_mix.py`). Hein's crop-anchored recipe raised the head's pseudo-unseen disease accuracy from 6% to 20–23%, but added only +0.3 to +0.4 to the ensemble's validation HM. In hard routing it matched ConvNeXt alone (64.0% vs 63.6%). Not adopted.

### E.5 Calibration and routing

**Validation split design** (v4 → v4.1, v4.2). v4's validation chose a γ that sent 53% of seen test images to unseen pairs. Two fixes corrected this. The 10 real unseen pairs were hidden from the validation label space, so validation sees as many unseen candidates as test does. A lower learning rate made training steadier. Three splits instead of one (v4.2) confirmed the noise: each split's best γ differed (0.0425, 0.0275, 0.0375).

**A γ per pair instead of one γ.** +3.7 on validation, but worse on test. Rejected; 20 pseudo-unseen pairs are likely too few to fit a separate γ for each pair.

**A learned seen/unseen gate** (v15). A logistic gate on both members' scores scored −0.4 against the γ ensemble on the same split. Three further gate families were tried, built on nearest-neighbour support, training counts, and the margins of four members. None beat a single γ: their detection accuracy stayed at AUROC 0.92–0.93. The reason is in the data. 89% of misrouted seen images are confused with an unseen pair of their own crop, so deciding seen versus unseen is in practice a within-crop disease decision.

### E.6 Ensembles with a CNN

Mixing a CLIP model with a CNN on a log-probability scale consistently helped: v6 (v5 + ResNet-50) scored test HM 70.1% against 65.1% and 59.7% for its members. Better members helped further (v6.3: v10.3 + ConvNeXt-Base, 66.0% validation). Later member swaps and a three-member mix (v6.4–v6.7) each added 0.4–1.6 points, within noise. v6.8 (v11.3 + ConvNeXt) passed the rule (+2.8), because the GAN model makes different errors from the ConvNeXt. The best test result in this family is v6.7, at 77.1%.
