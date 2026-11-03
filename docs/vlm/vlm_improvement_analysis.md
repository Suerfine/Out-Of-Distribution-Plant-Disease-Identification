# VLM pipeline: which operations to improve

Companion to `docs/diagrams/vlm_pipeline.drawio` (open it at app.diagrams.net, or regenerate it with
`python docs/diagrams/make_vlm_pipeline.py`). Operation ids (A1, C6, H5, ...) refer to that diagram. Base model: v5.5,
validation HM 56.9%; test HM 64.5%, test seen disease top-1 56.2%, test unseen disease top-1 84.4%.

## What the experiments so far say

| Operation | Tried | Validation HM vs base | Verdict |
|---|---|---|---|
| C6 LoRA capacity | v5.3: 8 blocks, rank 16 | +5.7 | kept (the only clear win since v5) |
| H11 ensemble with the CNN | v6 | +1.8 | kept as a separate run family |
| C7 pooling | v7: patch-token disease score | +1.5 | not kept, test went down |
| F1 / H3 test views | v5.2: flip TTA | +0.4 | not kept |
| C2 / C4 input | v5.1: 336 px | -0.2 | not kept |
| E3 loss | v7.2: within-crop pair CE | -0.5 | not kept |
| C2 augmentation | v7.1: crop scale 0.25-1 | -1.6 | not kept (test went up) |
| C1 sampling | v5.4: class-balanced | -2.2 | not kept |
| H6 calibration | per-pair gamma | +3.7 val, worse test | rejected |
| B2 descriptions | v8: contrastive rewrites | running | |

Two patterns stand out:

1. **Only changes to how much the model can change its image features helped.** Text, loss, sampling, augmentation and
   pooling tweaks all landed inside the ~3-point run-to-run noise. The image side (C4-C8) is the bottleneck: 70-75% of
   seen disease errors already have the crop right, so the image embedding does not separate diseases within a crop.
2. **The decision procedure (F) is too noisy to judge most ideas.** One validation split (20 pseudo-unseen pairs,
   ~800 images) plus a 2-point margin over a 3-point noise means a real +2 gain is often missed, and v7.1 shows
   validation and test can disagree in sign. Fixing F makes every later experiment cheaper to judge.

## Opportunities, by operation

Ranked by expected value per GPU hour. "Retrain" = needs a new training run (~45 min).

### 1. H5 + E4/E5: use the crop and disease heads at prediction time (no new training method)

The crop prompts (E4) and disease prompts (E5) are trained every step, but prediction (H5) uses only the pair score.
Compositional zero-shot work combines the pair, state and object "experts" at inference: HPL learns three hierarchical
prompts and fuses them [3], and PLID's primitive-decomposition module fuses composition and primitive decisions [4].
For an unseen pair the primitive scores are exactly the evidence that transfers, because each primitive was trained on
other pairs.

- Score = pair + a·crop(pair's crop) + b·disease(pair's disease), with a, b (and gamma) chosen on validation.
- Cost: one stage-1 refit at the selected epoch that saves the validation embeddings (train/ensemble.py already does
  this for its members), then free sweeps. The text embeddings are recomputable from `prompt_vectors.pt`.
- Expected: the most direct lever on "disease wrong, crop right". Can be rejected cheaply if it does nothing.

### 2. C6: interpolate the LoRA update toward zero (WiSE-FT style), chosen on validation

Scaling the LoRA delta, W + s·(alpha/r)·BA with s in [0, 1], moves between frozen CLIP (s = 0, best zero-shot
generality) and the fine-tuned model (s = 1, best seen fit). Weight-space interpolation is the standard fix for
fine-tuning hurting out-of-distribution accuracy, and it fits the HM objective, which trades seen against unseen. It
costs no extra training, and the same stage-1 refit as idea 1 serves both sweeps.

### 3. C4-C8 backbone: a disease-pretrained CLIP (not BioCLIP; that was already tried)

BioCLIP [5] has already been tested and dropped:

- **v3.9 (zero-shot):** HM 3.3% vs 10.2% for OpenAI CLIP.
- **v4.3 (prompt learning, frozen encoders):** HM 48.6% vs 51.8% for v4.2.

Its image features fit the seen pairs as well as CLIP's, but unseen accuracy fell from 69.7% to 54.7%. The likely
cause is its text encoder, trained on taxonomic captions, which makes it a poor base for composing pairs. LoRA (v5)
adapts only the image side, so the text weakness would remain. BioCLIP 2 [6] keeps the same taxonomic-caption
objective. Neither is worth another run.

A backbone whose *text* side knows disease symptoms would be different. SCOLD was trained on 186k leaf images paired
with symptom captions, and it beats CLIP-L, BioCLIP and SigLIP2 on zero-shot and few-shot leaf disease [1]. Try it only
if:

- its weights load in open_clip with a 77-token text encoder (otherwise it is an engineering job), and
- its zero-shot HM on our prompts beats OpenAI CLIP's v3 score (10.2%).

That zero-shot test is a cheap filter, about one minute with `train.zeroshot`.

### 4. E6: regularise toward frozen CLIP and ensemble over epochs

Base-to-novel prompt learning is our seen-to-unseen setting. Several methods prevent the fine-tuned model from
drifting away from CLIP's general features:

- PromptSRC [7]: (a) an L1 agreement loss between fine-tuned and frozen features, (b) a Gaussian-weighted self-ensemble
  of the prompts over the training trajectory, (c) more textual diversity.
- CoPrompt [8]: a consistency loss between the trainable and pre-trained models, applied on perturbed inputs.
- KgCoOp-style anchoring keeps the learned text embeddings near the hand-written prompt embeddings; DeKg reports that
  this anchoring alone biases toward pre-training and adds an independence term [9].
- Class-similarity alignment keeps the pairwise similarity structure of the hand-crafted prompts [10].

For us that means:

- an L1 term between the LoRA image feature and the frozen-CLIP feature of the same image (one extra frozen forward,
  under no_grad);
- an anchor on the text embeddings to their initial values;
- an exponential moving average (or Gaussian average) of the vectors and LoRA weights across epochs.

The weight average also smooths the epoch choice in F5, which is itself a noise source. One retrain per variant, so
add the terms one at a time.

### 5. B2/B3: several image-grounded descriptions per disease instead of one

v8 tests whether *contrastive* wording helps. The next step up is *diversity*, which `build_prompts` and
`prompt_learner` already support (a description may be a list):

- PLID turns several LLM-generated descriptions per class into a class distribution, which improves compositional
  zero-shot results [4].
- DPR adds LLM-generated descriptive sentences per composition [11].
- For crop disease specifically, captions written by Qwen-VL from representative *training images* beat captions
  written by a text-only LLM [2]. That fits our problem: our v1 descriptions are generic and unverified.

Cost: generating the text (an offline VLM pass over a few training images per disease), then one retrain.

### 6. F1-F5: make decisions less noisy

- **Paired comparison.** Both runs score the same validation images, so compare them per image with a McNemar test or
  a bootstrap confidence interval of the HM difference, instead of a fixed 2-point margin. Free: the predictions
  already exist in each run's history. This would have said whether v7's +1.5 was noise.
- **Two or three validation splits.** `validation.n_splits` is already supported; v4.2 used 3. This roughly doubles or
  triples stage-1 time, so turn it on only for final candidates.
- **Weight averaging** (idea 4) reduces the epoch-to-epoch variance that F5 picks from.

### 7. C7: local lesion evidence, done better than v7

v7's top-k patch score helped validation (+1.5) but hurt seen disease on test, likely because raw patch tokens are not
aligned with text (CLIP never trained them to be). Approaches that worked in the literature add region guidance:

- AG-CLIP mines attributes from class descriptions and uses an open-vocabulary detector to crop the matching regions;
  it reports 84.6% zero-shot on a plant disease set [12].
- MPPE adds SAM masks as an extra alpha channel into the CLIP visual encoder [13].

Both cost a lot more (a detector or SAM pass per image, cached once). Lower priority than ideas 1-4.

### 8. E2/E3: soften the targets between near-duplicate classes

The pair CE treats `septoria` vs `septoria_leaf_spot`, or `mildew` vs `powdery_mildew`, as fully wrong. SCOLD's soft
contrastive targets reduce overconfidence and improve fine-grained generalisation [1]. A label-smoothing variant that
gives some target mass to pairs sharing the crop or the disease name, or a learnable logit scale instead of the fixed
100, is a one-line loss change. Expected gain: small.

### 9. Lower value, or already covered

- **C2 augmentation, C1 sampling, C4 resolution.** Three tries, all noise or negative.
- **H6 calibration beyond one gamma.** Per-pair calibration overfit, because the pseudo-unseen pairs are not
  representative (`tomato_early_blight` is 472 of the 1,448 unseen test images).
- **H10 retrieval.** It works and is not evaluated by the protocol; changing it would only follow from a backbone
  change.
- **Generating unseen-pair features** (e.g. Duplex's recombined visual prototypes for unseen pairs [14]). Promising for
  seen bias, but a larger engineering job.

## Suggested order

1. **One stage-1 refit of v5.5 that saves its validation embeddings and model**, then sweep idea 1 (primitive fusion)
   and idea 2 (LoRA scale) on validation. Use idea 6's paired test for the decision. About 20 minutes of GPU time in
   total.
2. **Idea 4, one term at a time.**
3. **Idea 3: zero-shot check of SCOLD**, if its weights load in open_clip; train with it only if it passes.
4. **Idea 5, if v8 shows that description text matters at all.** If contrastive rewrites do nothing, more text is
   unlikely to do much either.

## References

1. [A Vision-Language Foundation Model for Leaf Disease Identification](https://consensus.app/papers/details/f0e61317fff155b094cac40f39101aff/?utm_source=claude_desktop) (Nguyen et al., 2025, Expert Syst. Appl.), SCOLD
2. [Few-Shot Image Classification of Crop Diseases Based on Vision–Language Models](https://consensus.app/papers/details/ea4b240cd05e5bd6b97c9e24deaa6dab/?utm_source=claude_desktop) (Zhou et al., 2024, Sensors)
3. [Hierarchical Prompt Learning for Compositional Zero-Shot Recognition](https://consensus.app/papers/details/e4b2d4f6424a5f7b921a9c9dca5bf753/?utm_source=claude_desktop) (Wang et al., 2023)
4. [Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](https://consensus.app/papers/details/5804c71a2440587d86c19aa2ef241516/?utm_source=claude_desktop) (Bao et al., 2023), PLID
5. [BioCLIP: A Vision Foundation Model for the Tree of Life](https://consensus.app/papers/details/f6e8f9806b875b86b8a037eafa870ff1/?utm_source=claude_desktop) (Stevens et al., CVPR 2024)
6. [BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](https://consensus.app/papers/details/fe49722e7a745479aa748c597f705e6a/?utm_source=claude_desktop) (Gu et al., 2025)
7. [Self-regulating Prompts: Foundational Model Adaptation without Forgetting](https://consensus.app/papers/details/4f799e31fcac5855b0981cc4067973c7/?utm_source=claude_desktop) (Khattak et al., ICCV 2023), PromptSRC
8. [Consistency-guided Prompt Learning for Vision-Language Models](https://consensus.app/papers/details/4e24b2c7ffd55d409ffb01d3e81b3e7c/?utm_source=claude_desktop) (Roy et al., 2023), CoPrompt
9. [Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning](https://consensus.app/papers/details/06a639bacb2d557a9938d465ad1626f3/?utm_source=claude_desktop) (Li et al., 2025), DeKg
10. [Learning Generalizable Prompt for CLIP with Class Similarity Knowledge](https://consensus.app/papers/details/539e91f75d7a5e60b92e45436ab88a61/?utm_source=claude_desktop) (Jung et al., 2025), SAR
11. [Learning Diversified Primitive Prompts for Compositional Zero-Shot Learning](https://consensus.app/papers/details/87a0e8e5813853cdb38f57c19b2b375c/?utm_source=claude_desktop) (Chen et al., 2025), DPR
12. [AG-CLIP: Attribute-Guided CLIP for Zero-Shot Fine-Grained Recognition](https://consensus.app/papers/details/3e978a29dc315278a3938b58dbadb210/?utm_source=claude_desktop) (Ahmad et al., 2026)
13. [Multi-Modal Prompts With Primitives Enhancement for Compositional Zero-Shot Learning](https://consensus.app/papers/details/f34ac24950dd5da88779ff2f1fb650de/?utm_source=claude_desktop) (Jin et al., 2025), MPPE
14. [Semantically guided dynamic visual prototype refinement for compositional zero-shot learning](https://consensus.app/papers/details/c033831770245c50a7697828acecd28a/?utm_source=claude_desktop) (Peng et al., 2025), Duplex

WiSE-FT (weight-space ensembling of zero-shot and fine-tuned models, Wortsman et al., CVPR 2022) is cited from general
knowledge; it was not returned by the literature search.
