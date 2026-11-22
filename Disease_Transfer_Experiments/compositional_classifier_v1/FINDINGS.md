# Compositional classifier: severe failure on unseen combinations

**This experimental architecture did not improve the baseline. It retained
similar seen accuracy but almost always predicted a training combination for
images whose true combination was unseen. Keep the saved original baseline.**

Training, evaluation and independent numerical verification completed on
6 November 2026 at 2:20 am Malaysia time. Plot review and post-hoc diagnostics
were completed later on 6 November. No further training is queued.

## Primary comparison: all 333 candidates

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved original baseline | 71.05% | 12.43% | 21.16% |
| Learned compositional classifier | 70.55% | 0.14% | 0.28% |

The new model correctly predicts 1,411/2,000 seen images, versus 1,421 for the
baseline, and only **2/1,448 unseen images**, versus **180** for the baseline.
The unseen difference is -12.29 percentage points. Its paired whole-composition
bootstrap 95% interval is [-33.08, -0.98] pp; the paired image interval is
[-14.02, -10.64] pp. These intervals condition on one trained model and the
observed dataset, not repeated training seeds.

## What the saved predictions show

Post-hoc inspection finds **1,446/1,448 (99.86%)** unseen images classified into
one of the 323 training combinations. Only two predictions enter the ten unseen
combinations, and both are correct. This directly documents strong preference
for training pairs in the final ranking. It does not establish which particular
training mechanism caused that preference.

The true pair appears in the Top-5 for 608/1,448 unseen images (41.99%), despite
the very low Top-1. The auxiliary heads score 74.45% crop and 9.88% disease
Top-1 on these images. Generating a vector from known crop/disease components
did not make the unseen pair competitive enough to win the joint ranking.

A separately labeled diagnostic restricting predictions to the ten unseen
pairs gets 909/1,448 (62.78%). This supplies knowledge that the answer is unseen
and is an easier protocol. It is not the model's primary performance, is not
used in H, and did not change any checkpoint, scoring rule or reported result.
See [post-hoc diagnostics](runs/full_seed42/posthoc_diagnostics.json).

## Training and verification

One complete joint CNN was trained from the cached ImageNet-22K weights, with
all 29,191,776 parameters trainable, including 27,820,128 backbone parameters.
It learned shared crop/disease embeddings and a shared pair-composition MLP,
with no separately learned classifier row for each unseen combination.
This is a plant-specific architecture inspired by CZSL principles, not an exact
reproduction of CGE or CompCos.

The run used the original 6,746 fitting / 1,254 seen-validation manifests, no
pseudo-unseen validation and no retrained control. The accuracy-first rule was
applied to the primary pair output on these validation images, with pair CE
breaking ties; this was explicitly fixed before training. Pair training CE used
only the 323 seen classes, while all 333 candidates were scored at evaluation.
Auxiliary factor probabilities were never multiplied to replace the pair score.

Training stopped after 25 epochs and selected epoch 17, with 85.81% validation
pair Top-1. Fitting pair/crop/disease accuracy reached 100%. Recorded epoch time
was 1.62 hours, including loading and validation. Dynamic AMP skipped eight
optimizer updates; all selected tensors and final scores are finite. The run
completed without an unresolved numerical or export failure.

All nine implementation tests and the real-CNN GPU probe passed. Independent
verification passed **27,200 checks**, covering source/data hashes, unchanged
historical artifacts, complete image exposure, shared-factor training, seen-only
training candidates, validation selection, and NumPy recomputation of raw joint
scores and exported rankings. The training figure was visually reviewed.

This result rejects this particular configuration as a baseline replacement.
It does not prove that compositional zero-shot methods in general cannot work.
The model uses a different joint architecture and checkpoint target from the
historical two-network baseline, and official tests were familiar from earlier
studies. No test-selected rescue or extra experiment was performed.

## Artifacts

- [Generated results](runs/full_seed42/results.md) and [raw metrics](runs/full_seed42/metrics.json)
- [Paired comparisons](runs/full_seed42/paired_analysis.json)
- [Independent verification](runs/full_seed42/independent_verification.json)
- [Training curves](runs/full_seed42/training_curves.png)
- [Fixed protocol](PLAN.md)
