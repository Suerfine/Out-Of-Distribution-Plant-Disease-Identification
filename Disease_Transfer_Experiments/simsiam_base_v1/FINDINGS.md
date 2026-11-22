# Completed SSL experiment: saved Base baseline remains better

All20 SimSiam epochs and15 supervised disease epochs completed. Official
evaluation and33,909 independent verification checks passed on7 November2026
at11:35 pm Malaysia time. The training figure was visually reviewed afterward.
No further training is queued.

## Same existing calibrated scoring: gamma6.2

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved ConvNeXt-Base baseline | 67.00% | 76.52% | 71.44% |
| SimSiam → supervised Base | 66.10% | 74.65% | 70.12% |

The SSL pipeline correctly predicts1,322/2,000 seen images and1,081/1,448
unseen images, versus1,340 and1,108 for the baseline. That is18 fewer seen
and27 fewer unseen predictions. Unseen accuracy falls1.86 percentage points;
H falls approximately1.32 points.

The paired whole-composition bootstrap intervals are[-8.56,+2.95] pp for
the unseen difference and[-4.45,+1.18] pp for H; both include zero. These
intervals condition on this single trained seed and the observed test dataset.
This particular configuration's point estimates are worse; the experiment does
not establish the effect of other SSL budgets, settings or random seeds.

## Uncalibrated scoring: gamma0

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved ConvNeXt-Base baseline | 73.80% | 14.02% | 23.56% |
| SimSiam → supervised Base | 72.60% | 13.05% | 22.13% |

Raw unseen counts are189 versus203. Both existing scoring policies gave lower
seen/unseen accuracy and H. No new gamma, test-based checkpoint choice or
experimental rescue scoring was used. Keep the saved Base baseline.

## What this run establishes

SSL began from the same pinned ImageNet-pretrained Base encoder as the baseline.
It used only the8,000 fitting images without crop/disease labels, with two
augmented views, stop-gradient SimSiam loss and real64-image normalization
batches implemented through gradient caching. All20 fitting-feature monitors
passed the noncollapse thresholds. View alignment improved, but that did not
produce an improvement in final disease/composition accuracy in this recipe.

The SSL heads were removed. A fresh disease head used the original seed, and
the whole SSL encoder plus head were fine-tuned for15 epochs. The original
crop model/logits, all333 candidates and frozen gamma0/gamma6.2 scoring settings
were reused. There was no new control, validation split or test-image SSL.
The experiment tests the complete adapted recipe, including extra SSL compute;
it does not isolate every design choice.

Independent checks confirm original source/data/Base hashes, deterministic
full exposure in both stages, actual updates to the Base encoder, exact SSL
encoder transfer, original classifier initialization, final checkpoint integrity
and independently reconstructed probabilities, stable rankings and metrics.

The run paused after SSL at the user's request, verified/saved the encoder,
and resumed only after the later “continue.” Supervised training completed
before a ranking-export API argument error interrupted evaluation. A tested,
evaluation-only compatibility adapter passed the dimension by keyword to the
same Torch stable-sort kernel. The frozen source files and final model checksum
were unchanged, no training was repeated, and final verification passed.

See [full results](runs/overnight_seed42/results.md),
[verification](runs/overnight_seed42/independent_verification.json),
[paired analysis](runs/overnight_seed42/paired_analysis.json) and
[evaluation-only recovery receipt](runs/overnight_seed42/evaluation_recovery_verified.json).
