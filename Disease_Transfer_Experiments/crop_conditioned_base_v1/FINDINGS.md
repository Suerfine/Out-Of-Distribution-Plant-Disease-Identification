# Completed Base experiment: no improvement over the saved baseline

Training, final official inference and independent numerical verification
completed on 6 November 2026 at 11:35 pm Malaysia time. The training plot was
visually reviewed later that evening. Keep the saved original Base baseline.

## Same existing calibration: gamma6.2, unchanged

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved ConvNeXt-Base | 67.00% | 76.52% | 71.44% |
| Crop-conditioned Base | 67.40% | 72.58% | 69.90% |

The new model correctly classifies 1,348/2,000 seen images versus 1,340 for
the baseline, and 1,051/1,448 unseen images versus 1,108. This is eight more
seen predictions but 57 fewer unseen predictions: unseen -3.94 percentage
points and H approximately -1.54 points.

The paired image-bootstrap unseen-difference interval is [-5.87,-1.93] pp.
The whole-composition interval is wider, [-10.77,+1.60] pp, and includes zero.
These are conditional on one training seed and the observed test data. This
configuration's observed result is worse; it does not establish that all
crop-conditioning methods fail or quantify variability across new trainings.

## Uncalibrated: gamma0

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved ConvNeXt-Base | 73.80% | 14.02% | 23.56% |
| Crop-conditioned Base | 74.15% | 13.74% | 23.19% |

Uncalibrated unseen counts are 199 versus 203. Neither scoring policy improved
the saved baseline's unseen accuracy or H. There was no new calibration search
or selection between scoring policies, epochs or experimental configurations.

## What was tested and verified

Only the current experiment was replaced with Base. Earlier Tiny experiments
remain preserved. The experimental disease network starts from the saved
full-data Base disease weights; the original Base crop model/logits are reused.
All disease parameters are trainable, including the 87,566,464-parameter Base
backbone and a shared crop-conditioned feature modulation module.

The run completed its fixed15 additional epochs on all8,000 training images,
with no new validation partition or checkpoint selection. All333 pair
candidates are scored using candidate-specific disease normalization and crop
probabilities; no true crop label is supplied at inference. It tests the full
warm-started fine-tuning recipe, without isolating conditioning from extra
training through a new control.

Ten mechanism tests and the discarded real-GPU probe passed. Final numerical
verification passed **35,164 checks**, including original Base/source/data hashes,
full image exposure, final checkpoint identity, actual Base backbone updates,
independent probabilities/rankings and both existing scoring policies.

An interruption and an epoch9 checkpoint serialization failure required
resuming from the last complete checkpoints. The logged I/O-only recovery
added explicit-stream writes, fsync, ZIP CRC verification and storage-error
retries. Its two tests and a complete1GB checkpoint write/read probe passed.
The frozen model/training/data/scoring sources were unchanged; final checkpoint
writes and operational provenance verification passed. No further training is
queued.

See [full results](runs/full_seed42/results.md),
[numerical verification](runs/full_seed42/independent_verification.json) and
[paired analysis](runs/full_seed42/paired_analysis.json).
