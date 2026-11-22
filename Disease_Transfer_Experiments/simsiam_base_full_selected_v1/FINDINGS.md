# Full-data selected SSL findings

Completed **8 November 2026 at 1:36 pm Malaysia time** on the first worker attempt. The full-data 20-epoch SSL encoder was verified and reused unchanged. A fresh disease head and optimizer then trained the full disease encoder/head for the development-selected **9 supervised epochs on all 8,000 images**. Official test evaluation followed the frozen budget; no new calibration or test-based checkpoint selection occurred.

## Primary official test results: no calibration

Every image is ranked against all 333 candidate combinations. The saved full-data Base crop model/logits and original Base comparison are reused.

| Model | Seen Top-1, n=2,000 | Unseen Top-1, n=1,448 | H |
|---|---:|---:|---:|
| Saved Base | 73.80% (1,476) | 14.02% (203) | 23.56% |
| Earlier full SSL, 20 SSL + 15 supervised | 72.60% (1,452) | 13.05% (189) | 22.13% |
| Selected full SSL, 20 SSL + 9 supervised | 73.00% (1,460) | 14.30% (207) | 23.91% |

Against Base, the new run changes seen accuracy by **-0.80 percentage points**, unseen by **+0.28 points**, and H by **+0.35 points**. The unseen gain is four additional correct predictions; seen loses sixteen. The shorter supervised budget improves the observed scores over the earlier 15-epoch SSL model, but this comparison does not establish an optimal epoch count.

The raw H point estimate is slightly higher than Base. The paired bootstrap over compositions gives a 95% interval for that H difference of **[-0.90, +1.84] percentage points**, including zero. Its unseen-accuracy difference interval is [-0.64, +1.18] points. These intervals describe variation under resampling the observed test compositions; they do not measure training-seed variability. This is a small exploratory gain from one seed, not convincing evidence of a dependable improvement. Preserve the candidate and the Base reference; do not claim a broad disease-recognition improvement from four additional correct pair predictions.

## Secondary historical comparison: unchanged fixed gamma 6.2

| Model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Saved Base | 67.00% (1,340) | 76.52% (1,108) | 71.44% |
| Selected full SSL | 65.80% (1,316) | 77.42% (1,121) | 71.14% |

With the existing fixed score offset, unseen gains 13 correct predictions while seen loses 24; H falls by 0.31 percentage points. Gamma 6.2 was not retuned for this model. Raw results remain primary. The previously inspected official tests provide descriptive comparisons rather than untouched confirmation.

## Execution and evidence

- Completed the fixed nine supervised epochs and both official test partitions, with no worker restart.
- All **33,937 independent verification checks passed**, covering unchanged source/data/reference hashes, SSL-reuse provenance, full per-epoch image exposure, fresh-head initialization, encoder transfer and training, frozen model identity, score construction, prediction exports, and raw/fixed-gamma metrics.
- The 13 setup tests and actual-encoder GPU update/exact checkpoint-replay probe passed before training.
- The training figure was visually reviewed: panels, curves, labels, and legends are readable; the reused SSL history is identified explicitly. Supervised loss decreases through epoch 9. This is not by itself evidence of improved test accuracy.
- The baseline and earlier models are preserved. This run is complete; no further training is queued by it.

Final disease checkpoint SHA256: `44a0aa6bd48661c6245266aa5c0157ab8e17a8e53a3031c386308eca4811bbf8`.

Saved [results](runs/full_selected_seed42/results.md), [metrics](runs/full_selected_seed42/metrics.json), [paired analysis](runs/full_selected_seed42/paired_analysis.json), [verification](runs/full_selected_seed42/independent_verification.json), [SSL reuse record](runs/full_selected_seed42/ssl_reuse.json), and [training figure](runs/full_selected_seed42/training_curves.png). See the [fixed protocol](PLAN.md) for training and selection details.
