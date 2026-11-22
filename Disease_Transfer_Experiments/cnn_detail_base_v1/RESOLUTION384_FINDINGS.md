# Resolution384 development result

Completed **9 November 2026 at 3:56 am Malaysia time** on the first worker attempt. Training stopped after 15 epochs under patience 8 and selected epoch 7 by maximum raw validation H. All 14,798 final verification checks passed.

These are uncalibrated development-validation results on 862 seen and 533 pseudo-unseen images, with 323 candidates. The original crop logits and Base reference were reused.

| Model | Seen Top-1 | Pseudo-unseen Top-1 | H |
|---|---:|---:|---:|
| Saved Base224 | 89.68% (773/862) | 4.50% (24/533) | 8.58% |
| Resolution384, selected epoch 7 | 89.33% (770/862) | 3.75% (20/533) | 7.20% |

This recipe did not improve the primary development measure. It loses three correct seen and four correct pseudo-unseen predictions relative to Base. This is one seed and one higher-resolution preprocessing/training recipe; it does not rule out all uses of higher-resolution CNN inputs.

The second, already-authorized global/local patch-attention arm started automatically. Its result remains pending as of the 9 November morning status check. No full-data refit or official-test evaluation has been performed for this study.

Saved [metrics](runs/development_seed42/resolution384/results.json), [report](runs/development_seed42/resolution384/results.md), [verification](runs/development_seed42/resolution384/verification.json), and [selection](runs/development_seed42/resolution384/selection_frozen.json).
