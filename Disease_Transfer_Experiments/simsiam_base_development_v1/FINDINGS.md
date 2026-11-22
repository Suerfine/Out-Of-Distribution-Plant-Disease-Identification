# SimSiam Base development findings

Completed **8 November 2026 at 5:32 am Malaysia time**. The run finished 20 SSL epochs and 17 supervised disease epochs, then stopped under the declared patience-8 rule. Supervised epoch 9 was the earliest maximum of uncalibrated validation composition Top-1 H. The existing Base reference remains better on that measure.

These are **development validation** results on 862 seen and 533 pseudo-unseen images, not the official 2,000/1,448-image tests. The exact original Base partition, saved development crop logits, and baseline were reused. Gamma was zero throughout selection and reporting; no calibration sweep or control retraining occurred.

| Model | Seen validation Top-1 | Pseudo-unseen Top-1 | H |
|---|---:|---:|---:|
| Saved Base reference | 89.68% (773/862) | 4.50% (24/533) | 8.58% |
| Selected SSL development model | 89.79% (774/862) | 3.75% (20/533) | 7.20% |
| Change, percentage points | +0.12 | -0.75 | -1.37 |

On pseudo-unseen images, SSL fixes 3 baseline mistakes but introduces 7 regressions, a net loss of 4 correct predictions. Seen validation gains one net correct prediction (23 fixes, 22 regressions). Pseudo-unseen pair Top-5 also falls, from 27.58% to 23.26%. The raw disease classifier's pseudo-unseen Top-1 is 3.75% for Base and 3.38% for SSL. The crop branch remains identical.

The selected epoch has seen/pseudo-unseen H of 7.20%. No later epoch improves it; after eight nonimproving supervised epochs, training stops at epoch 17 as planned. Lower training loss and lower SSL loss did not translate into better held-out-combination accuracy. This single-seed comparison does not prove that SSL as a whole is ineffective or establish a particular internal training defect.

## Execution and verification

The supervisor used two worker attempts. The first encountered a Windows `PermissionError` while atomically replacing `status.json` during validation. It restarted automatically from a checked checkpoint and finished successfully. No source, recipe, or data changes were made during recovery.

The final verification passed **131 checks**, including unchanged source/reference hashes, exactly one exposure per fitting image in every completed epoch, earliest-maximum selection, selected checkpoint/logit checksums, and reproduction of selected validation metrics. The startup GPU probe and all 12 setup tests also passed. These checks address execution integrity; they do not establish that the scientific recipe is beneficial.

No official-test inference or full-data refit was started. Recommendation: retain the Base reference and close this SSL recipe at the development stage. Use the development protocol to judge any separately declared next recipe before committing more full-data compute.

## Saved evidence

- [Results](runs/development_seed42/results.md) and [full metrics](runs/development_seed42/results.json)
- [Selection record](runs/development_seed42/selection_frozen.json)
- [Verification](runs/development_seed42/verification.json)
- [Supervised history](runs/development_seed42/finetune/history.json) and [SSL history](runs/development_seed42/ssl/history.json)
- [Recovery events](runs/development_seed42/supervisor_events.jsonl)
- [Fixed protocol](PLAN.md)
