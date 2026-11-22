# Feature-composition Base development findings

Completed **8 November 2026 at 5:02 pm Malaysia time**, on the first worker attempt. The run reused the fitting-only 20-epoch SSL encoder, completed **13 new composition-training epochs**, and stopped under the declared patience-8 rule. The earliest maximum of raw validation H was **epoch 5**.

The selected recipe did not improve either saved development reference. These are uncalibrated results on **862 seen-validation and 533 pseudo-unseen-validation images**, with 323 development candidates. They are separate from the official test results of the earlier full-data SSL models.

| Model | Seen validation Top-1 | Pseudo-unseen Top-1 | H |
|---|---:|---:|---:|
| Saved Base | 89.68% (773/862) | 4.50% (24/533) | 8.58% |
| Saved SSL development | 89.79% (774/862) | 3.75% (20/533) | 7.20% |
| Feature composition, selected epoch 5 | 87.47% (754/862) | 2.44% (13/533) | 4.75% |

Against Base, the new recipe loses 19 correct seen predictions and 11 correct pseudo-unseen predictions. On pseudo-unseen images it fixes 4 baseline mistakes but breaks 15 previously correct predictions. Pseudo-unseen pair Top-5 also falls, from 27.58% to 15.38%. Raw disease-head Top-1 falls from Base's 3.75% to 3.00%; the reused crop logits remain identical.

The intended feature synthesis did run: 75,374 primary image exposures, 150,088 total unique-within-step encoded image exposures including donors, and **37,856 synthetic features**, including **18,928 novel-pair targets**, across 13 epochs. Donors were drawn only from the 5,798 fitting images. Known candidate class identities were used for synthesis as predeclared; real validation, excluded and official test images supplied no donors.

All **7,747 final verification checks passed**, including unchanged sources/references/cached inputs, exact regenerated primary/donor/target exposure schedules, finite selected weights, selected checkpoint/logit hashes, metric reproduction and earliest-maximum selection. The 17 setup tests and actual-GPU stochastic replay/checkpoint-recovery probe passed before training. No worker restart occurred.

These checks establish execution of the declared recipe; they do not explain why it generalized poorly or establish that synthesized features resemble real unseen images. This is one seed and one complete recipe combining feature synthesis, auxiliary supervision, target balancing and extra donor exposure. It does not rule out every feature-composition method.

Recommendation: keep the saved references and close this recipe at development. It has not earned a full-data refit on these results. The run stopped as planned, with no official-test evaluation or full-data training queued.

Saved [results](runs/development_seed42/results.md), [full metrics](runs/development_seed42/results.json), [verification](runs/development_seed42/verification.json), [selection](runs/development_seed42/selection_frozen.json), [history](runs/development_seed42/finetune/history.json), and [fixed protocol](PLAN.md). The selected checkpoint SHA256 is `a20aa3d31900b1231773b8c672d43a5acc95f9a4e9348b786cedff59cf7aa66f`.
