# Audit: baseline identity, calibration and the negative SSL result

Post-hoc audit of existing configurations, saved logits and verified reports.
No new training, gamma search, model selection or change to primary results.

## The uncalibrated baseline already exists

The saved Base checkpoint supports both gamma0 and its existing gamma6.2.
These are scoring settings of the same crop/disease weights, not separate
training runs. Gamma subtracts from the323 training-seen pair columns; every
image is still scored against all333 candidates. No image's true test membership
or target label is supplied to that ranking adjustment.

| Scoring / model | Seen Top-1 | Unseen Top-1 | H |
|---|---:|---:|---:|
| Gamma0, saved Base | 73.80% | 14.02% | 23.56% |
| Gamma0, SSL Base | 72.60% | 13.05% | 22.13% |
| Fixed gamma6.2, saved Base | 67.00% | 76.52% | 71.44% |
| Same fixed gamma6.2, SSL Base | 66.10% | 74.65% | 70.12% |

Existing calibration was retained to compare the same decision policy and avoid
new test-based tuning. However, gamma6.2 was selected with the original Base
validation procedure, not optimized for the new SSL model. A fixed-gamma result
alone cannot isolate representation quality or establish which model has the
best achievable seen/unseen trade-off. Raw results are needed too. The raw SSL
result was also worse, so calibration alone does not explain this negative run.

## Direct disease-head diagnosis, without pair calibration

Using the saved175-way disease logits and aligned original test labels:

| Raw disease classifier | Seen Top-1 | Unseen Top-1 | Unseen Top-5 |
|---|---:|---:|---:|
| Saved Base | 74.25% | 14.50% | 33.84% |
| SSL Base | 72.80% | 11.26% | 33.36% |

The crop model is exactly reused: raw crop Top-1 remains94.30% seen /85.64%
unseen. On raw pair predictions SSL fixes14 baseline unseen mistakes but breaks
28 previously correct cases. Under fixed gamma6.2 it fixes80 but breaks107.
Thus the SSL run changes useful decisions, but regresses more than it improves.

The median disease-logit standard deviation changes only from.837 to.842 on
unseen images. That descriptive statistic supplies no evidence of a large
global score-scale change; it does not rule out class-specific confidence shifts.

## Aggregate accuracy conceals different composition outcomes

With the same fixed gamma6.2, SSL improves these unseen correct counts:

| Composition | Base correct | SSL correct | Total |
|---|---:|---:|---:|
| Cotton bacterial blight | 101 | 107 | 125 |
| Mango anthracnose | 90 | 95 | 95 |
| Sugarcane rust | 77 | 80 | 89 |
| Tomato early blight | 310 | 326 | 472 |
| Grape black rot | 204 | 162 | 271 |

The four gains total30 extra correct images. Grape black rot alone loses42;
losses on other compositions bring the net unseen difference to-27. Under
gamma0, pepper bacterial spot is the largest count regression(151→139).
These are post-hoc diagnostics of familiar official tests, not validation data
for selecting another method or tuning a loss/gamma.

## Experimental-design issues that limit conclusions

1. Earlier RL, weight-editing, contrastive and shared-composer experiments used
   the separate Tiny project. Selecting that reference for the intended Base
   project was an agent error. They do not establish those methods' outcomes
   on Base. Only the two recent complete Base configurations are directly
   relevant here; this is not a broad sweep of Base methods.
2. The saved Base package originally selected its recipe/epoch and gamma using
   familiar and pseudo-unseen validation before refitting on all8,000 images.
   Its selection manifest specifies20 held-out training pairs. Earlier claims
   that the baseline lacked pseudo-unseen validation described the separate
   Tiny protocol and must not be carried over to this Base package.
3. The recent Base trials used fixed full-data budgets without fresh validation.
   Fifteen supervised epochs match the Base refit length, but do not establish
   that15 is best for an SSL-adapted initialization. Twenty SSL epochs and its
   conservative learning rates were a practical first configuration, not
   validated optima. These comparisons test specific recipes, not method limits.
4. SimSiam's view-consistency objective does not directly require disease
   discrimination. The fitting features did not collapse and the loss improved,
   yet the final disease-head accuracy decreased. The evidence does not identify
   one cause, such as crop/background cues, forgetting or insufficient duration.
5. These are single-seed results. Whole-composition bootstrap difference intervals
   include zero and do not quantify retraining-seed variability. Numerical/hash
   checks validate execution and metric reconstruction; they do not certify
   scientific usefulness or prove every modeling choice is appropriate.

There is no evidence that the baseline is an unbeatable ceiling. There is also
no confirmed training bug explaining all negative results. The ranking-export
API error was repaired without changing model weights or predictions' intended
scoring rule. Before another long run, use these errors to define a specific
hypothesis and agree on a development/selection procedure. Do not silently
introduce new validation splits, retune on official tests or retrain controls.

Machine-readable diagnostic:
[saved-logit audit](simsiam_base_v1/runs/overnight_seed42/posthoc_method_audit.json).
Original selection provenance:
[Base selected source](../Out-Of-Distribution-Plant-Disease-Identification/runs/baseline/independent_cnn_baseline_convnext_base_seed42/selected_source.json).
