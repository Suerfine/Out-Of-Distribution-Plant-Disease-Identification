# Crop-conditioned disease recognition

One fully trainable disease CNN, initialized from the saved disease baseline,
with shared crop-conditioned feature adjustments. The original crop CNN stays
frozen. Original manifests and disease accuracy-first validation rule; no
pseudo-unseen validation or new control training. See [fixed protocol](PLAN.md).

Prediction considers every crop and all 333 candidate pairs. No true crop labels
are supplied at inference. This is a conditional-attribute-inspired adaptation,
not an exact paper reproduction: https://arxiv.org/abs/2305.17940.

Run `../../Plant_Disease/.venv/Scripts/python.exe train.py --preflight-only`
for checks, then `start.ps1` for the full resumable background job. Status,
15-minute snapshots and final reports appear under `runs/full_seed42`.
These logs do not create chat notifications.
