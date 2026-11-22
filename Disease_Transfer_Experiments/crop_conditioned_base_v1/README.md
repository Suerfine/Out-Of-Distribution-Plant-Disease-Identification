# Current experiment: ConvNeXt-Base with crop-conditioned disease recognition

This replaces only the current Tiny run, as requested on 6 November 2026.
The earlier Tiny runs remain preserved. Reuse the saved full-data Base disease
weights and frozen Base crop model; train only the modified disease network.

The saved Base checkpoint already saw all8,000 training images. Run a fixed15
additional epochs on that same training set, with no new validation selection
or pseudo-unseen partition. Keep Base preprocessing/regularization and report
both gamma0 and the baseline's existing fixed gamma6.2. No new calibration.
See the [fixed protocol](PLAN.md).

`start.ps1` launches the hidden resumable training process. Its status,
15-minute progress logs and automatic final results appear in `runs/full_seed42`.

## Operational recovery, 6 November

Training initially stopped during epoch8, then resumed from epoch7. A later
serialization I/O failure during the epoch9 save left the valid epoch8 checkpoint
intact. The failed temporary file remains preserved. The model, training recipe,
fixed15-epoch budget, data and frozen experiment sources are unchanged.

`start_recovery.ps1` resumes through the separately logged operational helper
`../operations/checkpoint_recovery.py`. It writes through an explicit file
stream, flushes and fsyncs, checks ZIP CRCs before atomic replacement, and retries
storage failures. A complete checkpoint write/read probe must pass first. The
helper's own source hash and I/O events are recorded alongside the original
experiment fingerprint; this is an I/O repair, not a new model configuration.
