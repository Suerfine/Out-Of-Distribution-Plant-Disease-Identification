# Current plan: finalize the three completed R0 baselines

Prepared at the user's request on 28 October 2026. The user authorized training
on 29 October; all six refits and official evaluations completed at 04:06 MYT.
The artifacts are independently verified and the completion monitor is stopped.
See the [verified final report](../runs/baseline/independent_cnn_baseline_r0_finalization_seed42/results.md).
This plan replaces the remaining LR/WD/dropout sweep for the baseline deliverable.
The original study, frozen split, R0 checkpoints and unfinished BiT R1 recovery
files remain available as historical artifacts.

## Frozen sources

Use the three completed R0 recipe pairs, with no additional selection training:

| Backbone | Refit crop epochs | Refit disease epochs | Validation-selected gamma |
|---|---:|---:|---:|
| BiT ResNet-v2-50x1 IN21K→IN1K | 3 | 3 | 6.6 |
| ConvNeXt-Base IN22K→IN1K | 15 | 15 | 6.2 |
| EfficientNetV2-L IN21K→IN1K | 3 | 3 | 6.6 |

The pinned source manifest is `configs/refit_r0_seed42.json`. It records the
original recipe hash and each R0 completion-record hash. Preparation also checks
the completed target weights, cached original pretrained weights, validation
partition and dataset identity. The partial R1 trial is excluded from selection.

## Approved work (completed)

1. Freshly initialize each independent crop/disease network from its original
   cached pretrained weights, seed 42, and refit on all 8,000 training images.
2. Keep the R0 preprocessing, augmentation, optimizer, LR/WD/dropout, AMP,
   normalization handling and physical/effective batch exactly as saved.
   Keep the original LR-versus-epoch schedule with its 30-epoch cosine horizon;
   run only the frozen 3/15/3 epochs per target. No validation search occurs.
3. Finish all six target refits before evaluating any official test split.
4. Evaluate all 333 composition candidates on 2,000 seen and 1,448 unseen images.
   Report gamma zero and the frozen validation-selected gamma, composition
   Top-1/Top-5 and harmonic means, plus factor diagnostics. Keep unseen-only
   candidate diagnostics separately labeled.
5. Verify prediction counts, finite logits/metrics, seen confusion matrix,
   checkpoint hashes and shared-app score parity before marking completion.

All six target refits and 42 target epochs are complete. Actual target training
was 3h 31m 02s; finalization including evaluation/checks took 3h 57m 23s.
The ten-hour invocation cap remains a recovery boundary, not a promised runtime.
This compares three fixed R0 training recipes rather than exhaustively optimized
architectures. Official test results must not be used for further selection.

## Commands from the repository root

Preparation only, with no model training:

```powershell
.\independent_cnn_baseline\run_overnight.ps1 -RefitOnly -Prepare
```

Original launch command (the approved run is complete):

```powershell
.\independent_cnn_baseline\run_overnight.ps1 -RefitOnly -Detach
```

Resume this smaller run after confirming its previous controller exited:

```powershell
.\independent_cnn_baseline\run_overnight.ps1 -RefitOnly -Detach -Resume
```

For a graceful pause of an active refit-only controller:

```powershell
& ..\Plant_Disease\.venv\Scripts\python.exe -B -m independent_cnn_baseline.finalize --request-stop
```

The active target epoch finishes and checkpoints, then the controller records
`paused`. A later explicit `-Resume` consumes the stop request. Full optimizer,
AMP scaler and random-state recovery are preserved for unfinished targets.

Both the original study lock and the refit-only lock are held during a run to
prevent conflicting controllers. No permanent power settings are changed.

## Files

Preparation, frozen selections, status and finalization log:

`runs/baseline/independent_cnn_baseline_r0_finalization_seed42/`

Final app-compatible model folders:

- `runs/baseline/independent_cnn_baseline_bit50_seed42/`
- `runs/baseline/independent_cnn_baseline_convnext_base_seed42/`
- `runs/baseline/independent_cnn_baseline_efficientnetv2_l_seed42/`

Each final model keeps its own weights, final refit histories and verification
record. The refit-only study's `results.md` and `study_summary.json` appear only
after all three official evaluations complete. Existing validation learning
curves remain linked through each chosen source and copied into the final run.

The original tuning study remains paused. The monitor now watches only this
refit-only run. No training or automatic restart is authorized by preparation alone.
