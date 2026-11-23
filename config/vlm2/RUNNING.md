# Running vlm2, and how the best model is chosen

This is the operating manual for the controlled ablation designed in [README.md](README.md): how to run each stage, how to read the results, and the fixed rules that lead from the results to one final model. Follow the stages in order. Each stage's configs depend on the previous stage's results.

```
Stage 0  s0_base, s0_base_r2             base + noise floor
Stage 1  s1_*  (leave-one-out)           what LoRA / CSP / descriptions / clean names contribute
Stage 2  s2_*  (add-one)                 which improvements are real on top of the base
            │  report: docs/vlm/vlm2_report.md
            ▼
Stage 3  s3_*  (pairs of winners)        complementary or overlapping?
            ▼
Stage 4  s4_final, s4_final_r2           the combination, refit + test once  ──►  best model: runs/vlm2/s4_final
```

## 0. Rules that hold everywhere

- **All of vlm2 runs on one machine, either this laptop or a Colab A100, never both.** The comparisons are paired against `s0_base`, and a 4050 and an A100 do not give bit-identical results.
- **Don't edit a config once its run has started.** To try something else, add a new config.
- **One training job at a time on the laptop.** Close other GPU apps (Blender, Bambu Studio) first. A full GPU spills into shared memory and runs about 8× slower.
- **Within a run, the epoch and γ are chosen by validation HM.** Between runs, decisions use **validation AUSUC** from the report. Test numbers are read only at stage 4.

## 1. Where to run

### Option A: this laptop (RTX 4050, about 40 h for all stages)

From the project root in PowerShell, start the stage 0–2 queue detached. It survives closing the terminal and keeps the machine awake:

```powershell
Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','runs\queue_vlm2.ps1' -WindowStyle Hidden
```

Check progress:

```powershell
Get-Content runs\queue_status.txt -Tail 5
```

The same queue also runs from Git Bash, in the foreground, with any list of run names:

```powershell
bash config/vlm2/run_queue.sh
```

### Option B: Colab A100 (about 8–12 h for all stages)

1. **Push the vlm2 code to GitHub first.** It is currently uncommitted on this laptop, and Colab clones from GitHub.
2. **Put the dataset on Drive.** Zip the `DPD_Small` folder (`DPD_Small/DPD_Small_dataset/` plus the three CSVs) and upload it to `MyDrive/dpd/DPD_Small.zip`.
3. Open a new notebook, choose **Runtime → Change runtime type → A100 GPU**, and run these cells in order:

```python
# 1. Drive (dataset in, results out)
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
# 2. Code: a private repo needs a GitHub token (store it in Colab secrets, never in the notebook)
cd /content
git clone https://github.com/Suerfine/Out-Of-Distribution-Plant-Disease-Identification.git repo
cd repo && git checkout <branch-with-vlm2>   # the branch you pushed in step 1
pip install -q open_clip_torch timm          # Colab already has CUDA torch/torchvision/pandas/pillow
```

```bash
%%bash
# 3. Dataset on the local SSD (reading images from Drive would be the bottleneck)
cd /content/repo
cp /content/drive/MyDrive/dpd/DPD_Small.zip /content/ && unzip -q /content/DPD_Small.zip -d /content/repo/
ls DPD_Small   # expect DPD_Small_dataset/ and DPD_Small_{train,test_seen,test_unseen}.csv
# 4. Results on Drive, so a disconnect loses nothing
mkdir -p /content/drive/MyDrive/dpd/runs_vlm2 runs
ln -sfn /content/drive/MyDrive/dpd/runs_vlm2 runs/vlm2
```

```bash
%%bash
# 5. Start the queue in the background (stages 0-2); it skips finished runs, so re-running this cell resumes
cd /content/repo
PYTHON=python nohup bash config/vlm2/run_queue.sh > runs/vlm2/queue_stdout.txt 2>&1 &
echo started
```

```bash
%%bash
# 6. Progress (re-run any time)
tail -n 5 /content/repo/runs/vlm2/queue_status.txt
```

After a disconnect, run cells 1–4 again, then cell 5. A run that was cut off midway starts again from scratch. Finished runs stay on Drive and are skipped. The resize cache (`runs/cache/resized_256`) is rebuilt automatically on the first run of each session, which takes a few minutes. Keep an eye on compute units: an A100 uses them quickly.

## 2. Stage 0–2: run, then read the report

The queue runs these 17 configs, most important first:

| Order | Runs | Notes |
|---|---|---|
| 1 | `s0_base` | Full run (refit + test). Every other run is compared with it, and `s2_G1_gan` uses its adapters |
| 2 | `s0_base_r2` | Same splits, new initialisation; gives the noise floor |
| 3 | `s1_no_lora`, `s1_no_csp` | Leave-one-out of the two foundations |
| 4 | `s2_T1_coop`, `s2_I1_ssl`, `s2_C1_primitive_branches`, `s2_G1_gan` | One per group, the strongest prior candidates |
| 5 | `s2_C3_drpt`, `s2_I3_lora_all_blocks`, `s2_T3_text_anchor`, `s2_I2_image_consistency`, `s2_T2_multi_description` | |
| 6 | `s1_no_descriptions`, `s1_raw_names`, `s2_C4_within_crop`, `s2_I4_ema` | |

`s2_C2_posthoc_fusion` needs no training. The report scores it from `s0_base`'s saved validation scores.

The report regenerates after every run. To rebuild it by hand:

```powershell
.venv\Scripts\python -m train.vlm2_report
```

On Colab, use `python -m train.vlm2_report`.

Output goes to `docs/vlm/vlm2_report.md`, with the full numbers in `runs/vlm2/vlm2_report.csv`. For every run it shows Δ validation AUSUC against `s0_base` with a paired-bootstrap 95% interval, Δ closed-set seen and unseen Top-1, Δ HM, and a verdict.

**Verdict rule** (fixed before the runs): a change is **real** when its AUSUC interval excludes 0 **and** |Δ AUSUC| ≥ max(1.5 pts, 2 × noise floor). The noise floor is |`s0_base_r2` − `s0_base`|, printed at the top of the report.

**Before going on, check the noise floor.** If it comes out above 2 pts, the base itself is unstable and a second replicate is worth running first. To make one, copy `s0_base_r2_config.json`, set `train_seed_offset` to 200, and give it a new name.

## 3. Stage 3: test the winners in pairs

1. **Pick the group winners.** In each group (T text, I image, C composition, G classifier), the winner is the real-positive change with the largest Δ AUSUC. On a tie, prefer the one that doesn't lower closed-set seen Top-1. A group with no real-positive change has no winner.
2. **Choose the pairs to run:**
   - every **cross-group** pair of winners (e.g. T1+I1, T1+C1, I1+C1);
   - per group with two or more real-positive changes, **one within-group pair**: the top two, to confirm they overlap;
   - if G1 is real-positive, pair it with the I winner, because the GAN is expected to overlap with image-side changes.
3. **Write each config with the combine helper.** It merges the single-factor changes onto `s0_base` and stops on conflicting keys:

   ```powershell
   .venv\Scripts\python -m train.vlm2_combine T1 I1
   ```
   ```powershell
   .venv\Scripts\python -m train.vlm2_combine T1 C1
   ```
   ```powershell
   .venv\Scripts\python -m train.vlm2_combine I1 C1
   ```

   These write `config/vlm2/s3_T1_I1_config.json` and so on. A pair with G1 also writes a full LoRA run, `<name>_lora`, which has to run before the G1 config.
4. **Run them**, by name:

   ```powershell
   bash config/vlm2/run_queue.sh s3_T1_I1 s3_T1_C1 s3_I1_C1
   ```

   On Colab, use cell 5 with the names added after `run_queue.sh`.
5. The report gains an **Interactions** table. For each pair it shows I = Δ(A+B) − ΔA − ΔB with a 95% interval, labelled:
   - **complementary**: I's interval contains 0, so the gains add;
   - **overlapping**: the pair is no better than the better part alone (I < 0);
   - **antagonistic**: the pair is worse than the better part, beyond noise;
   - **synergistic**: more than additive.

## 4. Stage 4: build and confirm the final model

**The final combination, by fixed rules:**

1. Start from the winner with the largest Δ AUSUC.
2. Add each other winner, in order of Δ AUSUC, if its pair with every factor already chosen was **complementary** or **synergistic**.
3. If a pair was **overlapping** or **antagonistic**, keep only the better of the two.
4. Untested combinations of three or more are allowed only if every pair inside them was tested and complementary.
5. If no stage 2 change is real-positive, the final model is `s0_base`, and stage 4 is skipped.

Write and run it:

```powershell
.venv\Scripts\python -m train.vlm2_combine T1 I1 C1 --final
```

```powershell
bash config/vlm2/run_queue.sh s4_final s4_final_r2
```

This writes `s4_final` and its replicate `s4_final_r2` (training seeds + 100). Both are full runs: validation, refit on all training images, and one test evaluation.

**Confirming:**
- The report shows both runs' validation Δ against `s0_base`.
- The final model is confirmed when **both** clear the decision threshold. That guards against having picked a lucky configuration out of many.
- Then, and only then, read test, from each run's `metrics.json`:
  - `harmonic_mean_top1`: the protocol HM at the validation γ;
  - `harmonic_mean_top1_gamma0`;
  - `test_gamma_free`: AUSUC and closed-set seen/unseen.
- Compare these with `s0_base`'s `metrics.json`.
- If `s4_final_r2` does **not** clear the threshold, the combination is not confirmed. Report it as such and fall back to the best single stage 2 winner, which then needs its own full run:

  ```powershell
  .venv\Scripts\python -m train.vlm2_combine T1 --final --name s4_single
  ```

**The best model is `runs/vlm2/s4_final`.** It is the configuration chosen above before test was read. `s4_final_r2` is the evidence that the choice reproduces. Never swap in `s4_final_r2` because its test score is higher: that would be selecting on test.

## 5. After the final model

1. **Log it** in `docs/all_results.csv`. This is required by AGENTS.md, and adds a γ=0 row and a calibrated row:

   ```powershell
   .venv\Scripts\python -m train.log_results runs/vlm2/s4_final
   ```

   Do the same for `runs/vlm2/s0_base` and `runs/vlm2/s4_final_r2`.
2. **Write `_result`** in each stage 4 config. Include the validation Δ with its interval, the test numbers, and whether the replicate confirmed it.
3. **Use it in the app.** `list_runs` picks up full runs under `runs/vlm2/` automatically. Validation-only runs write `val_metrics.json`, so the app ignores them. Start the app and pick `s4_final` on the inference page:

   ```powershell
   .venv\Scripts\streamlit run streamlit_app.py
   ```

4. **Optional: CNN ensemble.** The ConvNeXt ensemble was the largest single test gain so far. To add it to the final VLM:
   - copy `config/vlm/v6.5_convnext_v5.13_ensemble_config.json` to a new version;
   - set `members[0].run` to `runs/vlm2/s4_final`, and update `version` and `output_dir`;
   - run it:

     ```powershell
     .venv\Scripts\python -m train.ensemble --config config\vlm\<new>_config.json
     ```

   The ensemble picks its weight and γ on the seed-42 validation split, so treat it as an add-on after the VLM is fixed, not as part of the vlm2 comparison.

## 6. What each run leaves behind

| File | Written by | Contents |
|---|---|---|
| `runs/vlm2/<run>/val_metrics.json` | validation-only runs | selected epoch, γ, validation HM, `val_gamma_free` (AUSUC, closed seen, closed unseen; mean and per split) |
| `runs/vlm2/<run>/metrics.json` | full runs | everything above, plus test metrics at the validation γ, `harmonic_mean_top1_gamma0` and `test_gamma_free` |
| `runs/vlm2/<run>/val_scores_split{1,2,3}.npz` | all runs | raw validation scores per epoch (pair, crop, disease), used by the report |
| `runs/vlm2/<run>/val_gamma_free_grid.npz` | all runs | AUSUC and closed-set Top-1 per split and epoch |
| `runs/vlm2/<run>/history.csv`, `train.log` | all runs | per-epoch log |
| `runs/vlm2/<run>/lora_image_encoder.pt`, `prompt_vectors.pt`, `scores.npz` | full runs | the model (loaded by `dpd/inference.py`) and its test scores |
| `docs/vlm/vlm2_report.md`, `runs/vlm2/vlm2_report.csv` | `train/vlm2_report.py` | the decision table and interactions |
| `runs/queue_status.txt` (PowerShell queue), `runs/vlm2/queue_status.txt` (bash queue) | queues | start, done and FAILED lines with durations |

## 7. Troubleshooting

- **A run is FAILED in the status file.** Read the end of `runs/vlm2/<run>/train.log`.
- **Out of GPU memory, or about 8× slower than expected.** Another app holds GPU memory. Close it, delete the run's folder, and restart the queue; finished runs are skipped.
- **Windows error 1455, "paging file is too small".** Two jobs ran at once. Only one training job at a time.
- **Exit code 139 at the very end.** This is the known Windows shutdown segfault after torch. Outputs are already written, and the queue checks for them, not the exit code.
- **`NativeCommandError` lines in a PowerShell log.** These are stderr warnings, not failures.
- **The report fails with "validation images differ".** A run used other validation settings. Every vlm2 config must keep `s0_base`'s `validation` block apart from `only`, `save_scores` and `train_seed_offset`.
