#!/usr/bin/env bash
# Run vlm2 configs one after another (local Git Bash, Linux or Colab). Finished runs are skipped, so the queue can be
# restarted after a disconnect. The report (train/vlm2_report.py) is regenerated after every run.
#
#   bash config/vlm2/run_queue.sh                      # stages 0-2, priority order
#   bash config/vlm2/run_queue.sh s3_T1_C1 s3_I1_C1    # any configs by name (config/vlm2/<name>_config.json)
#
# PYTHON overrides the interpreter (default: .venv's python if present, else python).
set -u
cd "$(dirname "$0")/../.."

if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -e .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -e .venv/bin/python ]; then PY=.venv/bin/python
else PY=python; fi

STAGE_0_2=(
  s0_base s0_base_r2 s1_no_lora s1_no_csp
  s2_T1_coop s2_I1_ssl s2_C1_primitive_branches s2_G1_gan
  s2_C3_drpt s2_I3_lora_all_blocks s2_T3_text_anchor s2_I2_image_consistency
  s2_T2_multi_description s1_no_descriptions s1_raw_names s2_C4_within_crop s2_I4_ema
)
if [ $# -gt 0 ]; then RUNS=("$@"); else RUNS=("${STAGE_0_2[@]}"); fi

mkdir -p runs/vlm2
STATUS=runs/vlm2/queue_status.txt
log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*" | tee -a "$STATUS"; }

for r in "${RUNS[@]}"; do
  cfg=config/vlm2/${r}_config.json
  dir=runs/vlm2/$r
  if [ ! -f "$cfg" ]; then log "MISSING $cfg"; continue; fi
  if [ -f "$dir/metrics.json" ] || [ -f "$dir/val_metrics.json" ]; then log "skip $r (already done)"; continue; fi
  method=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['method'])" "$cfg")
  case "$method" in
    clip_prompt_lora) module=train.train_lora ;;
    clip_feature_gen) module=train.train_feature_gen ;;
    *) log "UNKNOWN method $method in $cfg"; continue ;;
  esac
  mkdir -p "$dir"
  start=$(date +%s)
  log "start $r ($module)"
  "$PY" -m "$module" --config "$cfg" > "$dir/train.log" 2>&1
  mins=$(( ($(date +%s) - start) / 60 ))
  if [ -f "$dir/metrics.json" ] || [ -f "$dir/val_metrics.json" ]; then log "done $r after $mins min"
  else log "FAILED $r after $mins min (see $dir/train.log)"; fi
  "$PY" -m train.vlm2_report > runs/vlm2/report.log 2>&1 || log "report not updated (see runs/vlm2/report.log)"
done
log "queue finished"
