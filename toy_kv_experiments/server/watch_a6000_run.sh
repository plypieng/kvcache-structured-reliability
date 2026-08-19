#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${SSH_TARGET:-plypieng@100.114.7.72}"
PROJECT_DIR="${PROJECT_DIR:-/home/plypieng/kvcache}"
INTERVAL="${INTERVAL:-10}"
LOG_FILE="${LOG_FILE:-}"

project_dir_q="$(printf "%q" "$PROJECT_DIR")"
log_file_q="$(printf "%q" "$LOG_FILE")"
interval_q="$(printf "%q" "$INTERVAL")"

ssh "$SSH_TARGET" \
  "PROJECT_DIR=$project_dir_q LOG_FILE=$log_file_q INTERVAL=$interval_q bash -s" <<'REMOTE_MONITOR'
set -euo pipefail
cd "$PROJECT_DIR"

if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$(
    ls -t \
      toy_kv_experiments/logs/kivi_block_sweep_*.nohup.log \
      toy_kv_experiments/logs/kivi_real_blockwise_sweep_*.nohup.log \
      toy_kv_experiments/logs/structure_fidelity_sweep_*.nohup.log \
      toy_kv_experiments/logs/kivi_residual_sweep_*.nohup.log \
      toy_kv_experiments/logs/kivi_reference_json100_*.nohup.log \
      toy_kv_experiments/logs/validity_gate_corrected_*.nohup.log \
      toy_kv_experiments/logs/post_attention_gate_v3_*.nohup.log \
      toy_kv_experiments/logs/post_attention_stage_a_v3_*.nohup.log \
      toy_kv_experiments/logs/poster_pilot_allformats100_*.nohup.log \
      toy_kv_experiments/logs/structeval_full_official_*.nohup.log \
      toy_kv_experiments/logs/validity_gate_json50_*.log \
      2>/dev/null | head -1 || true
  )"
fi

while true; do
  printf "\n%s\n" "============================================================"
  echo "A6000 StructEval monitor"
  echo "time: $(date)"
  echo

  echo "GPU"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
  echo

  echo "Active experiment processes"
  active_processes="$(pgrep -af "run_kivi_block|run_structure_token_fidelity|run_kivi_residual|run_validity_gate_json50|run_post_attention|run_poster_pilot|run_structeval_full|run_qwen_structeval|pretrained_kv_quantization" || true)"
  if [ -n "$active_processes" ]; then
    echo "$active_processes"
    echo
    echo "Experiment status: active"
  else
    echo "No matching experiment process"
    echo
    echo "Experiment status: no active run detected; showing latest completed or latest known log below"
  fi
  echo

  echo "Selected sweep log"
  if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
    echo "$LOG_FILE"
    tail -35 "$LOG_FILE"
    if grep -qE "\\[(kivi-block|structure-fidelity|residual)\\] complete|\\[run\\] saved result" "$LOG_FILE"; then
      echo
      echo "Latest selected log contains a completion marker."
    fi
  else
    echo "No sweep log found"
  fi
  echo

  echo "Latest atomic checkpoint status"
  latest_status="$(find toy_kv_experiments/results -type f -path '*.checkpoints/status.json' -exec ls -t {} + 2>/dev/null | head -1 || true)"
  if [ -n "$latest_status" ]; then
    echo "$latest_status"
    cat "$latest_status"
  else
    echo "No checkpoint status found"
  fi
  echo

  echo "Latest run logs"
  ls -lt \
    toy_kv_experiments/logs/kivi_block_limit*.log \
    toy_kv_experiments/logs/kivi_real_blockwise_limit*.log \
    toy_kv_experiments/logs/structure_fidelity_limit*.log \
    toy_kv_experiments/logs/kivi_residual_limit*.log \
    toy_kv_experiments/logs/kivi_reference_json100_*.nohup.log \
    toy_kv_experiments/logs/validity_gate_corrected_*.nohup.log \
    toy_kv_experiments/logs/post_attention_gate_v3_*.nohup.log \
    toy_kv_experiments/logs/post_attention_stage_a_v3_*.nohup.log \
    toy_kv_experiments/logs/poster_pilot_allformats100_*.nohup.log \
    toy_kv_experiments/logs/structeval_full_official_*.nohup.log \
    toy_kv_experiments/logs/validity_gate_json50_*.log \
    2>/dev/null | head -8 || true
  echo

  echo "Latest result files"
  ls -lhtr \
    toy_kv_experiments/results/*kivi_block*.json \
    toy_kv_experiments/results/*kivi_real_blockwise*.json \
    toy_kv_experiments/results/*structure_fidelity*.json \
    toy_kv_experiments/results/*kivi_residual*.json \
    toy_kv_experiments/results/*cache_reference*.json \
    toy_kv_experiments/results/*mode_real_blockwise*.json \
    toy_kv_experiments/results/post_attention_gate/*.json \
    toy_kv_experiments/results/post_attention_stage_a/*.json \
    toy_kv_experiments/results/poster_pilot_allformats100/*.json \
    toy_kv_experiments/results/structeval_full_official/*.json \
    2>/dev/null | tail -10 || true
  echo

  echo "Refresh interval: ${INTERVAL}s. Press Ctrl-C to stop."
  sleep "$INTERVAL"
done
REMOTE_MONITOR
