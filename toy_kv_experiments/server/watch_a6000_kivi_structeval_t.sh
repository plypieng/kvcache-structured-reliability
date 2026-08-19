#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET="${SSH_TARGET:-plypieng@192.168.10.113}"
INTERVAL="${INTERVAL:-30}"
FOLLOW="${FOLLOW:-0}"
REMOTE_WATCH="${REMOTE_WATCH:-/home/plypieng/kvcache/toy_kv_experiments/server/watch_official_kivi_structeval_t.sh}"

show_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_TARGET" \
    "bash '$REMOTE_WATCH'"
}

if [[ "$FOLLOW" == "1" ]]; then
  while true; do
    clear
    date
    show_status
    sleep "$INTERVAL"
  done
else
  show_status
fi
