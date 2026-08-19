from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "presentations" / "a6000_real_cache_status.json"


REMOTE_STATUS_SCRIPT = r"""
set -euo pipefail
cd /home/plypieng/kvcache
echo "__SECTION__date"
date '+%Y-%m-%d %H:%M:%S %Z'
echo "__SECTION__gpu"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits || true
echo "__SECTION__processes"
pgrep -af 'run_real_cache_precision_boundary|run_kivi_block|run_overnight_reliability_allocation|run_validity_gate_json50|run_post_attention|run_poster_pilot|run_structeval_full|pretrained_kv_quantization' || true
echo "__SECTION__queue_log"
latest_queue="$(ls -t toy_kv_experiments/logs/overnight_queue_nohup_*.log toy_kv_experiments/logs/real_cache*_queue_*.nohup.log toy_kv_experiments/logs/real_cache_*rerun_*.nohup.log toy_kv_experiments/logs/kivi_reference_json100_*.nohup.log toy_kv_experiments/logs/validity_gate_corrected_*.nohup.log toy_kv_experiments/logs/post_attention_gate_v3_*.nohup.log toy_kv_experiments/logs/post_attention_stage_a_v3_*.nohup.log toy_kv_experiments/logs/poster_pilot_allformats100_*.nohup.log toy_kv_experiments/logs/structeval_full_official_*.nohup.log 2>/dev/null | head -1 || true)"
echo "${latest_queue:-}"
if [ -n "${latest_queue:-}" ]; then
  tail -80 "$latest_queue" || true
fi
echo "__SECTION__checkpoint"
latest_checkpoint="$(find toy_kv_experiments/results -type f -path '*.checkpoints/status.json' -exec ls -t {} + 2>/dev/null | head -1 || true)"
echo "${latest_checkpoint:-}"
if [ -n "${latest_checkpoint:-}" ]; then
  cat "$latest_checkpoint" || true
fi
echo "__SECTION__summary_log"
latest_summary="$(ls -t toy_kv_experiments/logs/overnight_reliability_allocation_*.log toy_kv_experiments/logs/real_cache_precision_boundary_*.log toy_kv_experiments/logs/validity_gate_json50_*.log 2>/dev/null | head -1 || true)"
echo "${latest_summary:-}"
if [ -n "${latest_summary:-}" ]; then
  tail -120 "$latest_summary" || true
fi
echo "__SECTION__results"
ls -lt toy_kv_experiments/results/*limit100*.json toy_kv_experiments/results/*limit50*.json toy_kv_experiments/results/*kivi_real_blockwise*limit20*.json toy_kv_experiments/results/*mode_real_blockwise*.json toy_kv_experiments/results/poster_pilot_allformats100/*.json toy_kv_experiments/results/structeval_full_official/*.json 2>/dev/null | head -30 || true
echo "__SECTION__summary"
if ls toy_kv_experiments/results/*limit100*.json toy_kv_experiments/results/*limit50*.json toy_kv_experiments/results/*kivi_real_blockwise*limit20*.json toy_kv_experiments/results/*mode_real_blockwise*.json >/dev/null 2>&1; then
  source /home/plypieng/miniforge3/etc/profile.d/conda.sh
  conda activate kvcache-py311
  python -m toy_kv_experiments.summarize_structeval_results --latest-by-config toy_kv_experiments/results/*limit100*.json toy_kv_experiments/results/*limit50*.json toy_kv_experiments/results/*kivi_real_blockwise*limit20*.json toy_kv_experiments/results/*mode_real_blockwise*.json || true
fi
"""


def run_ssh(host: str, command: str, timeout: int) -> tuple[int, str, str]:
    password = os.environ.get("A6000_PASSWORD", "")
    if password:
        encoded_command = base64.b64encode(command.encode("utf-8")).decode("ascii")
        expect_program = f"""
set timeout {timeout}
spawn ssh -o StrictHostKeyChecking=accept-new {host} "echo {encoded_command} | base64 -d | bash"
expect {{
    -re "(?i)are you sure.*" {{ send "yes\\r"; exp_continue }}
    -re "(?i)password:" {{ send "$env(A6000_PASSWORD)\\r"; exp_continue }}
    eof
}}
"""
        proc = subprocess.run(
            ["expect", "-c", expect_program],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout + 10,
        )
        stdout = proc.stdout
        marker = "__SECTION__"
        if marker in stdout:
            stdout = stdout[stdout.index(marker) :]
        return proc.returncode, stdout, proc.stderr

    proc = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new", host, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "preamble"
    sections[current] = []
    for line in text.splitlines():
        if line.startswith("__SECTION__"):
            current = line.replace("__SECTION__", "", 1).strip()
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def parse_gpu(raw: str) -> dict[str, Any]:
    first = raw.splitlines()[0] if raw.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 5:
        return {}
    return {
        "name": parts[0],
        "memory_used_mb": _to_number(parts[1]),
        "memory_total_mb": _to_number(parts[2]),
        "utilization_percent": _to_number(parts[3]),
        "temperature_c": _to_number(parts[4]),
    }


def _to_number(text: str) -> int | float | str:
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


def detect_current_case(processes: str, logs: str) -> str:
    text = processes + "\n" + logs
    command_match = re.search(r"--bits\s+(\d+)(?:.*?--key-bits\s+(\d+))?(?:.*?--value-bits\s+(\d+))?", text)
    if command_match:
        bits = command_match.group(1)
        key_bits = command_match.group(2) or bits
        value_bits = command_match.group(3) or bits
        return f"K{key_bits}/V{value_bits}"
    run_match = re.search(r"\[real-cache\] running ([^\n]+)", logs)
    if run_match:
        return run_match.group(1).strip()
    for label in ("K8/V8", "K6/V6", "K4/V8", "K8/V4", "K4/V4"):
        if label in logs:
            return label
    return ""


def build_status(host: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
        code, stdout, stderr = run_ssh(host, REMOTE_STATUS_SCRIPT, timeout=timeout)
        sections = parse_sections(stdout)
        processes = sections.get("processes", "")
        summary_log = sections.get("summary_log", "")
        queue_log = sections.get("queue_log", "")
        active = bool(processes.strip())
        checkpoint_raw = sections.get("checkpoint", "")
        checkpoint_lines = checkpoint_raw.splitlines()
        checkpoint_file = checkpoint_lines[0] if checkpoint_lines else ""
        checkpoint_state: dict[str, Any] = {}
        if len(checkpoint_lines) > 1:
            try:
                checkpoint_state = json.loads("\n".join(checkpoint_lines[1:]))
            except json.JSONDecodeError:
                checkpoint_state = {"raw": "\n".join(checkpoint_lines[1:])}
        return {
            "ok": code == 0,
            "host": host,
            "local_updated_at": datetime.now(timezone.utc).isoformat(),
            "server_date": sections.get("date", ""),
            "elapsed_seconds": round(time.time() - started, 2),
            "active": active,
            "gpu": parse_gpu(sections.get("gpu", "")),
            "current_case": detect_current_case(processes, summary_log + "\n" + queue_log),
            "checkpoint_file": checkpoint_file,
            "checkpoint": checkpoint_state,
            "processes": processes,
            "queue_log": queue_log,
            "summary_log": summary_log,
            "results": sections.get("results", ""),
            "summary": sections.get("summary", ""),
            "stderr": stderr.strip(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "host": host,
            "local_updated_at": datetime.now(timezone.utc).isoformat(),
            "active": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll A6000 experiment status into a local JSON file.")
    parser.add_argument("--host", default=os.environ.get("A6000_HOST", "plypieng@100.114.7.72"))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    while True:
        status = build_status(args.host, timeout=args.timeout)
        write_status(out, status)
        print(f"[sync] wrote {out} ok={status.get('ok')} active={status.get('active')}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
