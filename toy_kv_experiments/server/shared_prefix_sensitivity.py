#!/usr/bin/env python3
"""Replay frozen FP16 outputs under a shared prefix and compare next-token decisions.

Run FP16 first to establish a replay trace, then run KIVI with that trace as
the comparison reference. Both conditions receive the same prompt and the
same previously generated FP16 tokens (teacher forcing), so a next-token
difference is associated with the cache condition rather than divergent text.

This experiment localizes decision sensitivity. It does not by itself identify
which cached position, layer, key, or value caused the difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .official_kivi_longbench_repro import (
        DEFAULT_KIVI_COMMIT,
        load_model,
        seed_everything,
        write_json_atomic,
    )
    from .official_kivi_structeval_t import build_prompt
except ImportError:  # Direct execution on the A6000 server.
    from official_kivi_longbench_repro import (
        DEFAULT_KIVI_COMMIT,
        load_model,
        seed_everything,
        write_json_atomic,
    )
    from official_kivi_structeval_t import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kivi-repo", required=True, type=Path)
    parser.add_argument("--expected-kivi-commit", default=DEFAULT_KIVI_COMMIT)
    parser.add_argument(
        "--model-name-or-path",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--model-revision",
        default="41b61a33a2483885c981aa79e0df6b32407ed873",
    )
    parser.add_argument("--model-cache-dir", required=True, type=Path)
    parser.add_argument("--fp16-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", choices=("fp16", "kivi"), required=True)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--fp16-trace", type=Path)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument("--selection-condition", default="KIVI-4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_evaluation(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    if len(payload) != 950:
        raise ValueError(f"expected 950 FP16 evaluation rows, found {len(payload)}")
    return payload


def selected_task_ids(args: argparse.Namespace) -> list[str]:
    selected = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    if args.selection_csv:
        with args.selection_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("condition") == args.selection_condition:
                    selected.append(str(row.get("task_id", "")))
    selected = list(dict.fromkeys(task_id for task_id in selected if task_id))
    if not selected:
        raise ValueError("select tasks with --task-ids or --selection-csv")
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def token_kind(text: str) -> str:
    if not text:
        return "empty"
    if text.isspace():
        return "whitespace"
    if any(character in text for character in '{}[]:,<>/="'):
        return "structural_marker"
    if "BEGIN_CODE" in text or "END_CODE" in text:
        return "control_marker"
    return "content"


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def logits_record(
    *,
    torch: Any,
    tokenizer: Any,
    logits: Any,
    target_id: int,
    top_k: int,
    fp16_top1_id: int | None,
) -> dict[str, Any]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probs, k=top_k)
    top_ids = [int(value) for value in indices.tolist()]
    top_log_probs = [float(value) for value in values.tolist()]
    result = {
        "target_token_id": target_id,
        "target_token": token_text(tokenizer, target_id),
        "target_token_kind": token_kind(token_text(tokenizer, target_id)),
        "target_log_probability": float(log_probs[target_id]),
        "top1_token_id": top_ids[0],
        "top1_token": token_text(tokenizer, top_ids[0]),
        "top1_log_probability": top_log_probs[0],
        "top1_margin": top_log_probs[0] - top_log_probs[1] if len(top_log_probs) > 1 else 0.0,
        "top1_matches_frozen_target": top_ids[0] == target_id,
        "top_tokens": [
            {
                "token_id": token_id,
                "token": token_text(tokenizer, token_id),
                "log_probability": log_probability,
            }
            for token_id, log_probability in zip(top_ids, top_log_probs, strict=True)
        ],
    }
    if fp16_top1_id is not None:
        result.update(
            {
                "fp16_top1_token_id": fp16_top1_id,
                "fp16_top1_token": token_text(tokenizer, fp16_top1_id),
                "fp16_top1_log_probability_under_candidate": float(log_probs[fp16_top1_id]),
                "top1_matches_fp16": top_ids[0] == fp16_top1_id,
            }
        )
    return result


def trace_task(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    model_device: Any,
    evaluation_row: dict[str, Any],
    top_k: int,
    max_positions: int,
    fp16_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, str(evaluation_row.get("query", "")))
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(model_device)
    frozen_ids = tokenizer.encode(
        str(evaluation_row.get("generation", "")),
        add_special_tokens=False,
    )
    if not frozen_ids:
        raise ValueError(f"task {evaluation_row.get('task_id')} has no frozen output tokens")
    if max_positions > 0:
        frozen_ids = frozen_ids[:max_positions]

    reference_positions = None
    if fp16_reference is not None:
        reference_ids = fp16_reference.get("frozen_token_ids")
        if reference_ids != frozen_ids:
            raise ValueError(f"task {evaluation_row.get('task_id')} tokenization differs from FP16 trace")
        if fp16_reference.get("prompt_tokens") != int(prompt_ids.shape[-1]):
            raise ValueError(f"task {evaluation_row.get('task_id')} prompt length differs from FP16 trace")
        reference_positions = fp16_reference.get("positions")
        if not isinstance(reference_positions, list) or len(reference_positions) != len(frozen_ids):
            raise ValueError(f"task {evaluation_row.get('task_id')} has an invalid FP16 trace")

    positions: list[dict[str, Any]] = []
    with torch.inference_mode():
        output = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
        past_key_values = output.past_key_values
        next_logits = output.logits[0, -1]
        for index, target_id in enumerate(frozen_ids):
            fp16_top1_id = None
            if reference_positions is not None:
                fp16_top1_id = int(reference_positions[index]["top1_token_id"])
            positions.append(
                {
                    "prediction_index": index,
                    "prefix_output_tokens": index,
                    **logits_record(
                        torch=torch,
                        tokenizer=tokenizer,
                        logits=next_logits,
                        target_id=target_id,
                        top_k=top_k,
                        fp16_top1_id=fp16_top1_id,
                    ),
                }
            )
            if index + 1 == len(frozen_ids):
                break
            token = torch.tensor([[target_id]], dtype=torch.long, device=model_device)
            output = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            next_logits = output.logits[0, -1]

    result = {
        "task_id": str(evaluation_row.get("task_id")),
        "task_name": str(evaluation_row.get("task_name")),
        "input_type": str(evaluation_row.get("input_type")),
        "output_type": str(evaluation_row.get("output_type")),
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "frozen_output_tokens": len(frozen_ids),
        "frozen_token_ids": frozen_ids,
        "positions": positions,
    }
    result.update(summarize_task_trace(result, has_fp16_reference=fp16_reference is not None))
    return result


def summarize_task_trace(task: dict[str, Any], *, has_fp16_reference: bool) -> dict[str, Any]:
    positions = task.get("positions", [])
    target_matches = sum(bool(row.get("top1_matches_frozen_target")) for row in positions)
    summary: dict[str, Any] = {
        "frozen_target_top1_matches": target_matches,
        "frozen_target_top1_match_rate": target_matches / len(positions) if positions else 0.0,
    }
    if has_fp16_reference:
        disagreement = [row for row in positions if not row.get("top1_matches_fp16")]
        structural = [row for row in positions if row.get("target_token_kind") in {"structural_marker", "control_marker"}]
        structural_disagreement = [row for row in structural if not row.get("top1_matches_fp16")]
        other = [row for row in positions if row not in structural]
        other_disagreement = [row for row in other if not row.get("top1_matches_fp16")]
        summary.update(
            {
                "top1_disagreements_from_fp16": len(disagreement),
                "top1_disagreement_rate": len(disagreement) / len(positions) if positions else 0.0,
                "first_top1_disagreement_index": (
                    int(disagreement[0]["prediction_index"]) if disagreement else None
                ),
                "structural_positions": len(structural),
                "structural_top1_disagreements": len(structural_disagreement),
                "structural_top1_disagreement_rate": (
                    len(structural_disagreement) / len(structural) if structural else 0.0
                ),
                "other_positions": len(other),
                "other_top1_disagreements": len(other_disagreement),
                "other_top1_disagreement_rate": (
                    len(other_disagreement) / len(other) if other else 0.0
                ),
            }
        )
    return summary


def summarize_run(tasks: list[dict[str, Any]], *, condition: str) -> dict[str, Any]:
    positions = [position for task in tasks for position in task["positions"]]
    summary: dict[str, Any] = {
        "tasks": len(tasks),
        "positions": len(positions),
        "frozen_target_top1_match_rate": (
            sum(bool(row["top1_matches_frozen_target"]) for row in positions) / len(positions)
            if positions else 0.0
        ),
    }
    if condition == "kivi":
        disagreement = [row for row in positions if not row["top1_matches_fp16"]]
        structural = [row for row in positions if row["target_token_kind"] in {"structural_marker", "control_marker"}]
        other = [row for row in positions if row not in structural]
        summary.update(
            {
                "tasks_with_top1_disagreement": sum(
                    task["top1_disagreements_from_fp16"] > 0 for task in tasks
                ),
                "top1_disagreements_from_fp16": len(disagreement),
                "top1_disagreement_rate": len(disagreement) / len(positions) if positions else 0.0,
                "structural_positions": len(structural),
                "structural_top1_disagreement_rate": (
                    sum(not row["top1_matches_fp16"] for row in structural) / len(structural)
                    if structural else 0.0
                ),
                "other_positions": len(other),
                "other_top1_disagreement_rate": (
                    sum(not row["top1_matches_fp16"] for row in other) / len(other)
                    if other else 0.0
                ),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    args.kivi_repo = args.kivi_repo.expanduser().resolve()
    args.model_cache_dir = args.model_cache_dir.expanduser().resolve()
    args.fp16_evaluation = args.fp16_evaluation.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.condition == "kivi" and args.fp16_trace is None:
        raise ValueError("--fp16-trace is required for a KIVI comparison")
    if args.k_bits not in (2, 4) or args.v_bits not in (2, 4):
        raise ValueError("official KIVI supports only 2- or 4-bit caches")
    if args.top_k < 2:
        raise ValueError("top-k must be at least 2")

    actual_commit = git_value(args.kivi_repo, "rev-parse", "HEAD")
    if actual_commit != args.expected_kivi_commit:
        raise RuntimeError(f"KIVI revision mismatch: {actual_commit}")
    if git_value(args.kivi_repo, "status", "--short", "--untracked-files=no"):
        raise RuntimeError("frozen KIVI checkout has tracked changes")

    sys.path.insert(0, str(args.kivi_repo))
    os.environ["WANDB_DISABLED"] = "true"
    import numpy
    import torch
    import transformers

    if args.condition == "kivi":
        import kivi_gemv  # noqa: F401

    seed_everything(torch, numpy, args.seed)
    rows = load_evaluation(args.fp16_evaluation)
    indexed = {str(row["task_id"]): row for row in rows}
    task_ids = selected_task_ids(args)
    missing = [task_id for task_id in task_ids if task_id not in indexed]
    if missing:
        raise ValueError(f"selected tasks are absent from FP16 evaluation: {missing}")

    fp16_tasks: dict[str, dict[str, Any]] = {}
    if args.fp16_trace:
        fp16_payload = json.loads(args.fp16_trace.read_text(encoding="utf-8"))
        fp16_tasks = {str(task["task_id"]): task for task in fp16_payload["task_traces"]}
        missing_trace = [task_id for task_id in task_ids if task_id not in fp16_tasks]
        if missing_trace:
            raise ValueError(f"FP16 trace is missing selected tasks: {missing_trace}")

    model_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        model_cache_dir=args.model_cache_dir,
        condition=args.condition,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
    )
    started = time.time()
    model, tokenizer = load_model(model_args, torch, transformers)
    model_device = model.get_input_embeddings().weight.device
    task_traces = []
    for task_id in task_ids:
        task_traces.append(
            trace_task(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                model_device=model_device,
                evaluation_row=indexed[task_id],
                top_k=args.top_k,
                max_positions=args.max_positions,
                fp16_reference=fp16_tasks.get(task_id),
            )
        )
        print(
            f"{args.condition} {task_id}: "
            f"{task_traces[-1]['frozen_output_tokens']} positions",
            flush=True,
        )

    payload = {
        "method": "shared-prefix teacher-forced next-token sensitivity",
        "causal_boundary": (
            "The trace isolates next-token changes under an identical prefix, but does not "
            "attribute a change to a specific cache position, layer, key, or value."
        ),
        "condition": args.condition,
        "k_bits": 16 if args.condition == "fp16" else args.k_bits,
        "v_bits": 16 if args.condition == "fp16" else args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "kivi_commit": actual_commit,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_positions": args.max_positions,
        "selected_task_ids": task_ids,
        "elapsed_seconds": time.time() - started,
        "summary": summarize_run(task_traces, condition=args.condition),
        "task_traces": task_traces,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
