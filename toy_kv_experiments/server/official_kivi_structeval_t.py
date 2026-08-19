#!/usr/bin/env python3
"""Run the complete StructEval-T track with the frozen official KIVI model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .official_kivi_longbench_repro import (
        DEFAULT_KIVI_COMMIT,
        load_model,
        read_jsonl,
        seed_everything,
        write_json_atomic,
    )
except ImportError:  # Direct execution on the A6000 server.
    from official_kivi_longbench_repro import (
        DEFAULT_KIVI_COMMIT,
        load_model,
        read_jsonl,
        seed_everything,
        write_json_atomic,
    )


STRUCTEVAL_SUFFIX = (
    "\n\nIMPORTANT: Only output the required output format. You must start the "
    "format/code with <|BEGIN_CODE|> and end the format/code with  <|END_CODE|>. "
    "No other text output (explanation, comments, etc.) are allowed.  "
    "Do not use markdown code fences."
)
EXPECTED_CATEGORY_COUNTS = {
    "T-generation": 250,
    "T-conversion": 700,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kivi-repo", type=Path, required=True)
    parser.add_argument("--expected-kivi-commit", default=DEFAULT_KIVI_COMMIT)
    parser.add_argument(
        "--model-name-or-path",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--model-revision",
        default="41b61a33a2483885c981aa79e0df6b32407ed873",
    )
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--structeval-source", type=Path, required=True)
    parser.add_argument("--structeval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("fp16", "kivi"), required=True)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def category(row: dict[str, Any]) -> str:
    track = "V" if bool(row.get("rendering", False)) else "T"
    task_kind = "generation" if row.get("input_type") == "Text" else "conversion"
    return f"{track}-{task_kind}"


def load_manifest_rows(
    source_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hash = sha256(source_path)
    if manifest.get("source_sha256") != source_hash:
        raise RuntimeError(
            "StructEval source hash mismatch: "
            f"manifest={manifest.get('source_sha256')} actual={source_hash}"
        )
    if manifest.get("sampling") != "category-complete":
        raise RuntimeError("StructEval-T run requires a category-complete manifest")
    if manifest.get("category_counts") != {
        "T-conversion": 700,
        "T-generation": 250,
    }:
        raise RuntimeError("StructEval-T manifest has unexpected category counts")

    source_rows = {
        str(row.get("task_id")): row
        for row in load_jsonl(source_path)
    }
    rows: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        task_id = str(task.get("task_id"))
        if task_id not in source_rows:
            raise RuntimeError(f"manifest task {task_id} is absent from the source")
        row = dict(source_rows[task_id])
        if bool(row.get("rendering", False)):
            raise RuntimeError(f"StructEval-T task {task_id} unexpectedly requires rendering")
        if category(row) not in EXPECTED_CATEGORY_COUNTS:
            raise RuntimeError(f"task {task_id} is outside StructEval-T")
        rows.append(row)

    if len(rows) != sum(EXPECTED_CATEGORY_COUNTS.values()):
        raise RuntimeError(f"expected 950 StructEval-T tasks, found {len(rows)}")
    if len({str(row.get("task_id")) for row in rows}) != len(rows):
        raise RuntimeError("StructEval-T manifest contains duplicate task IDs")
    return rows, manifest


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_smoke_rows(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Select smoke rows round-robin across the two StructEval-T categories."""
    grouped = {
        category_name: [
            row for row in rows if category(row) == category_name
        ]
        for category_name in EXPECTED_CATEGORY_COUNTS
    }
    offsets = {category_name: 0 for category_name in EXPECTED_CATEGORY_COUNTS}
    selected: list[dict[str, Any]] = []
    while len(selected) < min(limit, len(rows)):
        added = False
        for category_name in EXPECTED_CATEGORY_COUNTS:
            offset = offsets[category_name]
            if offset >= len(grouped[category_name]):
                continue
            selected.append(grouped[category_name][offset])
            offsets[category_name] = offset + 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def validate_resume_rows(
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    run_fingerprint: str,
) -> None:
    if len(rows) > len(tasks):
        raise RuntimeError("checkpoint contains more rows than the manifest")
    for index, row in enumerate(rows):
        expected_task_id = str(tasks[index].get("task_id"))
        if row.get("_sample_index") != index:
            raise RuntimeError(f"checkpoint row {index} has a non-contiguous sample index")
        if str(row.get("task_id")) != expected_task_id:
            raise RuntimeError(
                f"checkpoint row {index} has task {row.get('task_id')}, "
                f"expected {expected_task_id}"
            )
        if row.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"checkpoint row {index} has a different run fingerprint")


def build_prompt(tokenizer: Any, query: str) -> str:
    messages = [{"role": "user", "content": f"{query}{STRUCTEVAL_SUFFIX}"}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def marker_stopping_criteria(transformers: Any, tokenizer: Any) -> Any:
    marker_variants = []
    for text in ("<|END_CODE|>", " <|END_CODE|>", "\n<|END_CODE|>"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids and token_ids not in marker_variants:
            marker_variants.append(token_ids)

    class StopOnEndCode(transformers.StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            sequence = input_ids[0].tolist()
            return any(
                len(sequence) >= len(marker)
                and sequence[-len(marker) :] == marker
                for marker in marker_variants
            )

    return transformers.StoppingCriteriaList([StopOnEndCode()])


def compile_inference(
    checkpoint_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    write_json_atomic(output_path, checkpoint_rows)


def main() -> None:
    args = parse_args()
    args.kivi_repo = args.kivi_repo.expanduser().resolve()
    args.model_cache_dir = args.model_cache_dir.expanduser().resolve()
    args.structeval_source = args.structeval_source.expanduser().resolve()
    args.structeval_manifest = args.structeval_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    actual_commit = git_value(args.kivi_repo, "rev-parse", "HEAD")
    if actual_commit != args.expected_kivi_commit:
        raise RuntimeError(
            f"KIVI commit mismatch: expected {args.expected_kivi_commit}, found {actual_commit}"
        )
    tracked_diff = git_value(args.kivi_repo, "status", "--short", "--untracked-files=no")
    if tracked_diff:
        raise RuntimeError(f"Frozen KIVI checkout has tracked changes:\n{tracked_diff}")

    tasks, manifest = load_manifest_rows(
        args.structeval_source,
        args.structeval_manifest,
    )
    full_task_count = len(tasks)
    if args.limit > 0:
        tasks = select_smoke_rows(tasks, args.limit)
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    sys.path.insert(0, str(args.kivi_repo))
    os.environ["WANDB_DISABLED"] = "true"

    import numpy
    import torch
    import transformers
    from tqdm.auto import tqdm

    if args.condition == "kivi":
        import kivi_gemv  # noqa: F401

    seed_everything(torch, numpy, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "kivi_commit": actual_commit,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "structeval_source_sha256": sha256(args.structeval_source),
        "structeval_manifest_sha256": sha256(args.structeval_manifest),
        "structeval_manifest_task_count": full_task_count,
        "condition": args.condition,
        "k_bits": 16 if args.condition == "fp16" else args.k_bits,
        "v_bits": 16 if args.condition == "fp16" else args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "seed": args.seed,
        "generation_protocol": "structeval-prompt-paired-greedy-2048-v1",
    }
    run_fingerprint = fingerprint(provenance)
    metadata_path = args.output_dir / "run_metadata.json"
    checkpoint_path = args.output_dir / "inference.jsonl"
    inference_path = args.output_dir / "inference.json"
    completed_rows = read_jsonl(checkpoint_path)
    validate_resume_rows(completed_rows, tasks, run_fingerprint)

    started_at = time.time()
    metadata: dict[str, Any] = {
        **provenance,
        "status": "loading_model",
        "started_unix": started_at,
        "run_fingerprint": run_fingerprint,
        "kivi_repo": str(args.kivi_repo),
        "structeval_source": str(args.structeval_source),
        "structeval_manifest": str(args.structeval_manifest),
        "structeval_scope": "complete-T" if args.limit == 0 else "smoke",
        "structeval_categories": manifest.get("category_counts"),
        "official_kivi_execution": args.condition == "kivi",
        "official_structeval_prompt": True,
        "official_structeval_evaluator_required": True,
        "leaderboard_comparable": False,
        "leaderboard_boundary": (
            "Complete StructEval-T tasks and official evaluator, but paired greedy "
            "decoding with a 2048-token ceiling differs from the official inference API."
        ),
        "completed": len(completed_rows),
        "total": len(tasks),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json_atomic(metadata_path, metadata)

    model, tokenizer = load_model(args, torch, transformers)
    model_device = model.get_input_embeddings().weight.device
    stopping_criteria = marker_stopping_criteria(transformers, tokenizer)
    max_context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if max_context <= 0:
        raise RuntimeError("model config does not define max_position_embeddings")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    metadata.update(
        {
            "status": "running",
            "model_device": str(model_device),
            "max_context_tokens": max_context,
            "gpu_allocated_after_load_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_after_load_bytes": torch.cuda.memory_reserved(),
        }
    )
    write_json_atomic(metadata_path, metadata)

    with checkpoint_path.open("a", encoding="utf-8") as output_handle:
        iterator = range(len(completed_rows), len(tasks))
        for index in tqdm(
            iterator,
            initial=len(completed_rows),
            total=len(tasks),
            desc=f"{args.condition}:StructEval-T",
        ):
            row = tasks[index]
            prompt = build_prompt(tokenizer, str(row.get("query", "")))
            model_input = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
            ).to(model_device)
            input_tokens = int(model_input.input_ids.shape[-1])
            available_tokens = max_context - input_tokens
            if available_tokens <= 0:
                raise RuntimeError(
                    f"task {row.get('task_id')} prompt has {input_tokens} tokens, "
                    f"exceeding the {max_context}-token context"
                )
            generation_limit = min(args.max_new_tokens, available_tokens)

            sample_started = time.time()
            with torch.inference_mode():
                output = model.generate(
                    **model_input,
                    max_new_tokens=generation_limit,
                    num_beams=1,
                    do_sample=False,
                    stopping_criteria=stopping_criteria,
                )[0]
            torch.cuda.synchronize()
            generated_ids = output[input_tokens:]
            generation = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            generated_tokens = int(generated_ids.shape[-1])
            if "<|END_CODE|>" in generation:
                stop_reason = "end_code"
            elif generated_tokens >= generation_limit:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "eos"

            result_row = {
                **row,
                "generation": generation,
                "_sample_index": index,
                "run_fingerprint": run_fingerprint,
                "condition": args.condition,
                "key_bits": provenance["k_bits"],
                "value_bits": provenance["v_bits"],
                "group_size": args.group_size,
                "residual_length": args.residual_length,
                "generation_protocol": provenance["generation_protocol"],
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "generation_limit": generation_limit,
                "stop_reason": stop_reason,
                "elapsed_seconds": time.time() - sample_started,
            }
            output_handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
            completed_rows.append(result_row)
            metadata.update(
                {
                    "completed": len(completed_rows),
                    "last_task_id": row.get("task_id"),
                    "last_sample_seconds": result_row["elapsed_seconds"],
                    "elapsed_seconds": time.time() - started_at,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            write_json_atomic(metadata_path, metadata)

    compile_inference(completed_rows, inference_path)
    metadata.update(
        {
            "status": "complete",
            "completed": len(completed_rows),
            "elapsed_seconds": time.time() - started_at,
            "inference_path": str(inference_path),
        }
    )
    write_json_atomic(metadata_path, metadata)
    (args.output_dir / "COMPLETE").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n",
        encoding="utf-8",
    )
    print(f"complete: {inference_path}")


if __name__ == "__main__":
    main()
