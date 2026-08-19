#!/usr/bin/env python3
"""Resumable LongBench quality reproduction using the official KIVI code.

The frozen KIVI checkout remains untouched. This harness imports its model
classes, prompt configuration, and evaluator, then adds provenance, task
selection, and per-example checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PAPER_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "musique",
    "2wikimqa",
    "gov_report",
    "qmsum",
    "multi_news",
    "lcc",
    "repobench-p",
    "triviaqa",
    "samsum",
    "trec",
    "passage_retrieval_en",
)

NO_CHAT_TEMPLATE_TASKS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
}

DEFAULT_KIVI_COMMIT = "67aba607a1deaeb18b70ae796ab25d05a08b3345"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable official-KIVI LongBench reproduction."
    )
    parser.add_argument("--kivi-repo", type=Path, required=True)
    parser.add_argument(
        "--expected-kivi-commit",
        default=DEFAULT_KIVI_COMMIT,
        help="Exact KIVI source revision required for this run.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--model-revision",
        default="41b61a33a2483885c981aa79e0df6b32407ed873",
        help="Mistral repository revision available when KIVI reported its table.",
    )
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("fp16", "kivi"), required=True)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument(
        "--tasks",
        default=",".join(PAPER_TASKS),
        help="Comma-separated LongBench task names.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Zero runs every example. Positive values are smoke tests only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-mode",
        choices=("published-intent", "frozen-source"),
        default="frozen-source",
        help=(
            "published-intent fixes the official Mistral model-name typo so the "
            "documented chat template is applied. frozen-source executes the "
            "checked-out function literally."
        ),
    )
    parser.add_argument(
        "--dataset-revision",
        default="f72191f71cd6fcd0da8a54f0915078efda579449",
        help="LongBench snapshot predating the KIVI results.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
    return rows


def validate_resume_rows(rows: list[dict[str, Any]], task: str) -> None:
    indices = [row.get("_sample_index") for row in rows]
    expected = list(range(len(rows)))
    if indices != expected:
        raise RuntimeError(
            f"{task} checkpoint is not a contiguous prefix: "
            f"found {indices[:10]}, expected {expected[:10]}"
        )


def seed_everything(torch: Any, numpy: Any, seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_tasks(raw: str) -> list[str]:
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    if not tasks:
        raise ValueError("At least one task is required.")
    unknown = sorted(set(tasks) - set(PAPER_TASKS))
    if unknown:
        raise ValueError(f"Tasks are not in the published 15-task table: {unknown}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("The task list contains duplicates.")
    return tasks


def build_chat_prompt(
    *,
    official_prediction: Any,
    tokenizer: Any,
    prompt: str,
    model_name: str,
    prompt_mode: str,
) -> str:
    if prompt_mode == "frozen-source":
        return official_prediction.build_chat(tokenizer, prompt, model_name)

    lowered = model_name.lower()
    if "longchat" in lowered:
        return official_prediction.build_chat(tokenizer, prompt, model_name)
    if "mistral" in lowered and "instruct" in lowered and "v0.2" in lowered:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def prepare_prompt(
    *,
    row: dict[str, Any],
    task: str,
    prompt_format: str,
    tokenizer: Any,
    model_name: str,
    max_length: int,
    prompt_mode: str,
    official_prediction: Any,
) -> str:
    prompt = prompt_format.format(**row)
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(
            tokenized[:half],
            skip_special_tokens=True,
        ) + tokenizer.decode(
            tokenized[-half:],
            skip_special_tokens=True,
        )
    if task not in NO_CHAT_TEMPLATE_TASKS:
        prompt = build_chat_prompt(
            official_prediction=official_prediction,
            tokenizer=tokenizer,
            prompt=prompt,
            model_name=model_name,
            prompt_mode=prompt_mode,
        )
    return prompt


def load_model(args: argparse.Namespace, torch: Any, transformers: Any) -> tuple[Any, Any]:
    model_name = args.model_name_or_path
    config = transformers.MistralConfig.from_pretrained(
        model_name,
        cache_dir=args.model_cache_dir,
        revision=args.model_revision,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.model_cache_dir,
        revision=args.model_revision,
        use_fast=False,
        trust_remote_code=True,
    )

    if args.condition == "kivi":
        if args.k_bits not in (2, 4) or args.v_bits not in (2, 4):
            raise ValueError("The published official KIVI path supports 2 or 4 bits.")
        from models.mistral_kivi import MistralForCausalLM_KIVI

        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        model = MistralForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=args.model_cache_dir,
            revision=args.model_revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        model = transformers.MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            cache_dir=args.model_cache_dir,
            revision=args.model_revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def score_task(official_evaluation: Any, task: str, rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError(f"Cannot score empty task {task}")
    predictions = [row["pred"] for row in rows]
    answers = [row["answers"] for row in rows]
    all_classes = rows[-1]["all_classes"]
    return official_evaluation.scorer(task, predictions, answers, all_classes)


def main() -> None:
    args = parse_args()
    args.kivi_repo = args.kivi_repo.expanduser().resolve()
    args.model_cache_dir = args.model_cache_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    tasks = parse_tasks(args.tasks)

    expected_commit = args.expected_kivi_commit
    actual_commit = git_value(args.kivi_repo, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"KIVI commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    tracked_diff = git_value(args.kivi_repo, "status", "--short", "--untracked-files=no")
    if tracked_diff:
        raise RuntimeError(f"Frozen KIVI checkout has tracked changes:\n{tracked_diff}")

    sys.path.insert(0, str(args.kivi_repo))
    os.environ["WANDB_DISABLED"] = "true"

    import numpy
    import torch
    import transformers
    from datasets import load_dataset
    from tqdm.auto import tqdm

    # Torch must load its shared libraries before the compiled KIVI extension.
    if args.condition == "kivi":
        import kivi_gemv  # noqa: F401

    official_prediction = load_module(
        "official_kivi_prediction",
        args.kivi_repo / "pred_long_bench.py",
    )
    official_evaluation = load_module(
        "official_kivi_evaluation",
        args.kivi_repo / "eval_long_bench.py",
    )

    seed_everything(torch, numpy, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    model2maxlen = json.loads(
        (args.kivi_repo / "config" / "model2maxlen.json").read_text(encoding="utf-8")
    )
    max_length = int(model2maxlen[model_name])
    dataset2prompt = json.loads(
        (args.kivi_repo / "config" / "dataset2prompt.json").read_text(
            encoding="utf-8"
        )
    )
    dataset2maxlen = json.loads(
        (args.kivi_repo / "config" / "dataset2maxlen.json").read_text(
            encoding="utf-8"
        )
    )

    started_at = time.time()
    metadata_path = args.output_dir / "run_metadata.json"
    result_path = args.output_dir / "result.json"
    task_status: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "status": "loading_model",
        "started_unix": started_at,
        "kivi_repo": str(args.kivi_repo),
        "kivi_commit": actual_commit,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "condition": args.condition,
        "k_bits": 16 if args.condition == "fp16" else args.k_bits,
        "v_bits": 16 if args.condition == "fp16" else args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "max_context_tokens": max_length,
        "tasks": tasks,
        "limit_per_task": args.limit_per_task,
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
        "dataset_revision": args.dataset_revision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "paper_comparable": args.limit_per_task == 0 and tasks == list(PAPER_TASKS),
        "task_status": task_status,
    }
    write_json_atomic(metadata_path, metadata)

    model, tokenizer = load_model(args, torch, transformers)
    model_device = model.get_input_embeddings().weight.device
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    metadata["status"] = "running"
    metadata["model_device"] = str(model_device)
    metadata["model_supports_cache_class"] = bool(
        getattr(model, "_supports_cache_class", False)
    )
    metadata["gpu_allocated_after_load_bytes"] = torch.cuda.memory_allocated()
    metadata["gpu_reserved_after_load_bytes"] = torch.cuda.memory_reserved()
    write_json_atomic(metadata_path, metadata)

    scores: dict[str, float] = {}
    for task in tasks:
        task_started = time.time()
        output_path = args.output_dir / f"{task}.jsonl"
        completed_rows = read_jsonl(output_path)
        validate_resume_rows(completed_rows, task)

        load_kwargs: dict[str, Any] = {
            "path": "THUDM/LongBench",
            "name": task,
            "split": "test",
        }
        if args.dataset_revision:
            load_kwargs["revision"] = args.dataset_revision
        dataset = load_dataset(**load_kwargs)
        total_rows = len(dataset)
        if args.limit_per_task > 0:
            total_rows = min(total_rows, args.limit_per_task)
            dataset = dataset.select(range(total_rows))
        if len(completed_rows) > total_rows:
            raise RuntimeError(
                f"{task} has {len(completed_rows)} checkpoints for {total_rows} rows."
            )

        task_status[task] = {
            "status": "running",
            "completed": len(completed_rows),
            "total": total_rows,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        }
        write_json_atomic(metadata_path, metadata)

        with output_path.open("a", encoding="utf-8") as output_handle:
            iterator = range(len(completed_rows), total_rows)
            for index in tqdm(
                iterator,
                initial=len(completed_rows),
                total=total_rows,
                desc=f"{args.condition}:{task}",
            ):
                row = dict(dataset[index])
                prompt = prepare_prompt(
                    row=row,
                    task=task,
                    prompt_format=dataset2prompt[task],
                    tokenizer=tokenizer,
                    model_name=model_name,
                    max_length=max_length,
                    prompt_mode=args.prompt_mode,
                    official_prediction=official_prediction,
                )
                model_input = tokenizer(
                    prompt,
                    truncation=False,
                    return_tensors="pt",
                ).to(model_device)
                context_length = int(model_input.input_ids.shape[-1])
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": int(dataset2maxlen[task]),
                    "num_beams": 1,
                    "do_sample": False,
                    "temperature": 1.0,
                }
                if task == "samsum":
                    generation_kwargs.update(
                        {
                            "min_length": context_length + 1,
                            "eos_token_id": [
                                tokenizer.eos_token_id,
                                tokenizer.encode(
                                    "\n",
                                    add_special_tokens=False,
                                )[-1],
                            ],
                        }
                    )

                sample_started = time.time()
                with torch.inference_mode():
                    output = model.generate(
                        **model_input,
                        **generation_kwargs,
                    )[0]
                torch.cuda.synchronize()
                generated_tokens = int(output.shape[-1] - context_length)
                prediction = tokenizer.decode(
                    output[context_length:],
                    skip_special_tokens=True,
                )
                prediction = official_prediction.post_process(
                    prediction,
                    model_name,
                )
                result_row = {
                    "pred": prediction,
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row["length"],
                    "_sample_index": index,
                    "_prompt_tokens": context_length,
                    "_generated_tokens": generated_tokens,
                    "_elapsed_seconds": time.time() - sample_started,
                }
                output_handle.write(
                    json.dumps(result_row, ensure_ascii=False) + "\n"
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
                completed_rows.append(result_row)
                task_status[task]["completed"] = len(completed_rows)
                task_status[task]["last_sample_seconds"] = result_row[
                    "_elapsed_seconds"
                ]
                metadata["gpu_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated()
                )
                metadata["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                write_json_atomic(metadata_path, metadata)

        score = score_task(official_evaluation, task, completed_rows)
        scores[task] = score
        task_status[task].update(
            {
                "status": "complete",
                "score": score,
                "elapsed_seconds": time.time() - task_started,
            }
        )
        write_json_atomic(result_path, scores)
        write_json_atomic(metadata_path, metadata)

    metadata.update(
        {
            "status": "complete",
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "mean_score": sum(scores.values()) / len(scores),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    write_json_atomic(result_path, scores)
    write_json_atomic(metadata_path, metadata)
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
