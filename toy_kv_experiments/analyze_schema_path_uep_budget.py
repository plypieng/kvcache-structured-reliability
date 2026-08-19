from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from toy_kv_experiments import pretrained_kv_quantization as kvq


DEFAULT_STRUCTEVAL_JSONL = Path("toy_kv_experiments/data/structeval_full/structeval_test.jsonl")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return float(ordered[index])


def effective_bits(
    token_count: int,
    protected_older_count: int,
    residual_length: int,
    base_bits: float,
    protected_bits: float,
) -> float:
    """Estimate average bits per K or V scalar after prefill quantization."""
    if token_count <= 0:
        return 0.0
    residual_count = min(max(0, residual_length), token_count)
    older_count = token_count - residual_count
    protected_older_count = min(max(0, protected_older_count), older_count)
    ordinary_older_count = older_count - protected_older_count
    total_bits = (
        ordinary_older_count * base_bits
        + protected_older_count * protected_bits
        + residual_count * 16
    )
    return total_bits / token_count


def analyze_rows(
    tokenizer,
    rows: list[dict[str, Any]],
    base_bits: int,
    protected_bits: int,
    residual_length: int,
    protection_mode: str,
    target_average_bits: float | None,
    protection_budget_order: str,
    official_prompt: bool,
    official_model_name: str,
) -> dict[str, Any]:
    per_task = []
    term_counter: Counter[str] = Counter()

    for row in rows:
        prompt = str(row["query"])
        if official_prompt:
            prompt = kvq.build_structeval_official_prompt(prompt, model_name=official_model_name)
        chat_prompt = kvq.build_chat_prompt(tokenizer, prompt)
        token_ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"][0]
        required = row.get("raw_output_metric") or []
        terms = kvq.constraint_terms_from_required_paths(required)
        weights = kvq.constraint_term_weights_from_required_paths(required)
        term_counter.update(terms)
        candidate_mask = kvq.structure_token_mask(
            tokenizer,
            token_ids,
            mode=protection_mode,
            constraint_terms=terms,
        )
        candidate_scores = kvq.structure_token_scores(
            tokenizer,
            token_ids,
            mode=protection_mode,
            constraint_term_weights=weights,
        )
        mask = kvq.budgeted_protection_mask(
            candidate_mask,
            base_bits=base_bits,
            protected_bits=protected_bits,
            residual_length=residual_length,
            target_average_bits=target_average_bits,
            order=protection_budget_order,
            scores=candidate_scores,
        )
        candidate_protected_count = int(candidate_mask.sum().item()) if candidate_mask is not None else 0
        protected_count = int(mask.sum().item()) if mask is not None else 0
        token_count = int(token_ids.numel())
        residual_count = min(max(0, residual_length), token_count)
        older_count = token_count - residual_count
        candidate_score_sum = (
            float(candidate_scores[:older_count][candidate_mask[:older_count]].sum().item())
            if candidate_mask is not None and older_count > 0
            else 0.0
        )
        protected_score_sum = float(candidate_scores[mask].sum().item()) if mask is not None else 0.0
        protected_older_count = int(mask[:older_count].sum().item()) if mask is not None and older_count > 0 else 0
        candidate_protected_older_count = (
            int(candidate_mask[:older_count].sum().item()) if candidate_mask is not None and older_count > 0 else 0
        )
        avg_bits = effective_bits(
            token_count=token_count,
            protected_older_count=protected_older_count,
            residual_length=residual_length,
            base_bits=base_bits,
            protected_bits=protected_bits,
        )
        per_task.append(
            {
                "task_id": row.get("task_id"),
                "task_name": row.get("task_name"),
                "prompt_tokens": token_count,
                "required_path_count": len(required),
                "constraint_term_count": len(terms),
                "candidate_protected_tokens": candidate_protected_count,
                "candidate_protected_fraction": candidate_protected_count / token_count if token_count else 0.0,
                "protected_tokens": protected_count,
                "protected_fraction": protected_count / token_count if token_count else 0.0,
                "candidate_score_sum": candidate_score_sum,
                "protected_score_sum": protected_score_sum,
                "protected_score_fraction": protected_score_sum / candidate_score_sum if candidate_score_sum > 0 else 0.0,
                "residual_tokens": residual_count,
                "residual_fraction": residual_count / token_count if token_count else 0.0,
                "older_tokens": older_count,
                "candidate_protected_older_tokens": candidate_protected_older_count,
                "candidate_protected_older_fraction": candidate_protected_older_count / older_count
                if older_count
                else 0.0,
                "protected_older_tokens": protected_older_count,
                "protected_older_fraction": protected_older_count / older_count if older_count else 0.0,
                "estimated_average_bits_per_kv_scalar": avg_bits,
                "estimated_compression_vs_fp16": 16 / avg_bits if avg_bits else 0.0,
                "constraint_terms": terms,
                "constraint_term_weights": weights,
            }
        )

    prompt_tokens = [float(row["prompt_tokens"]) for row in per_task]
    candidate_protected_fraction = [float(row["candidate_protected_fraction"]) for row in per_task]
    protected_fraction = [float(row["protected_fraction"]) for row in per_task]
    protected_score_fraction = [float(row["protected_score_fraction"]) for row in per_task]
    candidate_protected_older_fraction = [float(row["candidate_protected_older_fraction"]) for row in per_task]
    protected_older_fraction = [float(row["protected_older_fraction"]) for row in per_task]
    avg_bits_values = [float(row["estimated_average_bits_per_kv_scalar"]) for row in per_task]
    compression_values = [float(row["estimated_compression_vs_fp16"]) for row in per_task]
    summary = {
        "n": len(per_task),
        "base_bits": base_bits,
        "protected_bits": protected_bits,
        "residual_length": residual_length,
        "protection_mode": protection_mode,
        "target_average_bits": target_average_bits,
        "protection_budget_order": protection_budget_order,
        "official_prompt": official_prompt,
        "mean_prompt_tokens": round(mean(prompt_tokens), 2),
        "p90_prompt_tokens": round(percentile(prompt_tokens, 0.90), 2),
        "mean_candidate_protected_fraction": round(mean(candidate_protected_fraction), 4),
        "mean_protected_fraction": round(mean(protected_fraction), 4),
        "mean_protected_score_fraction": round(mean(protected_score_fraction), 4),
        "mean_candidate_protected_older_fraction": round(mean(candidate_protected_older_fraction), 4),
        "mean_protected_older_fraction": round(mean(protected_older_fraction), 4),
        "mean_estimated_average_bits_per_kv_scalar": round(mean(avg_bits_values), 4),
        "mean_estimated_compression_vs_fp16": round(mean(compression_values), 4),
        "top_constraint_terms": term_counter.most_common(25),
    }
    return {
        "summary": summary,
        "per_task": per_task,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Schema-path UEP budget estimate")
    print("=" * 36)
    for key in (
        "n",
        "base_bits",
        "protected_bits",
        "residual_length",
        "protection_mode",
        "target_average_bits",
        "protection_budget_order",
        "mean_prompt_tokens",
        "p90_prompt_tokens",
        "mean_candidate_protected_fraction",
        "mean_protected_fraction",
        "mean_protected_score_fraction",
        "mean_candidate_protected_older_fraction",
        "mean_protected_older_fraction",
        "mean_estimated_average_bits_per_kv_scalar",
        "mean_estimated_compression_vs_fp16",
    ):
        print(f"{key}: {summary[key]}")
    print()
    print("Top constraint terms:")
    for term, count in summary["top_constraint_terms"][:12]:
        print(f"  {term}: {count}")
    print()
    print("Example tasks:")
    for row in report["per_task"][:5]:
        print(
            f"  {row['task_id']} tokens={row['prompt_tokens']} "
            f"protected={row['protected_tokens']} "
            f"score={row['protected_score_fraction']:.2f} "
            f"avg_bits={row['estimated_average_bits_per_kv_scalar']:.3f} "
            f"compression={row['estimated_compression_vs_fp16']:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate schema/path-token UEP coverage and KV-cache bit budget without model inference."
    )
    parser.add_argument("--model-dir", default=str(kvq.DEFAULT_MODEL_DIR))
    parser.add_argument("--structeval-jsonl", default=str(DEFAULT_STRUCTEVAL_JSONL))
    parser.add_argument("--output-type", default="JSON")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--base-bits", type=int, default=4)
    parser.add_argument("--protected-bits", type=int, default=8)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--target-average-bits", type=float, default=None)
    parser.add_argument("--protection-budget-order", choices=kvq.PROTECTION_BUDGET_ORDERS, default="prefix")
    parser.add_argument(
        "--protection-mode",
        choices=kvq.STRUCTURE_TOKEN_PROTECTION_MODES,
        default="constraint-paths",
    )
    parser.add_argument("--official-structeval-prompt", action="store_true")
    parser.add_argument("--official-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model_dir = kvq.resolve_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    rows = kvq.load_structeval_rows(args.structeval_jsonl, limit=args.limit, output_type=args.output_type)
    report = analyze_rows(
        tokenizer=tokenizer,
        rows=rows,
        base_bits=args.base_bits,
        protected_bits=args.protected_bits,
        residual_length=args.residual_length,
        protection_mode=args.protection_mode,
        target_average_bits=args.target_average_bits,
        protection_budget_order=args.protection_budget_order,
        official_prompt=args.official_structeval_prompt,
        official_model_name=args.official_model_name,
    )
    print_summary(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
