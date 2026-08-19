#!/usr/bin/env python3
"""Analyze paired official-KIVI StructEval-T evaluator outputs.

The analysis keeps aggregate benchmark values separate from paired task
transitions. Automated failure labels are surface-level diagnostics and are
exported alongside an annotation template; they are not causal attributions.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import statistics
import tomllib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONDITIONS = ("KIVI-4", "KIVI-2")


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in indexed:
            raise ValueError(f"{label} contains a missing or duplicate task ID: {task_id!r}")
        indexed[task_id] = row
    return indexed


def validate_pairs(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidate_label: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline = index_rows(baseline_rows, "FP16")
    candidate = index_rows(candidate_rows, candidate_label)
    if list(baseline) != list(candidate):
        if set(baseline) != set(candidate):
            raise ValueError(f"{candidate_label} task IDs do not match FP16")
        raise ValueError(f"{candidate_label} task order does not match FP16")
    pairs = [(baseline[task_id], candidate[task_id]) for task_id in baseline]
    for fp16, compressed in pairs:
        for field in ("input_type", "output_type", "query", "generation_protocol"):
            if fp16.get(field) != compressed.get(field):
                raise ValueError(
                    f"task {fp16['task_id']} differs in {field}: "
                    f"{fp16.get(field)!r} != {compressed.get(field)!r}"
                )
    return pairs


def category(row: dict[str, Any]) -> str:
    return "T-generation" if row.get("input_type") == "Text" else "T-conversion"


def score(row: dict[str, Any]) -> float:
    return float(row.get("final_eval_score") or 0.0)


def path_score(row: dict[str, Any]) -> float:
    return float(row.get("key_validation_score") or 0.0)


def parse_success(row: dict[str, Any]) -> bool:
    return float(row.get("render_score") or 0.0) == 1.0


def exact_mcnemar_pvalue(fp16_only: int, candidate_only: int) -> float:
    discordant = fp16_only + candidate_only
    if discordant == 0:
        return 1.0
    lower = min(fp16_only, candidate_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(deltas)
    estimates = sorted(
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def stratified_t_score_bootstrap_ci(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for fp16, candidate in pairs:
        grouped[category(fp16)].append(score(candidate) - score(fp16))
    expected = {"T-generation", "T-conversion"}
    if set(grouped) != expected:
        raise ValueError(f"expected {expected}, found {set(grouped)}")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        category_means = []
        for name in sorted(expected):
            values = grouped[name]
            category_means.append(
                statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            )
        estimates.append(statistics.fmean(category_means))
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def extract_code(generation: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    begin = generation.find("<|BEGIN_CODE|>")
    end = generation.rfind("<|END_CODE|>")
    if begin < 0:
        issues.append("missing_begin_marker")
        start = 0
    else:
        start = begin + len("<|BEGIN_CODE|>")
    if end < 0 or end < start:
        issues.append("missing_end_marker")
        end = len(generation)
    return generation[start:end].strip(), issues


def normalized_parser_error(output_type: str, code: str) -> tuple[str, str]:
    try:
        if output_type == "JSON":
            json.loads(code)
        elif output_type == "XML":
            ET.fromstring(code)
        elif output_type == "TOML":
            tomllib.loads(code)
        elif output_type == "CSV":
            list(csv.reader(io.StringIO(code), strict=True))
        elif output_type == "YAML":
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError:
                return "parser_unavailable", "PyYAML is not installed"
            yaml.safe_load(code)
        else:
            return "unsupported_format", f"no local diagnostic parser for {output_type}"
    except Exception as error:  # Parsers expose several format-specific error types.
        message = str(error).replace("\n", " ")
        lowered = message.lower()
        if re.match(r"^\\[nrt]", code):
            kind = "literal_escaped_layout_outside_string"
        elif "unterminated string" in lowered or "unclosed token" in lowered:
            kind = "unterminated_string_or_token"
        elif "escape" in lowered or "control character" in lowered:
            kind = "invalid_escape_or_control_character"
        elif "mapping values are not allowed" in lowered or "could not find expected" in lowered:
            kind = "indentation_or_mapping_syntax"
        elif "initial character for a key" in lowered:
            kind = "invalid_key_syntax"
        elif "mismatched tag" in lowered or "no element found" in lowered:
            kind = "unbalanced_or_mismatched_delimiter"
        elif "delimiter" in lowered or "expecting property name" in lowered:
            kind = "unbalanced_or_mismatched_delimiter"
        elif "extra data" in lowered or "junk after document" in lowered:
            kind = "extra_content_or_multiple_roots"
        elif "expecting value: line 1 column 1" in lowered:
            kind = "invalid_leading_token"
        elif "invalid token" in lowered or "not well-formed" in lowered:
            kind = "invalid_token"
        elif "number" in lowered:
            kind = "invalid_number"
        else:
            kind = "other_parser_error"
        return kind, message[:400]
    return "local_parser_succeeded", ""


def diagnose_parse_failure(row: dict[str, Any]) -> dict[str, str]:
    generation = str(row.get("generation") or "")
    code, marker_issues = extract_code(generation)
    stop_reason = str(row.get("stop_reason") or "unknown")
    if stop_reason == "max_new_tokens":
        failure_type = "generation_limit"
        detail = "generation reached max_new_tokens"
    elif marker_issues:
        failure_type = marker_issues[0]
        detail = ", ".join(marker_issues)
    elif stop_reason == "eos" and "<|END_CODE|>" not in generation:
        failure_type = "eos_before_end_marker"
        detail = "EOS occurred before the required end marker"
    elif not code:
        failure_type = "empty_payload"
        detail = "no payload was extracted"
    else:
        failure_type, detail = normalized_parser_error(str(row.get("output_type")), code)
    return {
        "automatic_failure_type": failure_type,
        "automatic_failure_detail": detail,
        "payload_preview": code[:500].replace("\r", "\\r").replace("\n", "\\n"),
    }


def transition_label(fp16: float, candidate: float, *, tolerance: float = 1e-12) -> str:
    delta = candidate - fp16
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "regressed"
    return "unchanged"


def paired_report(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    label: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    parse_transitions: Counter[str] = Counter()
    score_transitions: Counter[str] = Counter()
    path_transitions: Counter[str] = Counter()
    by_output: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for fp16, candidate in pairs:
        base_parse = parse_success(fp16)
        compressed_parse = parse_success(candidate)
        parse_transition = (
            "both_pass" if base_parse and compressed_parse else
            "fp16_only" if base_parse else
            "candidate_only" if compressed_parse else
            "both_fail"
        )
        score_transition = transition_label(score(fp16), score(candidate))
        path_transition = transition_label(path_score(fp16), path_score(candidate))
        parse_transitions[parse_transition] += 1
        score_transitions[score_transition] += 1
        path_transitions[path_transition] += 1
        by_output[str(fp16.get("output_type"))][parse_transition] += 1
        by_category[category(fp16)][parse_transition] += 1

        detail = {
            "condition": label,
            "task_id": str(fp16.get("task_id")),
            "task_name": str(fp16.get("task_name")),
            "category": category(fp16),
            "input_type": str(fp16.get("input_type")),
            "output_type": str(fp16.get("output_type")),
            "parse_transition": parse_transition,
            "score_transition": score_transition,
            "path_transition": path_transition,
            "fp16_parse": int(base_parse),
            "candidate_parse": int(compressed_parse),
            "fp16_score": score(fp16),
            "candidate_score": score(candidate),
            "score_delta": score(candidate) - score(fp16),
            "fp16_path_score": path_score(fp16),
            "candidate_path_score": path_score(candidate),
            "path_delta": path_score(candidate) - path_score(fp16),
            "fp16_stop_reason": str(fp16.get("stop_reason")),
            "candidate_stop_reason": str(candidate.get("stop_reason")),
            "fp16_generated_tokens": int(fp16.get("generated_tokens") or 0),
            "candidate_generated_tokens": int(candidate.get("generated_tokens") or 0),
        }
        task_rows.append(detail)
        if parse_transition == "fp16_only":
            failure_rows.append(
                {
                    **detail,
                    **diagnose_parse_failure(candidate),
                    "manual_failure_type": "",
                    "manual_notes": "",
                }
            )

    score_deltas = [row["score_delta"] for row in task_rows]
    path_deltas = [row["path_delta"] for row in task_rows]
    category_score_deltas = {
        name: statistics.fmean(
            score(candidate) - score(fp16)
            for fp16, candidate in pairs
            if category(fp16) == name
        )
        for name in ("T-generation", "T-conversion")
    }
    report = {
        "condition": label,
        "tasks": len(pairs),
        "parse_transitions": dict(parse_transitions),
        "parse_mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(
            parse_transitions["fp16_only"], parse_transitions["candidate_only"]
        ),
        "score_transitions": dict(score_transitions),
        "required_path_transitions": dict(path_transitions),
        "task_weighted_mean_score_delta": statistics.fmean(score_deltas),
        "task_weighted_mean_score_delta_95ci": paired_bootstrap_ci(
            score_deltas, samples=bootstrap_samples, seed=seed
        ),
        "category_mean_score_deltas": category_score_deltas,
        "structeval_t_delta": statistics.fmean(category_score_deltas.values()),
        "structeval_t_delta_95ci": stratified_t_score_bootstrap_ci(
            pairs, samples=bootstrap_samples, seed=seed + 1
        ),
        "mean_required_path_delta": statistics.fmean(path_deltas),
        "mean_required_path_delta_95ci": paired_bootstrap_ci(
            path_deltas, samples=bootstrap_samples, seed=seed + 2
        ),
        "parse_transitions_by_output_type": {
            name: dict(counts) for name, counts in sorted(by_output.items())
        },
        "parse_transitions_by_category": {
            name: dict(counts) for name, counts in sorted(by_category.items())
        },
        "automatic_failure_types": dict(
            Counter(row["automatic_failure_type"] for row in failure_rows)
        ),
        "diagnostic_boundary": (
            "Automatic failure types describe parser-visible surface errors. "
            "They do not identify which cached token or layer caused the error."
        ),
    }
    return report, task_rows, failure_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def format_ci(values: Iterable[float]) -> str:
    low, high = values
    return f"[{low:.4f}, {high:.4f}]"


def write_markdown(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Paired StructEval-T analysis",
        "",
        "The same 950 tasks are compared under FP16, KIVI-4, and KIVI-2.",
        "Confidence intervals are paired bootstrap intervals with resampling",
        "inside the generation and conversion categories for the StructEval-T score.",
        "",
        "| Condition | T-score delta | 95% CI | FP16-only parse | Candidate-only parse | McNemar p | Path regressions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in CONDITIONS:
        report = reports[label]
        transitions = report["parse_transitions"]
        lines.append(
            f"| {label} | {report['structeval_t_delta']:+.4f} | "
            f"{format_ci(report['structeval_t_delta_95ci'])} | "
            f"{transitions.get('fp16_only', 0)} | "
            f"{transitions.get('candidate_only', 0)} | "
            f"{report['parse_mcnemar_exact_two_sided_p']:.4g} | "
            f"{report['required_path_transitions'].get('regressed', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A candidate-only parse success and an FP16-only parse success are both",
            "behavioral transitions under deterministic decoding. Aggregate gains do not",
            "prove that quantization improves the model, and surface parser errors do not",
            "identify the causal cache position. The exported annotation table must be",
            "reviewed before making a delimiter- or tag-specific claim.",
            "",
            "## Automated surface diagnostics",
            "",
        ]
    )
    for label in CONDITIONS:
        diagnostics = reports[label]["automatic_failure_types"]
        lines.append(f"### {label}")
        lines.append("")
        for name, count in sorted(diagnostics.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{name}`: {count}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", required=True, type=Path)
    parser.add_argument("--kivi4", required=True, type=Path)
    parser.add_argument("--kivi2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    fp16 = load_rows(args.fp16)
    candidates = {
        "KIVI-4": load_rows(args.kivi4),
        "KIVI-2": load_rows(args.kivi2),
    }
    if len(fp16) != 950:
        raise ValueError(f"expected 950 FP16 rows, found {len(fp16)}")

    reports: dict[str, dict[str, Any]] = {}
    all_tasks: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for offset, label in enumerate(CONDITIONS):
        pairs = validate_pairs(fp16, candidates[label], label)
        report, task_rows, failure_rows = paired_report(
            pairs,
            label=label,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + offset * 100,
        )
        reports[label] = report
        all_tasks.extend(task_rows)
        all_failures.extend(failure_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "scope": "complete paired StructEval-T analysis",
        "fp16_file": str(args.fp16),
        "candidate_files": {"KIVI-4": str(args.kivi4), "KIVI-2": str(args.kivi2)},
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "conditions": reports,
    }
    (args.output_dir / "paired_analysis.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "task_transitions.csv", all_tasks)
    write_csv(args.output_dir / "parse_failure_annotations.csv", all_failures)
    write_markdown(args.output_dir / "SUMMARY.md", reports)
    print((args.output_dir / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
