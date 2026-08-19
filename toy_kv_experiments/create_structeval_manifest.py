"""Freeze a reproducible StructEval task manifest for paired experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from toy_kv_experiments.pretrained_kv_quantization import (
    load_structeval_manifest_rows,
    load_structeval_rows,
    structeval_selection_counts,
)
from toy_kv_experiments.summarize_structeval_official import (
    CATEGORY_ORDER,
    structeval_official_category,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_category_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    categories: tuple[str, ...] = CATEGORY_ORDER,
) -> list[dict[str, Any]]:
    """Sample categories and task types reproducibly, as broadly as quota permits."""
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be a non-empty sequence without duplicates")
    unknown_categories = set(categories) - set(CATEGORY_ORDER)
    if unknown_categories:
        raise ValueError(f"unknown StructEval categories: {sorted(unknown_categories)}")
    if limit <= 0 or limit % len(categories):
        raise ValueError(f"category-balanced limit must be a positive multiple of {len(categories)}")
    category_quota = limit // len(categories)
    selected: list[dict[str, Any]] = []

    for category_index, category in enumerate(categories):
        category_rows = [row for row in rows if structeval_official_category(row) == category]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in category_rows:
            stratum = f"{row.get('input_type')}->{row.get('output_type')}"
            grouped.setdefault(stratum, []).append(row)

        rng = random.Random(seed + (category_index * 1009))
        for group_rows in grouped.values():
            rng.shuffle(group_rows)
        strata = sorted(grouped)
        rng.shuffle(strata)
        offsets = {stratum: 0 for stratum in strata}
        category_selection: list[dict[str, Any]] = []
        while len(category_selection) < category_quota:
            added = False
            for stratum in strata:
                offset = offsets[stratum]
                if offset >= len(grouped[stratum]):
                    continue
                category_selection.append(grouped[stratum][offset])
                offsets[stratum] = offset + 1
                added = True
                if len(category_selection) == category_quota:
                    break
            if not added:
                raise ValueError(f"category {category} contains fewer than {category_quota} tasks")
        selected.extend(category_selection)

    random.Random(seed).shuffle(selected)
    result = []
    for selection_index, source_row in enumerate(selected):
        row = dict(source_row)
        row["_structeval_sampling"] = "category-balanced"
        row["_structeval_seed"] = seed
        row["_structeval_stratum"] = f"{row.get('input_type')}->{row.get('output_type')}"
        row["_structeval_selection_index"] = selection_index
        result.append(row)
    return result


def select_complete_category_rows(
    rows: list[dict[str, Any]],
    *,
    categories: tuple[str, ...],
    seed: int,
) -> list[dict[str, Any]]:
    """Select every row from the requested official categories in source order."""
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories must be a non-empty sequence without duplicates")
    unknown_categories = set(categories) - set(CATEGORY_ORDER)
    if unknown_categories:
        raise ValueError(f"unknown StructEval categories: {sorted(unknown_categories)}")

    requested = set(categories)
    selected = [row for row in rows if structeval_official_category(row) in requested]
    observed = {structeval_official_category(row) for row in selected}
    missing = requested - observed
    if missing:
        raise ValueError(f"source contains no rows for categories: {sorted(missing)}")

    result = []
    for selection_index, source_row in enumerate(selected):
        row = dict(source_row)
        row["_structeval_sampling"] = "category-complete"
        row["_structeval_seed"] = seed
        row["_structeval_stratum"] = f"{row.get('input_type')}->{row.get('output_type')}"
        row["_structeval_selection_index"] = selection_index
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="StructEval JSONL source file.")
    parser.add_argument("--out", required=True, help="Destination manifest JSON file.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-type", default="JSON")
    parser.add_argument(
        "--sampling",
        choices=("stratified", "head", "category-balanced", "category-complete"),
        default="stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORY_ORDER,
        default=list(CATEGORY_ORDER),
        help="Official categories included by category-balanced sampling.",
    )
    parser.add_argument(
        "--parent-manifest",
        help="Optionally sample only from a previously frozen manifest.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    all_output_types = args.output_type.strip().upper() == "ALL"
    if args.sampling in {"category-balanced", "category-complete"}:
        if not all_output_types:
            parser.error(f"{args.sampling} sampling requires --output-type ALL")
        if args.sampling == "category-complete" and args.parent_manifest:
            parser.error("category-complete sampling must use the complete source dataset")
        source_rows = (
            load_structeval_manifest_rows(source, args.parent_manifest)
            if args.parent_manifest
            else load_all_rows(source)
        )
        if args.sampling == "category-balanced":
            rows = select_category_balanced_rows(
                source_rows,
                limit=args.limit,
                seed=args.seed,
                categories=tuple(args.categories),
            )
        else:
            rows = select_complete_category_rows(
                source_rows,
                categories=tuple(args.categories),
                seed=args.seed,
            )
    else:
        if args.parent_manifest:
            parser.error("--parent-manifest requires --sampling category-balanced")
        if tuple(args.categories) != CATEGORY_ORDER:
            parser.error("--categories requires --sampling category-balanced")
        rows = load_structeval_rows(
            source,
            limit=args.limit,
            output_type=None if all_output_types else args.output_type,
            sampling=args.sampling,
            seed=args.seed,
        )
    category_counts = Counter(structeval_official_category(row) for row in rows)
    output_type_counts = Counter(str(row.get("output_type")) for row in rows)
    manifest = {
        "manifest_version": 1,
        "source_file": source.name,
        "source_sha256": sha256(source),
        "output_type": args.output_type,
        "sampling": args.sampling,
        "seed": args.seed,
        "categories": list(args.categories),
        "task_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "output_type_counts": dict(sorted(output_type_counts.items())),
        "stratum_counts": structeval_selection_counts(rows),
        "tasks": [
            {
                "selection_index": row["_structeval_selection_index"],
                "task_id": row.get("task_id"),
                "task_name": row.get("task_name"),
                "input_type": row.get("input_type"),
                "output_type": row.get("output_type"),
                "stratum": row["_structeval_stratum"],
                "leaderboard_category": structeval_official_category(row),
            }
            for row in rows
        ],
    }
    if args.parent_manifest:
        parent_manifest = Path(args.parent_manifest).resolve()
        manifest["parent_manifest"] = parent_manifest.name
        manifest["parent_manifest_sha256"] = sha256(parent_manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tasks to {out}")
    print(json.dumps(manifest["category_counts"], sort_keys=True))
    print(json.dumps(manifest["stratum_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
