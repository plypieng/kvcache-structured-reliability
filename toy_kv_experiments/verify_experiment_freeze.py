#!/usr/bin/env python3
"""Verify checksums, provenance, and pairing of the frozen StructEval-T run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
    "model_revision": "41b61a33a2483885c981aa79e0df6b32407ed873",
    "kivi_commit": "67aba607a1deaeb18b70ae796ab25d05a08b3345",
    "group_size": 32,
    "residual_length": 128,
    "seed": 42,
    "max_new_tokens": 2048,
    "generation_protocol": "structeval-prompt-paired-greedy-2048-v1",
}
CONDITIONS = {
    "fp16": ("fp16", 16, 16),
    "kivi4": ("kivi", 4, 4),
    "kivi2": ("kivi", 2, 2),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    checksum_path = root / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, relative = line.split(maxsplit=1)
        relative = relative.strip()
        actual_hash = sha256(root / relative)
        if actual_hash != expected_hash:
            raise ValueError(f"checksum mismatch for {relative}: {actual_hash}")
        checksums[relative] = actual_hash
    if len(checksums) != 6:
        raise ValueError(f"expected six frozen files, found {len(checksums)}")
    return checksums


def verify_condition(root: Path, label: str) -> tuple[list[str], dict[str, Any]]:
    expected_condition, expected_k, expected_v = CONDITIONS[label]
    metadata = json.loads((root / label / "run_metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((root / label / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 950:
        raise ValueError(f"{label} must contain exactly 950 evaluated rows")
    for field, expected in EXPECTED.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{label} metadata differs in {field}: {metadata.get(field)!r}")
    if metadata.get("condition") != expected_condition:
        raise ValueError(f"{label} has unexpected condition")
    if (metadata.get("k_bits"), metadata.get("v_bits")) != (expected_k, expected_v):
        raise ValueError(f"{label} has unexpected bit widths")
    if metadata.get("status") != "complete" or metadata.get("completed") != 950:
        raise ValueError(f"{label} metadata is not complete")

    task_ids: list[str] = []
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in task_ids:
            raise ValueError(f"{label} contains a missing or duplicate task ID")
        if row.get("_sample_index") != index:
            raise ValueError(f"{label} row {index} has an invalid sample index")
        if (row.get("key_bits"), row.get("value_bits")) != (expected_k, expected_v):
            raise ValueError(f"{label} task {task_id} has unexpected bit widths")
        if row.get("generation_protocol") != EXPECTED["generation_protocol"]:
            raise ValueError(f"{label} task {task_id} has an unexpected protocol")
        task_ids.append(task_id)
    return task_ids, metadata


def verify(root: Path) -> dict[str, Any]:
    checksums = verify_checksums(root)
    task_orders: dict[str, list[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for label in CONDITIONS:
        task_orders[label], metadata[label] = verify_condition(root, label)
    if task_orders["kivi4"] != task_orders["fp16"] or task_orders["kivi2"] != task_orders["fp16"]:
        raise ValueError("condition task orders are not paired")
    return {
        "valid": True,
        "conditions": list(CONDITIONS),
        "tasks_per_condition": 950,
        "paired_task_order": True,
        "verified_files": len(checksums),
        "model_revision": EXPECTED["model_revision"],
        "kivi_commit": EXPECTED["kivi_commit"],
        "run_fingerprints": {
            label: metadata[label].get("run_fingerprint") for label in CONDITIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("artifacts/structeval_t_20260731"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
