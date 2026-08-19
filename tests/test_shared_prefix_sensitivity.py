import argparse
import csv

from toy_kv_experiments.server import shared_prefix_sensitivity as sensitivity


def test_token_kind_marks_structure_without_calling_content_structural():
    assert sensitivity.token_kind("{") == "structural_marker"
    assert sensitivity.token_kind("</") == "structural_marker"
    assert sensitivity.token_kind("Alice") == "content"
    assert sensitivity.token_kind(" ") == "whitespace"


def test_selection_csv_filters_condition_and_preserves_order(tmp_path):
    path = tmp_path / "selection.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "task_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"condition": "KIVI-4", "task_id": "a"},
                {"condition": "KIVI-2", "task_id": "b"},
                {"condition": "KIVI-4", "task_id": "c"},
            ]
        )
    args = argparse.Namespace(
        task_ids="c,d",
        selection_csv=path,
        selection_condition="KIVI-4",
        limit=0,
    )

    assert sensitivity.selected_task_ids(args) == ["c", "d", "a"]


def test_trace_summary_separates_structural_and_other_positions():
    task = {
        "positions": [
            {
                "prediction_index": 0,
                "target_token_kind": "structural_marker",
                "top1_matches_frozen_target": True,
                "top1_matches_fp16": False,
            },
            {
                "prediction_index": 1,
                "target_token_kind": "content",
                "top1_matches_frozen_target": True,
                "top1_matches_fp16": True,
            },
        ]
    }

    summary = sensitivity.summarize_task_trace(task, has_fp16_reference=True)

    assert summary["first_top1_disagreement_index"] == 0
    assert summary["structural_top1_disagreement_rate"] == 1.0
    assert summary["other_top1_disagreement_rate"] == 0.0
