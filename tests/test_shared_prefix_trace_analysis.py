from toy_kv_experiments import analyze_shared_prefix_trace as analysis


def position(token_id, token, *, target_id=7, target="}", kind="structural_marker"):
    return {
        "prediction_index": 0,
        "target_token_id": target_id,
        "target_token": target,
        "target_token_kind": kind,
        "target_log_probability": -0.2,
        "top1_token_id": token_id,
        "top1_token": token,
    }


def trace(condition, positions):
    task = {
        "task_id": "a",
        "task_name": "Text to JSON",
        "input_type": "Text",
        "output_type": "JSON",
        "prompt_tokens": 8,
        "frozen_token_ids": [7],
        "positions": positions,
    }
    if condition == "kivi":
        task.update(
            {
                "top1_disagreements_from_fp16": 1,
                "top1_disagreement_rate": 1.0,
                "structural_positions": 1,
                "structural_top1_disagreements": 1,
                "structural_top1_disagreement_rate": 1.0,
                "other_positions": 0,
                "other_top1_disagreements": 0,
                "other_top1_disagreement_rate": 0.0,
            }
        )
    return {
        "method": "shared-prefix teacher-forced next-token sensitivity",
        "causal_boundary": "No cache-entry attribution.",
        "condition": condition,
        "k_bits": 16 if condition == "fp16" else 4,
        "v_bits": 16 if condition == "fp16" else 4,
        "model_name_or_path": "model",
        "model_revision": "revision",
        "kivi_commit": "commit",
        "seed": 42,
        "selected_task_ids": ["a"],
        "task_traces": [task],
    }


def test_analyze_reports_structural_top1_disagreement():
    fp16 = trace("fp16", [position(7, "}")])
    kivi = trace("kivi", [position(8, "]")])

    report, task_rows, disagreement_rows = analysis.analyze(fp16, kivi)

    assert report["tasks_with_top1_disagreement"] == 1
    assert report["structural_top1_disagreements"] == 1
    assert report["other_top1_disagreements"] == 0
    assert task_rows[0]["first_top1_disagreement_index"] == 0
    assert disagreement_rows[0]["fp16_top1_token"] == "}"
    assert disagreement_rows[0]["kivi_top1_token"] == "]"


def test_validate_rejects_different_task_selection():
    fp16 = trace("fp16", [position(7, "}")])
    kivi = trace("kivi", [position(8, "]")])
    kivi["selected_task_ids"] = ["b"]

    try:
        analysis.validate_pair(fp16, kivi)
    except ValueError as error:
        assert "selected_task_ids" in str(error)
    else:
        raise AssertionError("mismatched selection must be rejected")
