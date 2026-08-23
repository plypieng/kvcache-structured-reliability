from toy_kv_experiments.make_shared_prefix_control import select_controls


def row(task_id, transition, input_type="CSV", output_type="JSON", length="100"):
    return {
        "condition": "KIVI-4",
        "task_id": task_id,
        "task_name": "task",
        "input_type": input_type,
        "output_type": output_type,
        "parse_transition": transition,
        "fp16_generated_tokens": length,
    }


def test_selects_nearest_unused_both_pass_control():
    rows = [
        row("f1", "fp16_only", length="100"),
        row("f2", "fp16_only", length="200"),
        row("c1", "both_pass", length="98"),
        row("c2", "both_pass", length="205"),
    ]

    selected = select_controls(rows)

    assert [(item["matched_failure_task_id"], item["task_id"]) for item in selected] == [
        ("f1", "c1"),
        ("f2", "c2"),
    ]
    assert selected[0]["length_delta"] == "-2"


def test_control_strata_must_exist():
    rows = [row("f1", "fp16_only", input_type="XML")]
    try:
        select_controls(rows)
    except ValueError as error:
        assert "no controls" in str(error)
    else:
        raise AssertionError("missing stratum should fail closed")
