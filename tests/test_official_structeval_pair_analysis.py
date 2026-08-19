import json

from toy_kv_experiments import analyze_official_structeval_pairs as analysis


def row(
    task_id: str,
    *,
    input_type: str = "Text",
    output_type: str = "JSON",
    parse: bool = True,
    score: float = 1.0,
    path_score: float = 1.0,
    generation: str = '<|BEGIN_CODE|>{"ok": true}<|END_CODE|>',
    stop_reason: str = "end_code",
) -> dict:
    return {
        "task_id": task_id,
        "task_name": "fixture",
        "input_type": input_type,
        "output_type": output_type,
        "query": "Return the requested structure.",
        "generation_protocol": "test-v1",
        "render_score": int(parse),
        "key_validation_score": path_score,
        "final_eval_score": score,
        "generation": generation,
        "stop_reason": stop_reason,
        "generated_tokens": 10,
    }


def test_validate_pairs_rejects_reordered_tasks():
    baseline = [row("a"), row("b")]
    candidate = [row("b"), row("a")]

    try:
        analysis.validate_pairs(baseline, candidate, "KIVI-4")
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("reordered tasks must be rejected")


def test_pair_report_separates_parse_directions_and_score_changes():
    baseline = [
        row("a", parse=True, score=1.0),
        row("b", input_type="CSV", parse=False, score=0.0),
        row("c", input_type="XML", parse=True, score=0.8),
        row("d", input_type="YAML", parse=False, score=0.1),
    ]
    candidate = [
        row("a", parse=False, score=0.0, generation="<|BEGIN_CODE|>{bad}<|END_CODE|>"),
        row("b", input_type="CSV", parse=True, score=1.0),
        row("c", input_type="XML", parse=True, score=0.8),
        row("d", input_type="YAML", parse=False, score=0.1),
    ]
    pairs = analysis.validate_pairs(baseline, candidate, "KIVI-4")

    report, task_rows, failures = analysis.paired_report(
        pairs,
        label="KIVI-4",
        bootstrap_samples=100,
        seed=7,
    )

    assert report["parse_transitions"] == {
        "fp16_only": 1,
        "candidate_only": 1,
        "both_pass": 1,
        "both_fail": 1,
    }
    assert report["score_transitions"] == {
        "regressed": 1,
        "improved": 1,
        "unchanged": 2,
    }
    assert report["parse_mcnemar_exact_two_sided_p"] == 1.0
    assert len(task_rows) == 4
    assert len(failures) == 1
    assert failures[0]["automatic_failure_type"] == "unbalanced_or_mismatched_delimiter"


def test_failure_diagnostics_distinguish_truncation_and_markers():
    truncated = row(
        "a",
        parse=False,
        generation='<|BEGIN_CODE|>{"unfinished":',
        stop_reason="max_new_tokens",
    )
    no_marker = row("b", parse=False, generation='{"x": 1}', stop_reason="eos")

    assert analysis.diagnose_parse_failure(truncated)["automatic_failure_type"] == "generation_limit"
    assert analysis.diagnose_parse_failure(no_marker)["automatic_failure_type"] == "missing_begin_marker"


def test_real_frozen_outputs_are_complete_and_paired():
    root = analysis.Path(__file__).resolve().parents[1] / "artifacts" / "structeval_t_20260731"
    if not root.exists():
        return
    fp16 = analysis.load_rows(root / "fp16" / "evaluation.json")
    kivi4 = analysis.load_rows(root / "kivi4" / "evaluation.json")
    kivi2 = analysis.load_rows(root / "kivi2" / "evaluation.json")

    assert len(analysis.validate_pairs(fp16, kivi4, "KIVI-4")) == 950
    assert len(analysis.validate_pairs(fp16, kivi2, "KIVI-2")) == 950
    assert sum(analysis.parse_success(row) for row in fp16) == 527
    assert sum(analysis.parse_success(row) for row in kivi4) == 564
    assert sum(analysis.parse_success(row) for row in kivi2) == 520
