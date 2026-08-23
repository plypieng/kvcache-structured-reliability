import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from toy_kv_experiments import analyze_schema_path_uep_budget as uep_budget
from toy_kv_experiments import analyze_paired_structeval as paired_analysis
from toy_kv_experiments import check_structeval_official_artifact as artifact_check
from toy_kv_experiments import check_structeval_poster_manifest as poster_manifest_check
from toy_kv_experiments import check_structeval_t_manifest as structeval_t_manifest_check
from toy_kv_experiments import check_structeval_text_dev_manifest as text_dev_manifest_check
from toy_kv_experiments import create_structeval_manifest as manifest_creator
from toy_kv_experiments import make_structeval_evaluator_smoke as evaluator_smoke
from toy_kv_experiments import pretrained_kv_quantization as kvq
from toy_kv_experiments import summarize_structeval_matrix as matrix_summary
from toy_kv_experiments import summarize_structeval_official as official_summary
from toy_kv_experiments import summarize_structeval_poster as poster_summary
from toy_kv_experiments import summarize_structeval_results as result_summary

STRUCTEVAL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "StructEval"
if STRUCTEVAL_ROOT.exists():
    sys.path.insert(0, str(STRUCTEVAL_ROOT))


def test_official_structeval_prompt_adds_required_tags_instruction():
    prompt = kvq.build_structeval_official_prompt("Return a JSON object.")

    assert "Return a JSON object." in prompt
    assert "<|BEGIN_CODE|>" in prompt
    assert "<|END_CODE|>" in prompt
    assert "Do not use markdown code fences." in prompt


def test_official_extraction_reads_begin_end_block():
    text = """<|BEGIN_CODE|>
{"right": true}
<|END_CODE|>"""

    extracted = kvq.extract_structeval_official_code(text, output_type="json")

    assert json.loads(extracted) == {"right": True}


def test_official_json_parse_and_score_match_structeval_nonrenderable_formula():
    parsed, candidate, error = kvq.parse_json_output_with_error(
        "<|BEGIN_CODE|>\n{\"items\": [{\"name\": \"Alice\"}]}\n<|END_CODE|>",
        official_structeval=True,
    )

    assert error == ""
    assert json.loads(candidate) == {"items": [{"name": "Alice"}]}
    assert kvq.required_json_path_rate(parsed, ["items.*.name"]) == 1.0
    assert kvq.structeval_nonrenderable_score(render_score=1.0, key_validation_score=0.75) == 0.8


def test_json_path_checker_matches_official_structeval_for_json_paths():
    eval_utils_path = STRUCTEVAL_ROOT / "structeval" / "eval_engine" / "eval_utils.py"
    if not eval_utils_path.exists():
        return
    for optional_module in ("xmltodict", "toml"):
        sys.modules.setdefault(optional_module, types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location(
        "structeval_eval_utils",
        eval_utils_path,
    )
    assert spec is not None and spec.loader is not None
    official_eval_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_eval_utils)

    parsed = {
        "novel": {
            "title": "The Great Gatsby",
            "characters": [
                {"name": "Nick", "role": "Narrator"},
                {"name": "Gatsby", "role": "Protagonist"},
            ],
        }
    }
    paths = [
        "novel.title",
        "novel.characters[0].name",
        "novel.characters.*.role",
        "novel.*.name",
        "novel.characters[2].name",
        "novel.publisher.name",
    ]

    for path in paths:
        assert kvq.has_path(parsed, kvq.path_parts(path)) == official_eval_utils.path_exists(parsed, path)


def test_progress_iterator_preserves_items_when_disabled():
    pairs = [("task-a", 16), ("task-a", 8)]

    assert list(kvq.progress_iter(pairs, total=2, enabled=False, desc="test")) == pairs


def _write_structeval_fixture(path: Path) -> None:
    rows = []
    for input_type in ("Text", "CSV", "XML", "YAML", "TOML"):
        for index in range(4):
            rows.append(
                {
                    "task_id": f"{input_type}-{index}",
                    "task_name": f"{input_type} to JSON",
                    "input_type": input_type,
                    "output_type": "JSON",
                    "query": f"Convert {input_type} example {index}",
                    "raw_output_metric": ["result.value"],
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_structeval_stratified_sampling_balances_and_scrambles_families(tmp_path):
    fixture = tmp_path / "structeval.jsonl"
    _write_structeval_fixture(fixture)

    rows = kvq.load_structeval_rows(fixture, limit=10, output_type="JSON", sampling="stratified", seed=7)

    assert kvq.structeval_selection_counts(rows) == {"CSV": 2, "TOML": 2, "Text": 2, "XML": 2, "YAML": 2}
    assert [row["input_type"] for row in rows] != ["Text"] * 4 + ["CSV"] * 4 + ["XML"] * 2
    assert [row["task_id"] for row in rows] == [
        row["task_id"]
        for row in kvq.load_structeval_rows(fixture, limit=10, output_type="JSON", sampling="stratified", seed=7)
    ]


def test_structeval_stratified_sampling_distributes_remainder_evenly(tmp_path):
    fixture = tmp_path / "structeval.jsonl"
    _write_structeval_fixture(fixture)

    rows = kvq.load_structeval_rows(fixture, limit=7, output_type="JSON", sampling="stratified", seed=11)
    counts = list(kvq.structeval_selection_counts(rows).values())

    assert sum(counts) == 7
    assert max(counts) - min(counts) <= 1


def test_structeval_head_sampling_preserves_legacy_prefix(tmp_path):
    fixture = tmp_path / "structeval.jsonl"
    _write_structeval_fixture(fixture)

    rows = kvq.load_structeval_rows(fixture, limit=5, output_type="JSON", sampling="head", seed=42)

    assert [row["task_id"] for row in rows] == ["Text-0", "Text-1", "Text-2", "Text-3", "CSV-0"]


def test_structeval_all_format_loading_preserves_every_output_family(tmp_path):
    fixture = tmp_path / "structeval.jsonl"
    rows = [
        {
            "task_id": "json-task",
            "query": "Return JSON.",
            "input_type": "Text",
            "output_type": "JSON",
        },
        {
            "task_id": "svg-task",
            "query": "Return SVG.",
            "input_type": "Text",
            "output_type": "SVG",
            "rendering": True,
        },
    ]
    fixture.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    selected = kvq.load_structeval_rows(fixture, limit=2, output_type=None, sampling="head", seed=42)

    assert [row["task_id"] for row in selected] == ["json-task", "svg-task"]
    assert kvq.structeval_selection_counts(selected) == {"Text->JSON": 1, "Text->SVG": 1}


def test_unlimited_generation_uses_remaining_model_context():
    model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=128))

    assert kvq.resolve_generation_limit(model, input_length=28, requested_max_new_tokens=0) == 100
    assert kvq.resolve_generation_limit(model, input_length=28, requested_max_new_tokens=32) == 32


def test_official_summary_uses_four_unweighted_leaderboard_categories():
    rows = [
        {"task_id": "t-gen-a", "rendering": False, "input_type": "Text", "final_eval_score": 1.0},
        {"task_id": "t-gen-b", "rendering": False, "input_type": "Text", "final_eval_score": 0.0},
        {"task_id": "t-conv", "rendering": False, "input_type": "CSV", "final_eval_score": 0.25},
        {"task_id": "v-gen", "rendering": True, "input_type": "Text", "final_eval_score": 0.75},
        {"task_id": "v-conv", "rendering": True, "input_type": "HTML", "final_eval_score": 0.0},
    ]

    summary = official_summary.summarize_official_evaluation(rows)

    assert summary["categories"]["T-generation"]["score_percent"] == 50.0
    assert summary["categories"]["T-conversion"]["score_percent"] == 25.0
    assert summary["categories"]["V-generation"]["score_percent"] == 75.0
    assert summary["categories"]["V-conversion"]["score_percent"] == 0.0
    assert summary["official_unweighted_average_percent"] == 37.5
    assert summary["task_weighted_average_percent"] == 40.0
    assert summary["official_scope_complete"] is False


def test_official_inference_validator_rejects_duplicate_or_partial_scope():
    rows = [
        {
            "task_id": "duplicate",
            "input_type": "Text",
            "output_type": "JSON",
            "query": "Return JSON.",
            "generation": "{}",
            "rendering": False,
            "key_bits": 16,
            "value_bits": 16,
        },
        {
            "task_id": "duplicate",
            "input_type": "CSV",
            "output_type": "JSON",
            "query": "Convert CSV.",
            "generation": "{}",
            "rendering": False,
            "key_bits": 16,
            "value_bits": 16,
        },
    ]

    validation = artifact_check.validate_inference(
        rows,
        expected_key_bits=16,
        expected_value_bits=16,
    )

    assert validation["valid"] is False
    assert validation["dataset_rows"] == 2
    assert validation["unique_task_ids"] == 1
    assert validation["duplicate_task_ids"] == ["duplicate"]


def test_official_inference_validator_accepts_exact_full_scope():
    rows = []
    category_specs = (
        ("T-generation", 250, "Text", False),
        ("T-conversion", 700, "JSON", False),
        ("V-generation", 650, "Text", True),
        ("V-conversion", 435, "HTML", True),
    )
    for category, count, input_type, rendering in category_specs:
        for index in range(count):
            rows.append(
                {
                    "task_id": f"{category}-{index}",
                    "input_type": input_type,
                    "output_type": "SVG" if rendering else "JSON",
                    "query": "Generate the requested structure.",
                    "generation": "<|BEGIN_CODE|>{}<|END_CODE|>",
                    "rendering": rendering,
                    "key_bits": 8,
                    "value_bits": 4,
                }
            )

    validation = artifact_check.validate_inference(
        rows,
        expected_key_bits=8,
        expected_value_bits=4,
        expected_task_ids=[row["task_id"] for row in rows],
    )

    assert validation["valid"] is True
    assert validation["dataset_rows"] == 2035
    assert validation["unique_task_ids"] == 2035
    assert validation["exact_reference_task_order"] is True

    wrong_order = artifact_check.validate_inference(
        rows,
        expected_key_bits=8,
        expected_value_bits=4,
        expected_task_ids=[row["task_id"] for row in reversed(rows)],
    )
    assert wrong_order["valid"] is False
    assert wrong_order["exact_reference_task_order"] is False


def test_official_matrix_summary_preserves_four_category_scores(tmp_path):
    summaries = []
    for label, offset in (("FP16", 0.0), ("K8_V4", -1.0)):
        summary = {
            "dataset_rows": 2035,
            "scored_rows": 2035,
            "unscored_rows": 0,
            "official_scope_complete": True,
            "official_unweighted_average_percent": 60.0 + offset,
            "categories": {
                category: {
                    "count": official_summary.EXPECTED_CATEGORY_COUNTS[category],
                    "score_percent": 60.0 + offset,
                }
                for category in official_summary.CATEGORY_ORDER
            },
        }
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        summaries.append((label, path))

    matrix = matrix_summary.build_matrix(summaries)

    assert matrix["all_conditions_official_scope_complete"] is True
    assert [row["condition"] for row in matrix["conditions"]] == ["FP16", "K8_V4"]
    assert matrix["conditions"][1]["categories"]["V-conversion"] == 59.0


def test_poster_t_summary_uses_two_equal_category_means():
    rows = []
    for index in range(25):
        rows.append(
            {
                "task_id": f"t-gen-{index}",
                "input_type": "Text",
                "rendering": False,
                "final_eval_score": 0.8,
            }
        )
        rows.append(
            {
                "task_id": f"t-conv-{index}",
                "input_type": "JSON",
                "rendering": False,
                "final_eval_score": 0.6,
            }
        )

    summary = poster_summary.summarize_poster_evaluation(rows)

    assert summary["categories"]["T-generation"]["score_percent"] == 80.0
    assert summary["categories"]["T-conversion"]["score_percent"] == 60.0
    assert summary["t_category_average_percent"] == 70.0
    assert summary["t_pilot_complete"] is True
    assert summary["official_leaderboard_comparable"] is False


def test_text_dev_summary_accepts_ten_tasks_per_text_category():
    rows = []
    for index in range(10):
        rows.append(
            {
                "task_id": f"t-gen-{index}",
                "input_type": "Text",
                "rendering": False,
                "final_eval_score": 0.7,
            }
        )
        rows.append(
            {
                "task_id": f"t-conv-{index}",
                "input_type": "JSON",
                "rendering": False,
                "final_eval_score": 0.5,
            }
        )

    summary = poster_summary.summarize_poster_evaluation(rows, expected_category_count=10)

    assert summary["t_category_average_percent"] == 60.0
    assert summary["t_pilot_complete"] is True


def test_evaluator_smoke_fixture_covers_every_official_output_format():
    rows = [
        {"task_id": f"task-{index}", "output_type": output_type}
        for index, output_type in enumerate(evaluator_smoke.SMOKE_GENERATIONS)
    ]

    smoke_rows = evaluator_smoke.build_smoke_rows(rows)

    assert len(smoke_rows) == 18
    assert {row["output_type"] for row in smoke_rows} == set(evaluator_smoke.SMOKE_GENERATIONS)
    assert all(row["generation"].startswith("<|BEGIN_CODE|>") for row in smoke_rows)
    assert all(row["generation"].endswith("<|END_CODE|>") for row in smoke_rows)


def test_structeval_manifest_freezes_checked_task_order(tmp_path):
    source = tmp_path / "structeval.jsonl"
    _write_structeval_fixture(source)
    selected = kvq.load_structeval_rows(source, limit=10, output_type="JSON", sampling="stratified", seed=7)
    manifest = {
        "source_sha256": kvq._file_sha256(source),
        "sampling": "stratified",
        "seed": 7,
        "tasks": [
            {
                "selection_index": row["_structeval_selection_index"],
                "task_id": row["task_id"],
                "input_type": row["input_type"],
                "output_type": row["output_type"],
                "stratum": row["_structeval_stratum"],
            }
            for row in selected
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = kvq.load_structeval_manifest_rows(source, manifest_path)

    assert [row["task_id"] for row in loaded] == [row["task_id"] for row in selected]
    assert kvq.structeval_selection_counts(loaded) == {"CSV": 2, "TOML": 2, "Text": 2, "XML": 2, "YAML": 2}


def test_category_balanced_poster_manifest_covers_official_scope():
    source = Path(__file__).resolve().parents[1] / "toy_kv_experiments/data/structeval_full/structeval_test.jsonl"
    rows = manifest_creator.select_category_balanced_rows(
        manifest_creator.load_all_rows(source),
        limit=100,
        seed=42,
    )

    validation = poster_manifest_check.validate_rows(rows)

    assert validation["valid"] is True
    assert validation["category_counts"] == {
        "T-generation": 25,
        "T-conversion": 25,
        "V-generation": 25,
        "V-conversion": 25,
    }
    assert validation["task_type_count"] == 44
    assert validation["normalized_output_format_count"] == 18


def test_category_balanced_text_dev_manifest_uses_only_textual_categories():
    source = Path(__file__).resolve().parents[1] / "toy_kv_experiments/data/structeval_full/structeval_test.jsonl"
    rows = manifest_creator.select_category_balanced_rows(
        manifest_creator.load_all_rows(source),
        limit=20,
        seed=42,
        categories=("T-generation", "T-conversion"),
    )

    validation = text_dev_manifest_check.validate_rows(rows, expected_rows=20)

    assert validation["valid"] is True
    assert validation["category_counts"] == {"T-generation": 10, "T-conversion": 10}
    assert validation["rendering_task_ids"] == []


def test_complete_structeval_t_manifest_covers_every_non_rendered_task():
    source = Path(__file__).resolve().parents[1] / "toy_kv_experiments/data/structeval_full/structeval_test.jsonl"
    source_rows = manifest_creator.load_all_rows(source)
    rows = manifest_creator.select_complete_category_rows(
        source_rows,
        seed=42,
        categories=("T-generation", "T-conversion"),
    )

    validation = structeval_t_manifest_check.validate_rows(
        rows,
        source_rows=source_rows,
    )

    assert validation["valid"] is True
    assert validation["rows"] == 950
    assert validation["category_counts"] == {
        "T-generation": 250,
        "T-conversion": 700,
    }
    assert validation["rendering_task_ids"] == []


def test_result_summarizer_splits_multiple_bit_settings_in_one_file(tmp_path):
    result_path = tmp_path / "mixed.json"
    rows = []
    for bits in (4, 8):
        for task_id in ("task-a", "task-b"):
            rows.append(
                {
                    "task_id": task_id,
                    "input_type": "Text",
                    "output_type": "JSON",
                    "bits": bits,
                    "key_bits": bits,
                    "value_bits": bits,
                    "cache_quantization_mode": "real-blockwise",
                    "residual_length": 32,
                    "kv_group_size": 32,
                    "json_parse_success": True,
                    "json_required_path_rate": 1.0,
                    "structeval_final_eval_score": 1.0,
                }
            )
    result_path.write_text(json.dumps(rows), encoding="utf-8")

    summaries = result_summary.summarize_file(result_path)

    assert len(summaries) == 2
    assert {summary["key_bits"] for summary in summaries} == {"4", "8"}
    assert {summary["n"] for summary in summaries} == {2}


def test_paired_analysis_requires_matched_ids_and_isolates_induced_failures():
    fp16 = [
        {"task_id": "a", "input_type": "Text", "output_type": "JSON", "json_parse_success": True, "json_required_path_rate": 1.0},
        {"task_id": "b", "input_type": "CSV", "output_type": "JSON", "json_parse_success": False, "json_required_path_rate": 0.0},
    ]
    candidate = [
        {"task_id": "a", "input_type": "Text", "output_type": "JSON", "json_parse_success": False, "json_required_path_rate": 0.0},
        {"task_id": "b", "input_type": "CSV", "output_type": "JSON", "json_parse_success": False, "json_required_path_rate": 0.0},
    ]

    pairs = paired_analysis.assert_paired(fp16, candidate)
    summary = paired_analysis.pair_summary(pairs, bootstrap_samples=100, seed=7)

    assert summary["compression_induced_parse_failures"] == 1
    assert summary["compression_induced_parse_failure_rate_given_fp16_success"] == 1.0
    assert summary["parse_mcnemar_pvalue"] == 1.0


def test_paired_analysis_reports_fp16_reference_leaf_value_agreement():
    fp16 = {
        "task_id": "a",
        "json_parse_success": True,
        "json_candidate": '{"user":{"name":"Alice","age":7}}',
    }
    candidate = {
        "task_id": "a",
        "json_parse_success": True,
        "json_candidate": '{"user":{"name":"Alice","age":8}}',
    }

    agreement = paired_analysis.fp16_reference_semantic_agreement(fp16, candidate)

    assert agreement is not None
    assert agreement["fp16_reference_common_path_value_accuracy"] == 0.5
    assert agreement["fp16_reference_leaf_pair_f1"] == 0.5
    assert agreement["fp16_reference_exact_json_match"] is False


def test_structeval_runner_checkpoints_each_task_and_resumes_without_inference(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=1,
            num_attention_heads=1,
            hidden_size=4,
        )
    )

    def completed_generation(*args, **kwargs):
        return {
            "text": '{"name":"Alice"}',
            "key_bits": 16,
            "value_bits": 16,
            "input_tokens": 4,
            "generated_tokens": 5,
            "generation_limit": 8,
            "stop_reason": "eos",
            "protected_bits": None,
            "target_average_bits": None,
            "protection_budget_model": "token-count-v1",
            "protection_budget_estimated_storage_bytes": 0,
            "protection_budget_target_storage_bytes": 0,
            "protection_budget_order": "prefix",
            "protection_target": "both",
            "constraint_terms": [],
            "constraint_term_weights": {},
            "cache_quantization_mode": "repeated",
            "quantize_block_size": 1,
            "kv_group_size": None,
            "cache_algorithm_revision": "not-applicable",
            "prefill_cache_policy": "mode_dependent",
            "cache_sequence_length": 9,
            "cache_dtype_bytes": 4,
            "cache_storage_semantics": "full-precision-dynamic-cache",
            "persistent_cache_storage_bytes": 288,
            "persistent_cache_compression_ratio_vs_fp16": 0.5,
            "persistent_cache_effective_bits_per_scalar": 32.0,
            "mean_quantized_prefix_length": 0.0,
            "real_cache_storage_bytes": 0,
            "real_cache_storage_breakdown": {},
            "real_cache_mean_quantized_lengths": {},
            "fp16_equivalent_cache_bytes": 0,
            "real_cache_compression_ratio_vs_fp16": 0.0,
            "real_cache_effective_bits_per_scalar": 0.0,
            "protected_cache_positions": 0,
            "protected_cache_fraction": 0.0,
            "candidate_protected_cache_positions": 0,
            "candidate_protected_cache_fraction": 0.0,
            "protected_score_sum": 0.0,
            "candidate_score_sum": 0.0,
            "protected_score_fraction": 0.0,
        }

    monkeypatch.setattr(kvq, "generate_manual_kv", completed_generation)
    rows = [
        {
            "task_id": "task-a",
            "task_name": "Text to JSON",
            "input_type": "Text",
            "output_type": "JSON",
            "query": "Return name.",
            "raw_output_metric": ["name"],
            "_structeval_selection_index": 0,
        }
    ]
    metadata = {
        "run_id": "run-a",
        "run_fingerprint": "fingerprint-a",
        "combined_source_sha256": "source-a",
        "environment": {"device_name": "cpu", "torch": "test", "transformers": "test"},
        "model": {"config_sha256": "model-a"},
    }
    checkpoint_dir = tmp_path / "checkpoints"

    first = kvq.run_structeval_smoke(
        model,
        Tokenizer(),
        "cpu",
        rows=rows,
        bits_list=[16],
        max_new_tokens=8,
        show_progress=False,
        checkpoint_dir=checkpoint_dir,
        run_metadata=metadata,
    )

    assert len(first) == 1
    assert first[0]["run_fingerprint"] == "fingerprint-a"
    assert first[0]["generation"] == '{"name":"Alice"}'
    assert first[0]["query"] == "Return name."
    assert first[0]["raw_output_metric"] == ["name"]
    assert first[0]["rendering"] is False
    assert len(list((checkpoint_dir / "tasks").glob("*.json"))) == 1
    assert json.loads((checkpoint_dir / "status.json").read_text())["status"] == "complete"

    monkeypatch.setattr(
        kvq,
        "generate_manual_kv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resume reran inference")),
    )
    resumed = kvq.run_structeval_smoke(
        model,
        Tokenizer(),
        "cpu",
        rows=rows,
        bits_list=[16],
        max_new_tokens=8,
        show_progress=False,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        run_metadata=metadata,
    )

    assert resumed == first


def test_fake_quantize_cache_tensor_preserves_recent_residual_positions():
    x = kvq.torch.tensor(
        [[[[0.10, 0.20], [1.30, -0.70], [0.33, 0.44], [2.20, -1.10]]]],
        dtype=kvq.torch.float32,
    )

    quantized = kvq.fake_quantize_cache_tensor(
        x,
        bits=2,
        scale_dim=-1,
        residual_length=2,
    )

    assert not kvq.torch.equal(quantized[..., :2, :], x[..., :2, :])
    assert kvq.torch.equal(quantized[..., -2:, :], x[..., -2:, :])


def test_fake_quantize_cache_tensor_all_residual_returns_original_values():
    x = kvq.torch.randn(1, 2, 3, 4)

    quantized = kvq.fake_quantize_cache_tensor(
        x,
        bits=2,
        scale_dim=-1,
        residual_length=99,
    )

    assert kvq.torch.equal(quantized, x)


def test_fake_quantize_cache_tensor_preserves_arbitrary_protected_positions():
    x = kvq.torch.tensor(
        [[[[0.10, 0.20], [1.30, -0.70], [0.33, 0.44], [2.20, -1.10]]]],
        dtype=kvq.torch.float32,
    )
    protected = kvq.torch.tensor([False, True, False, True])

    quantized = kvq.fake_quantize_cache_tensor(
        x,
        bits=2,
        scale_dim=-1,
        protected_positions=protected,
    )

    assert kvq.torch.equal(quantized[..., 1, :], x[..., 1, :])
    assert kvq.torch.equal(quantized[..., 3, :], x[..., 3, :])
    assert not kvq.torch.equal(quantized[..., 0, :], x[..., 0, :])


def test_fake_quantize_cache_tensor_can_use_lower_protected_bits():
    x = kvq.torch.tensor(
        [[[[0.10, 0.20], [0.33, 0.44]]]],
        dtype=kvq.torch.float32,
    )
    protected = kvq.torch.tensor([False, True])

    quantized = kvq.fake_quantize_cache_tensor(
        x,
        bits=2,
        scale_dim=-1,
        protected_positions=protected,
        protected_bits=8,
    )

    base_expected = kvq.fake_quantize_symmetric(x, bits=2, dim=-1)
    protected_expected = kvq.fake_quantize_symmetric(x, bits=8, dim=-1)
    assert kvq.torch.equal(quantized[..., 0, :], base_expected[..., 0, :])
    assert kvq.torch.equal(quantized[..., 1, :], protected_expected[..., 1, :])
    assert not kvq.torch.equal(quantized[..., 1, :], x[..., 1, :])


def test_fake_quantize_cache_tensor_keeps_recent_residual_full_precision_with_protected_bits():
    x = kvq.torch.tensor(
        [[[[0.10, 0.20], [0.33, 0.44], [1.30, -0.70]]]],
        dtype=kvq.torch.float32,
    )
    protected = kvq.torch.tensor([True, False, False])

    quantized = kvq.fake_quantize_cache_tensor(
        x,
        bits=2,
        scale_dim=-1,
        residual_length=1,
        protected_positions=protected,
        protected_bits=8,
    )

    protected_expected = kvq.fake_quantize_symmetric(x, bits=8, dim=-1)
    assert kvq.torch.equal(quantized[..., 0, :], protected_expected[..., 0, :])
    assert kvq.torch.equal(quantized[..., -1, :], x[..., -1, :])


def test_structure_token_mask_marks_json_syntax_tokens():
    class ToyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            vocab = {
                1: "{",
                2: '"name"',
                3: "Alice",
                4: "<|BEGIN_CODE|>",
                5: "plain",
            }
            return vocab[int(ids[0])]

    token_ids = kvq.torch.tensor([1, 2, 3, 4, 5])

    mask = kvq.structure_token_mask(ToyTokenizer(), token_ids, mode="json-syntax")

    assert mask.tolist() == [True, True, False, True, False]


def test_structure_token_mask_all_marks_every_position():
    class ToyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return str(ids[0])

    mask = kvq.structure_token_mask(
        ToyTokenizer(),
        kvq.torch.tensor([1, 2, 3]),
        mode="all",
    )

    assert mask.tolist() == [True, True, True]


def test_constraint_terms_from_structeval_paths_split_schema_keys():
    terms = kvq.constraint_terms_from_required_paths(
        [
            "novel.title",
            "novel.author.birth_year",
            "novel.characters[0].name",
            "novel.characters.*.role",
        ]
    )

    assert terms == [
        "author",
        "birth",
        "birth_year",
        "characters",
        "name",
        "novel",
        "role",
        "title",
        "year",
    ]


def test_constraint_term_weights_prioritize_leaf_schema_keys():
    weights = kvq.constraint_term_weights_from_required_paths(
        [
            "novel.author.birth_year",
            "novel.publication.year",
        ]
    )

    assert weights["birth_year"] > weights["author"]
    assert weights["year"] > weights["publication"]
    assert weights["novel"] < weights["birth_year"]


def test_structure_token_mask_marks_constraint_path_terms():
    class ToyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            vocab = {
                1: "ordinary",
                2: '"birth',
                3: "_year",
                4: '"Alice"',
                5: "author",
            }
            return vocab[int(ids[0])]

    token_ids = kvq.torch.tensor([1, 2, 3, 4, 5])

    mask = kvq.structure_token_mask(
        ToyTokenizer(),
        token_ids,
        mode="constraint-paths",
        constraint_terms=["birth_year", "author"],
    )

    assert mask.tolist() == [False, True, True, False, True]


def test_prompt_visible_protection_does_not_read_evaluator_paths():
    prompt = """Convert CSV to JSON.\n<code>mission_name,launch.date,crew[0].name\nA,2025,B</code>"""

    prompt_terms, _ = kvq.resolve_protection_signals(
        prompt,
        required_items=["hidden.reference_only"],
        source="prompt-visible",
    )
    oracle_terms, _ = kvq.resolve_protection_signals(
        prompt,
        required_items=["hidden.reference_only"],
        source="oracle-required-paths",
    )

    assert {"mission", "name", "launch", "date", "crew"}.issubset(set(prompt_terms))
    assert "hidden" not in prompt_terms
    assert "hidden" in oracle_terms


def test_structure_token_scores_mark_leaf_terms_as_more_important():
    class ToyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            vocab = {
                1: "novel",
                2: "author",
                3: "birth_year",
                4: "ordinary",
            }
            return vocab[int(ids[0])]

    token_ids = kvq.torch.tensor([1, 2, 3, 4])
    weights = kvq.constraint_term_weights_from_required_paths(["novel.author.birth_year"])

    scores = kvq.structure_token_scores(
        ToyTokenizer(),
        token_ids,
        mode="constraint-paths",
        constraint_term_weights=weights,
    )

    assert scores[2] > scores[1] > scores[0] > scores[3]


def test_budgeted_protection_mask_limits_older_positions_and_preserves_residual_budget():
    candidate = kvq.torch.tensor([True, True, True, True, True])

    selected = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="prefix",
    )

    assert selected.tolist() == [True, True, False, False, False]


def test_budgeted_protection_mask_can_prioritize_high_scores():
    candidate = kvq.torch.tensor([True, True, True, True, True])
    scores = kvq.torch.tensor([1.0, 5.0, 2.0, 4.0, 100.0])

    selected = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="score",
        scores=scores,
    )

    assert selected.tolist() == [False, True, False, True, False]


def test_budgeted_protection_mask_can_prioritize_recent_older_positions():
    candidate = kvq.torch.tensor([True, True, True, True, True])

    selected = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="recent",
    )

    assert selected.tolist() == [False, False, True, True, False]


def test_budgeted_protection_mask_random_order_is_seeded():
    candidate = kvq.torch.tensor([True, True, True, True, True, True])

    first = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="random",
        random_seed=17,
    )
    second = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="random",
        random_seed=17,
    )
    different = kvq.budgeted_protection_mask(
        candidate,
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        target_average_bits=8.0,
        order="random",
        random_seed=18,
    )

    assert first.tolist() == second.tolist()
    assert first.tolist() != different.tolist()


def test_group_aware_budget_charges_key_protection_once_per_full_group():
    profile = kvq.RealCacheStorageBudgetProfile(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype_bytes=2,
        key_bits=4,
        value_bits=4,
        protected_bits=8,
        residual_length=4,
        group_size=4,
        protection_target="keys",
    )
    candidates = kvq.torch.tensor([True, True, True, True, True, True, True, True])
    baseline = kvq.real_cache_storage_bytes_for_mask(profile, seq_len=8)
    one_group = candidates.new_tensor([True, False, False, False, False, False, False, False])
    one_group_bytes = kvq.real_cache_storage_bytes_for_mask(profile, seq_len=8, selected_mask=one_group)
    scalar_count = 2 * 8 * profile.num_layers * profile.num_kv_heads * profile.head_dim
    target_average_bits = (one_group_bytes * 8) / scalar_count

    selected = kvq.budgeted_protection_mask(
        candidates,
        base_bits=4,
        protected_bits=8,
        residual_length=4,
        target_average_bits=target_average_bits,
        order="prefix",
        storage_profile=profile,
    )

    assert one_group_bytes > baseline
    assert selected.tolist() == [True, False, False, False, False, False, False, False]
    assert kvq.real_cache_storage_bytes_for_mask(profile, 8, selected) == one_group_bytes
    assert kvq.real_cache_storage_bytes_for_mask(profile, 8, selected) <= kvq.real_cache_target_storage_bytes(
        profile, 8, target_average_bits
    )


def test_group_aware_budget_keeps_value_protection_token_granular():
    profile = kvq.RealCacheStorageBudgetProfile(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype_bytes=2,
        key_bits=4,
        value_bits=4,
        protected_bits=8,
        residual_length=4,
        group_size=4,
        protection_target="values",
    )
    candidates = kvq.torch.ones(8, dtype=kvq.torch.bool)
    two_values = candidates.new_tensor([True, True, False, False, False, False, False, False])
    target_bytes = kvq.real_cache_storage_bytes_for_mask(profile, 8, two_values)
    scalar_count = 2 * 8 * profile.num_layers * profile.num_kv_heads * profile.head_dim

    selected = kvq.budgeted_protection_mask(
        candidates,
        base_bits=4,
        protected_bits=8,
        residual_length=4,
        target_average_bits=(target_bytes * 8) / scalar_count,
        order="prefix",
        storage_profile=profile,
    )

    assert selected.tolist() == [True, True, False, False, False, False, False, False]


def test_schema_path_uep_budget_keeps_residual_fp16():
    avg_bits = uep_budget.effective_bits(
        token_count=100,
        protected_older_count=10,
        residual_length=20,
        base_bits=4,
        protected_bits=8,
    )

    assert avg_bits == 6.8


def test_schema_path_uep_budget_analyzes_prompt_protection():
    class ToyTokenizer:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": kvq.torch.tensor([[1, 2, 3, 4, 5]])}

        def decode(self, ids, skip_special_tokens=False):
            vocab = {
                1: "Please",
                2: "novel",
                3: "author",
                4: "ordinary",
                5: "<|BEGIN_CODE|>",
            }
            return vocab[int(ids[0])]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

    report = uep_budget.analyze_rows(
        tokenizer=ToyTokenizer(),
        rows=[
            {
                "task_id": "toy",
                "task_name": "Toy JSON",
                "query": "Return novel.author.",
                "raw_output_metric": ["novel.author.name"],
            }
        ],
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        protection_mode="constraint-paths",
        target_average_bits=None,
        protection_budget_order="prefix",
        official_prompt=False,
        official_model_name="",
    )

    row = report["per_task"][0]
    assert row["protected_tokens"] == 3
    assert row["residual_tokens"] == 1
    assert row["protected_older_tokens"] == 2
    assert row["estimated_average_bits_per_kv_scalar"] == 8.0


def test_schema_path_uep_budget_score_order_keeps_leaf_score_under_budget():
    class ToyTokenizer:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": kvq.torch.tensor([[1, 2, 3, 4, 5]])}

        def decode(self, ids, skip_special_tokens=False):
            vocab = {
                1: "novel",
                2: "author",
                3: "birth_year",
                4: "ordinary",
                5: "<|BEGIN_CODE|>",
            }
            return vocab[int(ids[0])]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

    report = uep_budget.analyze_rows(
        tokenizer=ToyTokenizer(),
        rows=[
            {
                "task_id": "toy",
                "task_name": "Toy JSON",
                "query": "Return novel.author.birth_year.",
                "raw_output_metric": ["novel.author.birth_year"],
            }
        ],
        base_bits=4,
        protected_bits=8,
        residual_length=1,
        protection_mode="constraint-paths",
        target_average_bits=7.2,
        protection_budget_order="score",
        official_prompt=False,
        official_model_name="",
    )

    row = report["per_task"][0]
    assert row["protected_older_tokens"] == 1
    assert row["protected_score_fraction"] > 0.5


def test_blockwise_cache_quantization_only_quantizes_newly_evicted_positions():
    x = kvq.torch.tensor(
        [[[[0.10, 0.20], [1.30, -0.70], [0.33, 0.44], [2.20, -1.10], [0.80, -0.20]]]],
        dtype=kvq.torch.float32,
    )

    quantized, new_prefix = kvq.fake_quantize_cache_tensor_blockwise(
        x,
        bits=2,
        scale_dim=-1,
        quantized_prefix_length=1,
        residual_length=2,
    )

    expected_middle = kvq.fake_quantize_symmetric(x[..., 1:3, :], bits=2, dim=-1)
    assert new_prefix == 3
    assert kvq.torch.equal(quantized[..., :1, :], x[..., :1, :])
    assert kvq.torch.equal(quantized[..., 1:3, :], expected_middle)
    assert kvq.torch.equal(quantized[..., 3:, :], x[..., 3:, :])


def test_blockwise_cache_quantization_waits_until_a_full_block_is_evictable():
    x = kvq.torch.randn(1, 1, 5, 2)

    quantized, new_prefix = kvq.fake_quantize_cache_tensor_blockwise(
        x,
        bits=2,
        scale_dim=-1,
        quantized_prefix_length=0,
        residual_length=2,
        block_size=4,
    )

    assert new_prefix == 0
    assert kvq.torch.equal(quantized, x)


def test_dynamic_cache_quantization_supports_separate_key_value_bits_repeated():
    class Layer:
        is_initialized = True

        def __init__(self):
            self.keys = kvq.torch.tensor([[[[0.10, 0.20], [1.30, -0.70]]]], dtype=kvq.torch.float32)
            self.values = kvq.torch.tensor([[[[0.25, 0.50], [1.70, -0.30]]]], dtype=kvq.torch.float32)

    class Cache:
        def __init__(self):
            self.layers = [Layer()]

    cache = Cache()
    original_keys = cache.layers[0].keys.clone()
    original_values = cache.layers[0].values.clone()

    kvq.quantize_dynamic_cache_(
        cache,
        bits=4,
        key_bits=16,
        value_bits=2,
        key_axis="per-token",
        value_axis="per-token",
    )

    assert kvq.torch.equal(cache.layers[0].keys, original_keys)
    assert not kvq.torch.equal(cache.layers[0].values, original_values)


def test_dynamic_cache_quantization_supports_separate_key_value_bits_blockwise():
    class Layer:
        is_initialized = True

        def __init__(self):
            self.keys = kvq.torch.tensor(
                [[[[0.10, 0.20], [1.30, -0.70], [0.33, 0.44], [2.20, -1.10]]]],
                dtype=kvq.torch.float32,
            )
            self.values = kvq.torch.tensor(
                [[[[0.25, 0.50], [1.70, -0.30], [0.91, -0.14], [0.42, 0.38]]]],
                dtype=kvq.torch.float32,
            )

    class Cache:
        def __init__(self):
            self.layers = [Layer()]

    cache = Cache()
    original_keys = cache.layers[0].keys.clone()
    original_values = cache.layers[0].values.clone()
    prefixes: list[int] = []

    kvq.quantize_dynamic_cache_(
        cache,
        bits=4,
        key_bits=2,
        value_bits=16,
        key_axis="per-token",
        value_axis="per-token",
        residual_length=1,
        cache_quantization_mode="blockwise",
        quantized_prefix_lengths=prefixes,
        quantize_block_size=1,
    )

    assert prefixes == [3]
    assert not kvq.torch.equal(cache.layers[0].keys[..., :3, :], original_keys[..., :3, :])
    assert kvq.torch.equal(cache.layers[0].keys[..., -1:, :], original_keys[..., -1:, :])
    assert kvq.torch.equal(cache.layers[0].values, original_values)


def test_real_quantized_block_packs_4bit_values():
    x = kvq.torch.tensor([[[[0.10, 0.20], [1.30, -0.70]]]], dtype=kvq.torch.float32)

    block = kvq.real_quantized_cache.quantize_to_real_block(x, bits=4, scale_dim=-1)
    dequantized = block.dequantize()

    assert block.data.dtype == kvq.torch.uint8
    assert block.data.numel() == x.numel() // 2
    assert block.minimum.numel() > 0
    assert dequantized.shape == x.shape


def test_real_quantized_block_uses_all_four_affine_int2_codes():
    x = kvq.torch.tensor([[[[0.0, 1.0, 2.0, 3.0]]]], dtype=kvq.torch.float32)

    block = kvq.real_quantized_cache.quantize_to_real_block(
        x,
        bits=2,
        layout="values",
        group_size=4,
    )

    assert sorted(block.data.tolist()) == [0b11100100]
    assert kvq.torch.equal(block.dequantize(), x)


def test_real_quantized_block_exactly_represents_constant_affine_groups():
    x = kvq.torch.full((1, 1, 4, 4), 3.5, dtype=kvq.torch.float32)

    block = kvq.real_quantized_cache.quantize_to_real_block(
        x,
        bits=2,
        layout="keys",
        group_size=2,
    )

    assert kvq.torch.equal(block.dequantize(), x)


def test_real_quantized_cache_uses_full_precision_prefill_and_kivi_residuals():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(
            bits=4,
            residual_length=2,
            block_size=2,
        )
    )
    keys = kvq.torch.randn(1, 1, 5, 4)
    values = kvq.torch.randn(1, 1, 5, 4)

    out_keys, out_values = cache.update(keys, values, layer_idx=0)
    layer = cache.layers[0]

    assert kvq.torch.equal(out_keys, keys)
    assert kvq.torch.equal(out_values, values)
    assert layer.key_quantized_length == 4
    assert layer.key_residual.shape[-2] == 1
    assert layer.value_quantized_length == 3
    assert layer.value_residual.shape[-2] == 2
    assert layer.key_blocks[0].data.dtype == kvq.torch.uint8
    assert cache.get_seq_length() == 5
    assert cache.get_mask_sizes(query_length=1, layer_idx=0) == (6, 0)

    next_keys = kvq.torch.randn(1, 1, 1, 4)
    next_values = kvq.torch.randn(1, 1, 1, 4)
    decoded_keys, decoded_values = cache.update(next_keys, next_values, layer_idx=0)

    assert decoded_keys.shape[-2] == 6
    assert decoded_values.shape[-2] == 6
    assert layer.key_quantized_length == 4
    assert layer.key_residual.shape[-2] == 2
    assert layer.value_quantized_length == 3
    assert layer.value_residual.shape[-2] == 3

    cache.finalize_after_attention()

    assert layer.key_quantized_length == 6
    assert layer.key_residual.shape[-2] == 0
    assert layer.value_quantized_length == 4
    assert layer.value_residual.shape[-2] == 2


def test_real_quantized_cache_seals_residuals_only_after_decode_attention():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(
            bits=2,
            residual_length=4,
            block_size=4,
        )
    )
    prefill_keys = kvq.torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.8, 0.9]]]]
    )
    prefill_values = kvq.torch.tensor(
        [[[[0.0, 0.2, 0.7, 1.0], [0.1, 0.4, 0.6, 0.8], [0.2, 0.5, 0.9, 1.1]]]]
    )
    cache.update(prefill_keys, prefill_values, layer_idx=0)
    layer = cache.layers[0]

    fourth_key = kvq.torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    fourth_value = fourth_key.clone()
    original_four_keys = kvq.torch.cat([prefill_keys, fourth_key], dim=-2)
    decoded_keys, decoded_values = cache.update(fourth_key, fourth_value, layer_idx=0)

    assert layer.key_quantized_length == 0
    assert layer.key_residual.shape[-2] == 4
    assert kvq.torch.equal(decoded_keys, original_four_keys)
    assert layer.value_quantized_length == 0
    assert layer.value_residual.shape[-2] == 4
    assert kvq.torch.equal(decoded_values, kvq.torch.cat([prefill_values, fourth_value], dim=-2))

    cache.finalize_after_attention()

    assert layer.key_quantized_length == 4
    assert layer.key_residual.shape[-2] == 0
    assert not kvq.torch.equal(layer.dequantized_keys(), original_four_keys)

    fifth_key = kvq.torch.tensor([[[[1.2, 1.2, 1.2, 1.2]]]])
    fifth_value = kvq.torch.tensor([[[[1.3, 1.4, 1.5, 1.6]]]])
    original_five_values = kvq.torch.cat([prefill_values, fourth_value, fifth_value], dim=-2)
    _, decoded_values = cache.update(fifth_key, fifth_value, layer_idx=0)

    assert layer.value_quantized_length == 0
    assert layer.value_residual.shape[-2] == 5
    assert kvq.torch.equal(decoded_values, original_five_values)

    cache.finalize_after_attention()

    assert layer.value_quantized_length == 1
    assert layer.value_residual.shape[-2] == 4
    assert not kvq.torch.equal(layer.dequantized_values(), original_five_values)


def test_real_quantized_cache_rejects_decode_without_attention_finalize():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(bits=2, residual_length=2, block_size=2)
    )
    keys = kvq.torch.randn(1, 1, 1, 2)
    values = kvq.torch.randn(1, 1, 1, 2)
    cache.update(keys, values, layer_idx=0)
    cache.update(keys, values, layer_idx=0)

    with pytest.raises(RuntimeError, match="finalize_after_attention"):
        cache.update(keys, values, layer_idx=0)


def test_real_quantized_cache_matches_explicit_kivi_reference_trace():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(bits=2, residual_length=4, group_size=4)
    )
    prefill_keys = kvq.torch.tensor(
        [[[[0.0, 0.1, 0.2, 0.3], [0.2, 0.4, 0.6, 0.8], [0.7, 0.8, 0.9, 1.0]]]]
    )
    prefill_values = kvq.torch.tensor(
        [[[[0.0, 0.2, 0.7, 1.0], [0.1, 0.4, 0.6, 0.8], [0.2, 0.5, 0.9, 1.1]]]]
    )
    cache.update(prefill_keys, prefill_values, layer_idx=0)

    fourth_key = kvq.torch.tensor([[[[1.2, 1.3, 1.4, 1.5]]]])
    fourth_value = kvq.torch.tensor([[[[1.3, 1.4, 1.5, 1.6]]]])
    full_four_keys = kvq.torch.cat([prefill_keys, fourth_key], dim=-2)
    full_four_values = kvq.torch.cat([prefill_values, fourth_value], dim=-2)
    attention_keys, attention_values = cache.update(fourth_key, fourth_value, layer_idx=0)

    assert kvq.torch.equal(attention_keys, full_four_keys)
    assert kvq.torch.equal(attention_values, full_four_values)

    cache.finalize_after_attention()
    layer = cache.layers[0]
    reference_key_block = kvq.real_quantized_cache.quantize_to_real_block(
        full_four_keys,
        bits=2,
        layout="keys",
        group_size=4,
    )
    assert kvq.torch.equal(layer.key_blocks[0].data, reference_key_block.data)
    assert kvq.torch.equal(layer.key_blocks[0].scale, reference_key_block.scale)
    assert kvq.torch.equal(layer.key_blocks[0].minimum, reference_key_block.minimum)
    assert layer.value_blocks == []
    assert kvq.torch.equal(layer.value_residual, full_four_values)

    fifth_key = kvq.torch.tensor([[[[1.6, 1.7, 1.8, 1.9]]]])
    fifth_value = kvq.torch.tensor([[[[1.7, 1.8, 1.9, 2.0]]]])
    attention_keys, attention_values = cache.update(fifth_key, fifth_value, layer_idx=0)
    expected_attention_keys = kvq.torch.cat([reference_key_block.dequantize(), fifth_key], dim=-2)
    expected_attention_values = kvq.torch.cat([full_four_values, fifth_value], dim=-2)

    assert kvq.torch.equal(attention_keys, expected_attention_keys)
    assert kvq.torch.equal(attention_values, expected_attention_values)

    cache.finalize_after_attention()
    reference_value_block = kvq.real_quantized_cache.quantize_to_real_block(
        prefill_values[..., :1, :],
        bits=2,
        layout="values",
        group_size=4,
    )
    assert kvq.torch.equal(layer.value_blocks[0].data, reference_value_block.data)
    assert kvq.torch.equal(layer.value_blocks[0].scale, reference_value_block.scale)
    assert kvq.torch.equal(layer.value_blocks[0].minimum, reference_value_block.minimum)


def test_qwen_logits_use_full_precision_current_residual_at_key_boundary():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    kvq.torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
    ).eval()
    tokens = kvq.torch.tensor([[3, 5, 7, 11]])

    with kvq.torch.no_grad():
        full_logits = model(input_ids=tokens, use_cache=False, return_dict=True).logits[:, -1, :]
        cache = kvq.RealQuantizedCache(
            kvq.RealQuantizedCacheConfig(bits=2, residual_length=4, group_size=4)
        )
        model(input_ids=tokens[:, :3], past_key_values=cache, use_cache=True, return_dict=True)
        boundary = model(input_ids=tokens[:, 3:], past_key_values=cache, use_cache=True, return_dict=True)

    assert kvq.torch.allclose(boundary.logits[:, -1, :], full_logits, atol=1e-5, rtol=1e-5)
    assert all(layer.key_quantized_length == 0 for layer in cache.layers)
    assert all(layer.key_residual.shape[-2] == 4 for layer in cache.layers)

    cache.finalize_after_attention()
    assert all(layer.key_quantized_length == 4 for layer in cache.layers)
    assert all(layer.key_residual.shape[-2] == 0 for layer in cache.layers)


def test_result_summary_separates_legacy_real_cache_from_fp16_rows():
    legacy = {
        "quantize_cache": True,
        "cache_quantization_mode": "real-blockwise",
    }
    fp16 = {
        "quantize_cache": False,
        "cache_quantization_mode": "real-blockwise",
    }

    assert result_summary._value(legacy, "cache_algorithm_revision") == '"legacy-materialize-before-seal-v1"'
    assert result_summary._value(fp16, "cache_algorithm_revision") == '"not-applicable"'


def test_real_quantized_cache_uses_protected_bits_for_selected_positions():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(
            bits=4,
            residual_length=1,
            block_size=1,
            protected_bits=8,
        )
    )
    cache.set_protected_positions(kvq.torch.tensor([False, True, False]))
    keys = kvq.torch.randn(1, 1, 3, 4)
    values = kvq.torch.randn(1, 1, 3, 4)

    cache.update(keys, values, layer_idx=0)
    layer = cache.layers[0]

    assert [block.bits for block in layer.key_blocks] == [4, 8, 4]
    assert [block.bits for block in layer.value_blocks] == [4, 8]


def test_real_quantized_cache_respects_mixed_key_value_bits():
    cache = kvq.RealQuantizedCache(
        kvq.RealQuantizedCacheConfig(
            bits=8,
            key_bits=8,
            value_bits=4,
            residual_length=2,
            block_size=2,
        )
    )
    keys = kvq.torch.randn(1, 1, 5, 4)
    values = kvq.torch.randn(1, 1, 5, 4)

    cache.update(keys, values, layer_idx=0)
    layer = cache.layers[0]

    assert {block.bits for block in layer.key_blocks} == {8}
    assert {block.bits for block in layer.value_blocks} == {4}
