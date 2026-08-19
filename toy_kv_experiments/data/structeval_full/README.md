# StructEval Full Local Copy

- Dataset: `TIGER-Lab/StructEval`
- Config: `default`
- Split: `test`
- Rows: `2035`

## Files

- `structeval_test.parquet`: original Hugging Face converted parquet shard
- `structeval_test.jsonl`: full dataset as JSONL
- `structeval_test.csv`: full dataset as CSV
- `preview_first_20.json`: compact first-20-row preview
- `examples_by_output_type.md`: one compact example per output type
- `summary.json`: counts by input/output/task type

## Columns

- `task_id`
- `query`
- `feature_requirements`
- `task_name`
- `input_type`
- `output_type`
- `query_example`
- `VQA`
- `raw_output_metric`
- `rendering`

## Research Use

Use this as an evaluation benchmark for structured-output generation, not as the main toy training dataset.
