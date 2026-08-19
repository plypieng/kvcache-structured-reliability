# StructEval Official Protocol Notes

Local upstream checkout:

```text
third_party/StructEval
```

Source repository:

```text
https://github.com/TIGER-AI-Lab/StructEval
```

## Repository Pull

The repository was cloned as a shallow, blob-filtered checkout because the normal full clone stalled during object download.

Current checkout:

```text
third_party/StructEval
```

Commit observed:

```text
788a40c Update README.md
```

## Official Pipeline

StructEval uses a three-stage pipeline:

1. Inference
2. Render or extract generated output
3. Evaluate extracted output

The CLI entry point is:

```text
third_party/StructEval/structeval/cli.py
```

The official inference command is documented as:

```bash
python -m structeval.cli inference \
  --llm_model_name "model_name" \
  --llm_engine "engine_name" \
  --input_path "path/to/input.json" \
  --output_path "path/to/output.json"
```

## Official Inference Prompt Wrapper

The official CLI does not send the raw dataset query directly. It appends this instruction:

```text
IMPORTANT: Only output the required output format. You must start the format/code with <|BEGIN_CODE|> and end the format/code with <|END_CODE|>. No other text output (explanation, comments, etc.) are allowed. Do not use markdown code fences.
```

This is important for our experiments. Our current notebook sends the StructEval query directly through the chat template, so it is not perfectly matching the official inference prompt.

## Official Inference Engine

Official inference is in:

```text
third_party/StructEval/structeval/inference.py
```

It uses `LLM-Engines`:

```python
llm.load_model(
    model_name=model_name,
    engine=llm_engine,
    num_workers=1,
    num_gpu_per_worker=1,
    use_cache=False,
    **kwargs
)

responses = llm.batch_call_model(
    model_name,
    queries,
    num_proc=32,
    timeout=None,
    disable_batch_api=True,
    temperature=1.0,
    max_tokens=None,
)
```

For our KV-cache research, this official inference function is not directly usable because it hides model internals and sets `use_cache=False`. We need manual generation to intervene in `past_key_values`. However, we should copy the official prompt wrapper and evaluation logic.

## Official Non-Renderable Evaluation

For JSON/YAML/CSV/TOML/XML, StructEval treats the task as non-renderable.

The render/extraction stage is in:

```text
third_party/StructEval/structeval/render_engine/render_utils.py
```

For non-renderable tasks, it:

1. Extracts code from `<|BEGIN_CODE|> ... <|END_CODE|>`.
2. If markers are absent, tries markdown fences.
3. If no wrapper is found, falls back to the whole generation text.
4. Saves the extracted output to a file such as `000500.json`.
5. Attempts to parse that file.
6. Sets `render_score = 1` if the file parses as the requested format.

The path validation stage is in:

```text
third_party/StructEval/structeval/eval_engine/eval_nonrenderable.py
```

For each required path in `raw_output_metric`, it checks whether that path exists in the parsed structure.

The path utility is:

```text
third_party/StructEval/structeval/eval_engine/eval_utils.py
```

Important details:

- Dot paths are tokenized by `.`.
- Array indices such as `[0]` are supported.
- Wildcard `*` is supported over lists.
- CSV paths can use the special `csv::header` form.
- XML attribute fallback is supported: `@id` can match `id`.

## Official Non-Renderable Score

For non-renderable outputs, final score is:

```text
final_eval_score = 0.2 * render_score + 0.8 * key_validation_score
```

Where:

- `render_score` means parse/format validity.
- `key_validation_score` means required-path success rate.

Our current notebook reports these separately as:

- `json_parse_success`
- `json_required_path_rate`

This is actually more diagnostic for KV-cache research, but if we want official StructEval comparability, we should also report:

```text
official_like_score = 0.2 * json_parse_success + 0.8 * json_required_path_rate
```

## Dataset Match

The upstream `dataset/nonrenderable.json` contains the same JSON tasks we are using from:

```text
toy_kv_experiments/data/structeval_full/structeval_test.jsonl
```

Checked rows:

```text
000500 query_same=True metrics_same=True
000501 query_same=True metrics_same=True
000502 query_same=True metrics_same=True
000503 query_same=True metrics_same=True
```

## Action Items for Our Notebook

To better align with official StructEval while preserving KV-cache control:

1. Keep manual autoregressive generation so we can fake-quantize `past_key_values`.
2. Add the official `<|BEGIN_CODE|>` / `<|END_CODE|>` prompt wrapper.
3. Add official-style extraction before JSON parsing.
4. Continue reporting parse success and path rate separately.
5. Add `official_like_score = 0.2 * parse_success + 0.8 * path_rate`.
