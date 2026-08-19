# A6000 StructEval Workflow

The current reference experiment uses `Qwen/Qwen2.5-7B-Instruct` and a frozen,
balanced 100-task StructEval JSON manifest. It is intended to separate baseline
model difficulty from KV-cache fidelity effects.

The active cache revision is `kivi-post-attention-finalize-v3`. Decode attention
uses the newly appended full-precision residual and only then moves eligible
entries into packed storage. Older result revisions must not be mixed with v3.

## Setup

```bash
cd ~/kvcache
bash toy_kv_experiments/server/setup_a6000_env.sh
```

The expected environment is `kvcache-py311` under `~/miniforge3` on the RTX
A6000 workstation.

## Controlled KIVI-Style Reference

The reference cache uses affine min/max storage: keys are grouped across tokens,
values are grouped across channels per token, and prefill attention remains full
precision. It stores packed 2/4/8-bit cache payload plus scale/minimum metadata,
but dequantizes before standard Hugging Face attention. Do not interpret it as a
fused-kernel latency or peak-memory result.

Before JSON-100, run the paired ten-task validity gate:

```bash
cd ~/kvcache
nohup bash toy_kv_experiments/server/run_post_attention_validity_gate_json10.sh \
  > toy_kv_experiments/logs/post_attention_gate_v3.nohup.log 2>&1 &
```

Each task is atomically saved under `<result>.json.checkpoints/tasks/`. The
adjacent `status.json` reports the current task count, and a matching run can be
resumed without repeating completed tasks.

Run one setting at a time on the frozen 100-task manifest:

```bash
cd ~/kvcache
BITS=16 bash toy_kv_experiments/server/run_kivi_reference_json100.sh
BITS=8  bash toy_kv_experiments/server/run_kivi_reference_json100.sh
BITS=4  bash toy_kv_experiments/server/run_kivi_reference_json100.sh
BITS=2  bash toy_kv_experiments/server/run_kivi_reference_json100.sh
```

After the JSON-10 outputs and disagreements pass inspection, the prepared Stage
A queue is:

```bash
bash toy_kv_experiments/server/run_post_attention_stage_a_json100.sh
```

The default residual length is 128 and group size is 32. Only 2, 4, 8, and 16
are valid persistent-storage widths. Earlier 6/7-bit runs remain numerical
diagnostics only and must not be used as compression-ratio evidence.

Useful asymmetric baselines:

```bash
KEY_BITS=8 VALUE_BITS=4 BITS=8 bash toy_kv_experiments/server/run_kivi_reference_json100.sh
KEY_BITS=4 VALUE_BITS=2 BITS=4 bash toy_kv_experiments/server/run_kivi_reference_json100.sh
```

## Interpretation

- Compare only settings run on the same frozen manifest.
- Report absolute score and the failure rate among tasks where FP16 succeeds.
- Treat stored payload, metadata, and residual bytes separately.
- A valid parse does not imply semantic correctness; use required-path rate and
  final StructEval score as well. Paired reports also include FP16-reference
  leaf path-value agreement, which is a diagnostic rather than ground truth.
- Call this protocol the deterministic StructEval-T JSON subset. It is not the
  official full StructEval benchmark inference protocol.
- Match allocation methods by observed persistent-cache bytes or effective
  stored bits, not only their nominal K/V labels.

## Fidelity Allocation

After the uniform reference is stable, use the protection runners. Their default
signal source is `prompt-visible`, which extracts identifiers from source code or
schema text visible to the model. This is deployable. The historical evaluator
path source is available only as explicitly labeled oracle analysis:

```bash
PROTECTION_SIGNAL_SOURCE=oracle-required-paths \
  ALLOW_ORACLE_PROTECTION=1 \
  LIMIT=20 bash toy_kv_experiments/server/run_constraint_uep_structeval_jobs.sh
```

Do not use oracle and prompt-visible results as if they were the same method.
