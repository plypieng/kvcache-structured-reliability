# Evidence Freeze: July 23, 2026

This directory is the source of truth for the IEICE one-page manuscript, the
July 29 full-lab seminar, and the August 3 poster. Results produced after this
freeze must not be inserted into those artifacts without a new dated freeze.

## Experimental Identity

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Device: NVIDIA RTX A6000
- Generation: greedy decoding, temperature 0, seed 123
- Maximum generation: 2,048 tokens
- Stop marker: `<|END_CODE|>`
- Quantizer: packed affine KIVI-style reference cache
- Cache revision: `kivi-post-attention-finalize-v3`
- Key grouping: across token groups
- Value grouping: across channel groups per token
- Group size: 32
- Full-precision residual length: 128
- Prefill policy: full-precision attention followed by packed storage
- PyTorch: `2.11.0+cu128`
- Transformers: `5.12.1`
- Model configuration SHA-256:
  `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`
- Combined experiment-source SHA-256:
  `459747ca39641b0f536bd35cca0fb4c599f020428c5ac5d291568da868366600`

The workspace is not a Git repository. Artifact hashes in `SHA256SUMS` and the
embedded source/model fingerprints are therefore the provenance record.

## Evidence Set A: Balanced All-Format Pilot

The frozen manifest contains 100 tasks: 25 each from textual generation,
textual conversion, visual generation, and visual conversion. Only the 50
textual tasks have completed official StructEval evaluation. The 50 visual
tasks must not be included in a reported benchmark average because their VQA
evaluation is incomplete.

| Setting | Unchanged | Better | Worse | Mean size / FP16 | Changed task types |
|---|---:|---:|---:|---:|---|
| K8/V8 | 49 | 1 | 0 | 0.601 | Better: XML to JSON |
| K8/V4 | 47 | 2 | 1 | 0.495 | Better: XML to JSON, Text to TOML; worse: Text to YAML |

The apparent mean-score increases are caused by one or two changed tasks and
must not be presented as evidence that quantization improves the model.

## Evidence Set B: Focused Textual-20 Development Set

The development manifest was selected deterministically from Evidence Set A,
without using model outcomes. It contains ten textual-generation and ten
textual-conversion tasks across CSV, JSON, TOML, XML, and YAML.

| Setting | Unchanged | Better | Worse | New parse failures | Mean size / FP16 |
|---|---:|---:|---:|---:|---:|
| K8/V8 | 19 | 1 | 0 | 0 | 0.621 |
| K8/V4 | 19 | 1 | 0 | 0 | 0.518 |
| K4/V4 | 17 | 2 | 1 | 0 | 0.403 |
| K2/V2 | 13 | 3 | 4 | 1 | 0.288 |

K2/V2 is the first completed setting with several paired degradations and is
therefore the current development operating point. Its four degraded tasks are:

| Task | Conversion | FP16 | K2/V2 | Observation |
|---|---|---:|---:|---|
| `001830` | Text to YAML | 1.00 | 0.86 | Required-path coverage decreased |
| `001818` | Text to YAML | 1.00 | 0.00 | New parser failure |
| `000219` | Text to CSV | 1.00 | 0.20 | Structural score decreased |
| `051714` | JSON to XML | 0.85 | 0.20 | Required-path coverage decreased |

The three improved tasks are retained in the evidence because quantization can
change decoding in either direction. A valid comparison must report paired
improvements and regressions rather than only an aggregate average.

## Claims Allowed Before August 3

- A corrected packed KIVI-style cache was evaluated under paired decoding.
- Persistent cache accounting includes packed payload, scale/minimum metadata,
  and the full-precision residual.
- Most evaluated task scores were unchanged at moderate precision.
- K2/V2 produced four paired degradations and one new parser failure on the
  frozen textual-20 development set.
- Structured-output failures can be localized and used to design the next
  matched-budget fidelity-allocation experiment.

## Claims Not Allowed

- Exact reproduction of the KIVI paper.
- KIVI-equivalent latency, throughput, or peak-memory performance.
- Official full StructEval leaderboard comparability.
- Quantization improves structured generation.
- Structure-aware protection is effective; it has not yet been evaluated.
- Formal reliability guarantees or established novelty.

## Next Validation

The isolated official KIVI environment and CUDA extension passed smoke
validation on July 23; no benchmark has been launched. The repository must
next be evaluated independently on a supported Mistral or Llama model. Its
FP16, KIVI-4, and KIVI-2 results will then be compared with the same protocol
implemented by this project. That reproduction is not a dependency for the
July 29 manuscript or the August 3 poster.
