# StructEval-T Experiment Freeze

## Purpose

This record freezes the experiment underlying the IEICE Shin-Etsu manuscript.
It distinguishes the completed evidence from later mechanism analysis and
structure-aware protection experiments.

## Frozen configuration

| Item | Value |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Model revision | `41b61a33a2483885c981aa79e0df6b32407ed873` |
| KIVI revision | `67aba607a1deaeb18b70ae796ab25d05a08b3345` |
| StructEval scope | Complete textual track: 250 generation + 700 conversion tasks |
| Conditions | FP16, KIVI-4 (K4/V4), KIVI-2 (K2/V2) |
| KIVI group size | 32 |
| FP16 residual length | 128 tokens |
| Decoding | Greedy, seed 42 |
| Generation ceiling | 2,048 new tokens |
| Stop conditions | `<|END_CODE|>`, EOS, or generation ceiling |
| Hardware | NVIDIA RTX A6000 |
| PyTorch | 2.1.2+cu121 |
| Transformers | 4.36.2 |

## Dataset identity

- StructEval source SHA-256:
  `d4d838145fbe285d0c446b449470512c0c4d3922422894c77aa6183913751bda`
- Complete textual manifest SHA-256:
  `fdcc2880b57f84b03e08a2c1846e2873ffdd6d6d052c27249bb93e4dd1e8a58a`
- Manifest:
  `toy_kv_experiments/data/structeval_full/manifests/structeval_t_complete_950.json`

The downloaded source dataset is intentionally not committed. The manifest
fixes task identity and order, while the source hash detects dataset drift.

## Protocol boundary

The run uses all StructEval-T tasks and the official non-renderable evaluator.
It is not claimed as an official leaderboard submission because inference used
a controlled greedy protocol with a 2,048-token ceiling rather than the
benchmark's inference API defaults.

## Frozen outputs

The evaluated rows and run metadata are stored under
`artifacts/structeval_t_20260731/`. `SHA256SUMS` records their content hashes.
The local hashes were checked against the original A6000 server files after
transfer.

## Headline results

| Cache | Generation score | Conversion score | T score | Parse success |
|---|---:|---:|---:|---:|
| FP16 | 0.6553 | 0.3485 | 0.5019 | 527 / 950 |
| KIVI-4 | 0.6515 | 0.3846 | 0.5181 | 564 / 950 |
| KIVI-2 | 0.5079 | 0.3405 | 0.4242 | 520 / 950 |

These aggregate values do not establish that KIVI-4 improves structured
generation. Paired transitions and uncertainty estimates must be considered.

## Original server location

`/home/plypieng/official_baselines/results/kivi_structeval_t_20260729_170807`

The repository does not store server credentials or machine-specific secrets.

