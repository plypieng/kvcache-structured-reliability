# Frozen StructEval-T evaluator outputs

Each condition contains 950 rows produced by the official StructEval
non-renderable evaluator, plus the corresponding run metadata.

- `fp16/`: unquantized Key and Value cache;
- `kivi4/`: official KIVI with 4-bit keys and values;
- `kivi2/`: official KIVI with 2-bit keys and values;
- `analysis/`: deterministic reports generated from the three paired files.

The files are model generations and evaluator results, not benchmark source
examples or model weights. Verify them with:

```bash
shasum -a 256 -c artifacts/structeval_t_20260731/SHA256SUMS
```

