# Official KIVI on StructEval-T

| Condition | T-generation | T-conversion | T mean | Parse success |
|---|---:|---:|---:|---:|
| FP16 | 0.6553 | 0.3485 | 0.5019 | 527/950 |
| KIVI-4 | 0.6515 | 0.3846 | 0.5181 | 564/950 |
| KIVI-2 | 0.5079 | 0.3405 | 0.4242 | 520/950 |

| Pair | Regressed | Improved | New parse failures | Path regressions |
|---|---:|---:|---:|---:|
| KIVI-4 vs FP16 | 47 | 78 | 23 | 41 |
| KIVI-2 vs FP16 | 215 | 162 | 125 | 195 |
