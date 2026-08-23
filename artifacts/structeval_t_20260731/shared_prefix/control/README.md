# Matched non-regression control

This control contains 23 `both_pass` tasks from the frozen KIVI-4 evaluation.
Each task matches one of the 23 KIVI-4-induced parse regressions by input/output
format and nearest FP16 generated length. Selection is without replacement and
is deterministic; the manifest records each pairing and length difference.

The A6000 run completed on 2026-08-23 from 21:25:53 to 21:47:23 JST. The raw
traces are checked by `SHA256SUMS`. The final comparison is
`analysis/COMPARISON.md`.

This is a descriptive matched control, not a randomized causal experiment. It
does not justify a claim that structural tokens cause the observed failures.
