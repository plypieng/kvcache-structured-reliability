# IEICE Shin-Etsu One-Page Manuscript

Working title:

**An Experimental Evaluation of Structured-Output Reliability under KV-Cache Quantization**

The draft uses the official `ieicejsp` class and two completed evidence sets:

1. The completed official KIVI LongBench-15 validation for FP16, KIVI-4, and
   KIVI-2.
2. The complete 950-task StructEval-T comparison using the same
   Mistral model and official KIVI implementation.

All three StructEval-T conditions completed and passed the evaluator. The
paired comparison was regenerated with `python3` after the queue's original
post-processing command used an unavailable `python` executable. The
inference and evaluation stages were unaffected.

The manuscript uses four references: PagedAttention for the KV-cache memory
motivation, KIVI for the compression method, LongBench for implementation
validation, and StructEval for structured-output evaluation.

Before submission:

- confirm the author list and affiliation with the advisor;
- insert the presentation number if required by the submission system;
- confirm whether the conference wants an English-only or bilingual title;
- update the acknowledgement only if funding or computing-resource wording is
  required;
- keep LongBench numbers traceable to the completed official artifacts;
- keep the StructEval-T aggregate and paired counts traceable to
  `../../docs/evidence/20260731/official_kivi_structeval_t/comparison.json`;
- do not describe the proposed structure-aware allocation as an evaluated
  method;
- verify the final PDF is exactly one A4 page.

Build:

```bash
cd manuscripts/ieice_shinetsu_2026
platex manuscript.tex
platex manuscript.tex
dvipdfmx manuscript.dvi
```
