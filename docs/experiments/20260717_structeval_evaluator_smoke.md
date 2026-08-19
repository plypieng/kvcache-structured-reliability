# StructEval Evaluator Smoke

Date: July 17, 2026  
Status: Passed  
Evaluator: official `litellm` branch, commit `7781339`

## Purpose

Verify the real rendering and evaluation dependencies before waiting for a
2,035-task model run. Package imports alone are insufficient because StructEval
uses browsers, framework compilers, generated Python, TeX, Typst, and multiple
structured-data parsers.

## Environment

- Conda environment: `structeval-eval`
- Python: 3.12
- Node: 18.20.5
- Browser: Playwright Chromium 1228
- LaTeX/TikZ: Tectonic
- Typst: Typst CLI plus ImageMagick
- Other tools: Poppler, Graphviz, Ghostscript, Matplotlib
- Isolation: Bubblewrap, read-only root filesystem, writable per-run scratch,
  sanitized child environment, isolated PID/IPC/UTS namespaces

The upstream evaluator test suite passed: `22 passed`.

## Runtime smoke

A deterministic fixture selected one official dataset row for every output
format and inserted a small safe generation using the official BEGIN/END tags.

| Family | Formats | Result |
|---|---|---:|
| Browser/framework | Angular, Canvas, HTML, Markdown, Mermaid, React, SVG, Vega, Vue | 9/9 |
| Local renderers | Matplotlib, LaTeX, TikZ, Typst | 4/4 |
| Non-renderable parsers | CSV, JSON, TOML, XML, YAML | 5/5 |
| **Total** | 18 formats | **18/18** |

The first unsandboxed fixture run found a LaTeX-fixture defect: the minimal full
document did not load `xcolor`, while the evaluator injects color definitions.
Adding `xcolor` fixed the fixture. TikZ had already passed, which isolated the
problem from Tectonic itself.

## Artifacts

- Isolated passing run:
  `toy_kv_experiments/results/structeval_evaluator_smoke/20260717_isolated/`
- Fixture generator:
  `toy_kv_experiments/make_structeval_evaluator_smoke.py`
- Smoke runner:
  `toy_kv_experiments/server/run_structeval_evaluator_smoke.sh`
- Isolated renderer:
  `toy_kv_experiments/server/render_structeval_isolated.sh`

## Remaining prerequisite

The server has no `OPENAI_API_KEY`. Full StructEval-V scoring requires
GPT-4.1-mini to judge the 1,085 rendered examples. All local evaluation stages
are otherwise ready.
