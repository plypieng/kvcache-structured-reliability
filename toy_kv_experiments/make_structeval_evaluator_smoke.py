from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SMOKE_GENERATIONS = {
    "Angular": """import { Component } from '@angular/core';
@Component({
  selector: 'smoke-check',
  template: `<main><h1>Angular smoke</h1><p>Renderer is working.</p></main>`
})
export class SmokeComponent {}""",
    "Canvas": """const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#0f766e';
ctx.fillRect(40, 40, 240, 120);
ctx.fillStyle = 'white';
ctx.font = '24px sans-serif';
ctx.fillText('Canvas smoke', 70, 105);""",
    "CSV": "name,value\nalpha,1\nbeta,2",
    "HTML": """<!doctype html><html><body><main><h1>HTML smoke</h1></main></body></html>""",
    "JSON": '{"smoke": {"status": "ok", "value": 1}}',
    "Latex": r"""\documentclass{article}
\usepackage{xcolor}
\begin{document}
\section*{LaTeX smoke}
Renderer is working.
\end{document}""",
    "Markdown": "# Markdown smoke\n\nRenderer is **working**.",
    "Matplotlib": """plt.figure(figsize=(4, 3))
plt.plot([0, 1, 2], [0, 1, 0], marker='o')
plt.title('Matplotlib smoke')
plt.grid(True)""",
    "Mermaid": "graph TD\n  A[Input] --> B[Renderer]\n  B --> C[Image]",
    "React": """function App() {
  return <main><h1>React smoke</h1><p>Renderer is working.</p></main>;
}
export default App;""",
    "SVG": """<svg xmlns="http://www.w3.org/2000/svg" width="480" height="220">
  <rect width="480" height="220" fill="#f0fdfa"/>
  <circle cx="110" cy="110" r="60" fill="#0f766e"/>
  <text x="200" y="120" font-size="28">SVG smoke</text>
</svg>""",
    "TOML": "[smoke]\nstatus = \"ok\"\nvalue = 1",
    "Tikz": r"""\begin{tikzpicture}
\node[draw, rounded corners, fill=blue!10] (a) {TikZ smoke};
\node[draw, right=of a] (b) {Image};
\draw[->] (a) -- (b);
\end{tikzpicture}""",
    "Typst": "#set page(width: 12cm, height: 7cm)\n= Typst smoke\nRenderer is working.",
    "Vega": """{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {"values": [{"x": "A", "y": 3}, {"x": "B", "y": 5}]},
  "mark": "bar",
  "encoding": {"x": {"field": "x", "type": "nominal"}, "y": {"field": "y", "type": "quantitative"}}
}""",
    "Vue": """<template><main><h1>Vue smoke</h1><p>Renderer is working.</p></main></template>
<script>export default { name: 'SmokeApp' };</script>""",
    "XML": "<smoke><status>ok</status><value>1</value></smoke>",
    "YAML": "smoke:\n  status: ok\n  value: 1",
}


def wrap_generation(code: str) -> str:
    return f"<|BEGIN_CODE|>\n{code}\n<|END_CODE|>"


def build_smoke_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smoke_rows = []
    for output_type, generation in SMOKE_GENERATIONS.items():
        source = next((row for row in dataset_rows if row.get("output_type") == output_type), None)
        if source is None:
            raise ValueError(f"official dataset has no {output_type} task")
        row = dict(source)
        row["generation"] = wrap_generation(generation)
        smoke_rows.append(row)
    return smoke_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a safe one-task-per-format StructEval fixture.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    smoke_rows = build_smoke_rows(rows)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(smoke_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(smoke_rows)} smoke tasks to {output}")


if __name__ == "__main__":
    main()
