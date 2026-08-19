from __future__ import annotations

DATASET_SOURCES = {
    "tinystories": {
        "hf_id": "roneneldan/TinyStories",
        "description": "Synthetic short stories for training and evaluating small language models.",
        "paper": {
            "title": "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?",
            "authors": "Ronen Eldan and Yuanzhi Li",
            "arxiv": "2305.07759",
            "url": "https://arxiv.org/abs/2305.07759",
            "pdf": "https://arxiv.org/pdf/2305.07759",
        },
        "dataset_url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
    "structeval": {
        "hf_id": "TIGER-Lab/StructEval",
        "description": "Structured-output benchmark covering JSON, XML, YAML, CSV, TOML and renderable formats.",
        "paper": {
            "title": "StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs",
            "authors": "Jialin Yang et al.",
            "arxiv": "2505.20139",
            "url": "https://arxiv.org/abs/2505.20139",
            "pdf": "https://arxiv.org/pdf/2505.20139",
        },
        "dataset_url": "https://huggingface.co/datasets/TIGER-Lab/StructEval",
    },
    "synthetic_structured": {
        "hf_id": "mdonigian/synthetic-structured-output-dataset",
        "description": "Synthetic structured-output/function-calling style dataset.",
        "paper": None,
        "dataset_url": "https://huggingface.co/datasets/mdonigian/synthetic-structured-output-dataset",
    },
    "sob": {
        "hf_id": "interfaze-ai/sob",
        "description": "Structured Output Benchmark for schema compliance and value accuracy.",
        "paper": {
            "title": "The Structured Output Benchmark: A Multi-Source Benchmark for Evaluating Structured Output Quality in Large Language Models",
            "authors": "Abhinav Kumar Singh, Harsha Vardhan Khurdula, Yoeven D. Khemlani, and Vineet Agarwal",
            "arxiv": "2604.25359",
            "url": "https://arxiv.org/abs/2604.25359",
            "pdf": "https://arxiv.org/pdf/2604.25359",
        },
        "dataset_url": "https://huggingface.co/datasets/interfaze-ai/sob",
    },
}
