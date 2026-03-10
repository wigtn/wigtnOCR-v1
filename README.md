<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](pyproject.toml)

<a href="README.md"><img src="https://img.shields.io/badge/English-blue?style=for-the-badge" alt="English"></a>
<a href="README.ko.md"><img src="https://img.shields.io/badge/한국어-gray?style=for-the-badge" alt="Korean"></a>

</div>

# WigtnOCR

**VLM-based Korean Government Document Intelligence Parser**

> Structure-preserving document parsing via Vision-Language Models with LoRA fine-tuning for Korean public sector documents.

WigtnOCR is a 3-layer research framework:

1. **Framework** (`wigtnocr/`) — Installable Python package (`pip install wigtnocr`) with hybrid routing parser
2. **Fine-tuned Model** (`training/`) — Qwen3-VL-2B LoRA adaptation for Korean government documents
3. **Benchmark** (`evaluation/`) — KoGovDoc-Bench + OmniDocBench evaluation suite

---

## Quick Start

```bash
# Install
pip install -e .

# Parse a document (3 lines)
from wigtnocr import WigtnOCR

ocr = WigtnOCR(mode="hybrid")
result = ocr.parse("document.pdf")
print(result.content)
```

---

## Architecture

### Hybrid Routing Pipeline

WigtnOCR automatically routes documents based on complexity analysis:

```
PDF Input
    |
    v
[Complexity Analysis]
  scanned? / tables? / images? / text density?
    |
    +-- Simple (score < 0.5) --> CPU-only: PyMuPDF or RapidOCR
    |
    +-- Complex (score >= 0.5) --> GPU: Two-Stage (Extraction + VLM Structuring)
```

### 4-Parser Design

| Parser | Stage 1 (Extraction) | Stage 2 (Structuring) | Use Case |
|--------|---------------------|----------------------|----------|
| **Text-Baseline** | PyMuPDF | - | Digital PDF, speed-critical |
| **Image-Baseline** | RapidOCR | - | Scanned PDF, speed-critical |
| **Text-Advanced** | PyMuPDF | Qwen3-VL-2B | Digital PDF, structure-critical |
| **Image-Advanced** | RapidOCR | Qwen3-VL-2B | Scanned PDF, structure-critical |

### VLM Serving

Three model tiers served via vLLM on dual RTX PRO 6000 (96GB each):

| Model | Port | Purpose |
|-------|------|---------|
| Qwen3-VL-30B-A3B-Thinking | 8000 | Pseudo GT generation |
| Qwen3-VL-8B-Thinking-FP8 | 8004 | Validation / ablation |
| Qwen3-VL-2B-Instruct | 8010 | Production inference + LoRA base |

---

## 3-Layer Contribution Stack

### Layer 1: Framework (`wigtnocr/`)

Installable Python package with:
- **Hybrid routing** — Complexity-based automatic parser selection
- **Two-stage pipeline** — Extraction + VLM structuring
- **Semantic chunking** — LangChain SemanticChunker with BGE-M3 embeddings
- **MoC-based metrics** — BC (Boundary Clarity) and CS (Chunk Stickiness)

```python
from wigtnocr import WigtnOCR, TwoStageParser, HybridParser

# High-level API
ocr = WigtnOCR(mode="hybrid")
result = ocr.parse("document.pdf")

# Low-level API
parser = TwoStageParser(
    structurer_api_url="http://localhost:8010/v1/chat/completions",
    structurer_model="qwen3-vl-2b-instruct",
)
result = parser.parse_auto(pdf_bytes)
```

### Layer 2: Fine-tuned Model (`training/`)

LoRA fine-tuning pipeline for Qwen3-VL-2B:

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-VL-2B-Instruct |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Dropout | 0.05 |

**Pseudo GT Generation**: Claude/GPT-4o generates structured markdown from documents, auto-approved when CER < 5%.

### Layer 3: Benchmark (`evaluation/`)

| Benchmark | Language | Documents | Purpose |
|-----------|----------|-----------|---------|
| **KoGovDoc-Bench** | Korean | 11 government docs | Domain-specific evaluation |
| **OmniDocBench** | Multi | Varies | Cross-domain generalization |
| **DocVQA** | English | Varies | QA-based evaluation (planned) |

**Metrics**: CER, WER, Structure F1, TEDS, BC, CS

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for VLM inference)

### Install

```bash
# Core only
pip install -e .

# With OCR support (RapidOCR)
pip install -e ".[ocr]"

# With training dependencies
pip install -e ".[training]"

# With evaluation dependencies
pip install -e ".[evaluation]"

# Everything
pip install -e ".[all]"
```

### VLM Server

```bash
# Using vLLM (see docker-compose.vlm.yml for full config)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --served-model-name qwen3-vl-2b-instruct \
    --port 8010
```

---

## Usage

### Document Parsing

```bash
# Evaluate parsers on all datasets
python -m evaluation.cli parse --all

# Single document
python -m evaluation.cli parse --input datasets/documents/kogov_001/doc.pdf
```

### Chunking Evaluation

```bash
# Evaluate chunking quality
python -m evaluation.cli chunk --all

# Mock mode (no embedding API)
python -m evaluation.cli chunk --all --use-mock
```

### Training

```bash
# Generate pseudo ground truth
python scripts/generate_pseudo_gt.py

# Run LoRA fine-tuning
python -m training.lora_trainer --config configs/training.yaml
```

---

## Evaluation Metrics

| Stage | Metric | Formula | Role |
|-------|--------|---------|------|
| **Prerequisite** | CER | `(S+D+I)/N` | Text extraction quality |
| **Prerequisite** | WER | `(S+D+I)/N` | Word-level error rate |
| **Core** | Structure F1 | `2*P*R/(P+R)` | Structure preservation |
| **Core** | TEDS | Tree Edit Distance Similarity | Table structure accuracy |
| **Downstream** | BC | `1 - cosine_sim(chunk_i, chunk_i+1)` | Boundary independence |
| **Downstream** | CS | Structural Entropy | Chunk dependency |

---

## Project Structure

```
wigtnocr/                          # PyPI package (pip install wigtnocr)
    __init__.py                    # WigtnOCR main class (3-line API)
    parsers/
        base.py                    # Abstract BaseParser, ParseResult
        pymupdf.py                 # Text extraction (PyMuPDF)
        rapidocr.py                # Image OCR (RapidOCR)
        vlm.py                     # VLM structuring (TextStructurer)
    pipeline/
        two_stage.py               # Two-stage parser (Extraction + VLM)
        hybrid.py                  # Hybrid router (complexity-based)
    chunking/
        chunker.py                 # SemanticChunker (LangChain)
        embeddings.py              # API embedding client (BGE-M3)
        metrics.py                 # BC/CS metrics
    utils/
        markdown.py                # Markdown utilities

training/                          # LoRA fine-tuning pipeline
    config.py                      # LoRA/Training dataclasses
    gt_generator.py                # Pseudo GT generation (Claude/GPT-4o)
    data_prep.py                   # Training data preparation
    lora_trainer.py                # LoRA fine-tuning entry point
    prompts/
        templates.py               # System prompts for GT/training

evaluation/                        # Benchmark & evaluation suite
    metrics/
        cer.py                     # CER, WER (jiwer)
        structure.py               # Structure F1
        teds.py                    # Table Edit Distance Similarity
        chunking.py                # BC/CS wrapper
    benchmarks/
        kogovdoc.py                # KoGovDoc-Bench loader
        omnidocbench.py            # OmniDocBench adapter
        docvqa.py                  # DocVQA (planned)
    runners/
        parser_eval.py             # Parser evaluation runner
        chunking_eval.py           # Chunking evaluation runner
    cli.py                         # Unified CLI (parse/chunk)

configs/                           # YAML configurations
    default.yaml                   # VLM server endpoints
    training.yaml                  # LoRA hyperparameters
    evaluation.yaml                # Evaluation settings

datasets/                          # Benchmark datasets
    papers/                        # English academic papers (39)
    documents/                     # Korean government docs (11)
    omnidocbench/                  # OmniDocBench

scripts/                           # Utility scripts
    generate_pseudo_gt.py          # Pseudo GT generation script
    setup_datasets.py              # Dataset setup helper
```

---

## References

- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — Base VLM
- [MoC Paper](https://arxiv.org/abs/2503.09600) — BC/CS chunking metrics
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — Document benchmark
- [vLLM](https://github.com/vllm-project/vllm) — VLM serving engine

---

## License

MIT License

## Citation

```bibtex
@software{wigtnocr,
  title = {WigtnOCR: VLM-based Korean Government Document Intelligence Parser},
  author = {Kim, Hyeongseob},
  year = {2026},
  url = {https://github.com/Hyeongseob91/wigtnocr}
}
```
