<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](pyproject.toml)

<a href="README.md"><img src="https://img.shields.io/badge/English-blue?style=for-the-badge" alt="English"></a>
<a href="README.ko.md"><img src="https://img.shields.io/badge/한국어-gray?style=for-the-badge" alt="Korean"></a>

</div>

# Document Parsing Structure Preservation Test Framework

> **"Structure lost at parsing is structure lost forever — no downstream optimization can recover it."**

In enterprise RAG pipelines handling complex documents — nested tables, image-embedded tables, multi-column layouts — we discovered that **the parsing stage is the irreversible bottleneck**. Once document structure is lost during parsing, no amount of downstream optimization (chunking strategies, embedding model swaps, reranker additions) can recover it. This framework quantitatively validates that hypothesis by comparing traditional OCR methods against VLM-based structured parsing across the full pipeline.

### Key Results

| RQ | Question | Metric | Result |
|----|----------|--------|--------|
| **RQ1** | Is OCR extraction quality sufficient for VLM input? | CER, WER | Image-Advanced CER 33% (Best), Baseline CER 40-51% |
| **RQ2** | Does VLM Two-Stage Parsing preserve structure better? | Structure F1 | **0% → 77~79%** (Both Advanced, Recall 88%) |
| **RQ3** | Does structure preservation improve chunking? | BC, CS | BC 0.512, 18 natural section boundaries |

**Balanced Choice**: Image-Advanced — Best CER (33%), competitive F1 (78%), 17% faster than Text-Advanced

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run parser comparison benchmark
python -m src.eval_parsers --all

# Run chunking evaluation
python -m src.eval_chunking --all
```

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

### Problem Statement

Traditional RAG pipelines rely heavily on plain text extraction, which fails to preserve critical document structures:

- **Table Structure Loss**: Row-column relationships destroyed
- **Multi-column Errors**: Reading order confusion in two-column layouts
- **Header Hierarchy Loss**: Section relationships not preserved
- **Semantic Discontinuity**: Chunking breaks at wrong positions

**Root Cause**: All of these symptoms originate from a single point of failure — **structural information loss at the Data Parsing stage**, the very first step of the pipeline. All Baseline parsers produce **Structure F1 = 0%** — they extract text but generate no markdown structure elements.

### Hypothesis

> If the first stage of the pipeline (Data Parsing) preserves document structure, then the same downstream processing (chunking, embedding, retrieval) achieves significantly higher quality. Conversely, if structure is lost at parsing, no downstream optimization can overcome that ceiling.

### Structured Data Definition

> **Structured data** in this framework refers to text containing markdown structure elements: Headings (`#`), Lists (`-`/`1.`), Tables (`|...|`), and Code Blocks (`` ``` ``). Structure quality is measured by counting these elements in Ground Truth (GT) markdown files and computing **Structure F1** (Precision, Recall, F1).

### Research Questions

| RQ | Question | Metrics | Role |
|----|----------|---------|------|
| **RQ1** | Is OCR extraction quality sufficient for VLM input? | CER, WER | **Prerequisite validation** |
| **RQ2** | Does VLM Two-Stage Parsing preserve document structure better? | Structure F1 (Precision, Recall) | **Core hypothesis** |
| **RQ3** | Does structure preservation improve semantic chunking quality? | BC (Boundary Coherence), CS (Chunk Score) | **Downstream impact** |
| **RQ4** | Does improved chunking improve retrieval precision? | Hit Rate@k, MRR | **End-to-end validation** |

**Logic flow**: CER/WER prerequisite check → VLM structuring → Structure F1 measurement → BC/CS downstream → Retrieval validation

<div align="center">

<img src="docs/tech_report/figures/fig1_structure_f1_comparison.png" width="600" alt="Structure F1 Comparison: Baseline (0%) vs Advanced (~79%)">

**Fig 1.** Baseline parsers preserve 0% of document structure, while VLM-based Advanced parsers achieve ~79% Structure F1.

<br>

<img src="docs/tech_report/figures/fig3_tradeoff_scatter.png" width="600" alt="Trade-off: CER vs Structure F1">

**Fig 2.** Image-Advanced achieves the best balance: lowest CER (33%) with competitive Structure F1 (78%).

</div>

> For full experimental results and analysis, see the [Tech Report](docs/tech_report/).

---

## Key Features

### 4-Parser Architecture

A controlled experiment design with 4 parsers: **Baseline** (no structure preservation) vs **Advanced** (VLM-based structure preservation), across both **Text** (digital PDF) and **Image** (scanned PDF) extraction paths. This isolates the effect of structure preservation regardless of the extraction method.

| Parser | Description | Stage 1 | Stage 2 |
|--------|-------------|---------|---------|
| **Text-Baseline** | Digital PDF text extraction | PyMuPDF (`fitz`) | - |
| **Image-Baseline** | Scanned PDF OCR | RapidOCR | - |
| **Text-Advanced** | Digital + VLM structuring | PyMuPDF (`fitz`) | Qwen3-VL-2B-Instruct |
| **Image-Advanced** | Scanned + VLM structuring | RapidOCR | Qwen3-VL-2B-Instruct |

### VLM Prompt Strategy (v2 — CRITICAL RULES)

The prompt evolved through iterative experimentation:

1. **v1 (Initial)**: Generic "document structure expert" → Structure F1 = **0%** (no heading markers generated)
2. **v2 (commit `90d516d`)**: CRITICAL RULES + explicit heading level mapping → Structure F1 = **~79%**
   - System/User prompt separation
   - "MUST", "NEVER" enforcement directives
   - Number → markdown level mapping (1→`##`, 2.1→`###`, 3.1.1→`####`)

**Lesson**: For 2B-parameter models, explicit rules ("MUST use #") are far more effective than implicit instructions ("clean markdown").

### Semantic Chunking

Uses LangChain's SemanticChunker with embedding-based boundary detection:

- **Breakpoint Types**: percentile, standard_deviation, interquartile, gradient
- **Embedding Backend**: BGE-M3 via Infinity API or local sentence-transformers
- **Automatic Sizing**: Chunk size determined by semantic boundaries

### Model Selection: Qwen3-VL-2B

**Why VL (Vision-Language) over Text-only?**
- **Alternatives considered**: Qwen3-1.7B-Instruct (text-only) vs Qwen3-VL-2B (vision-language)
- **Decision rationale**: While a text-only model would suffice for this document parsing task, a separate internal pipeline required multi-modal input. Given GPU constraints (single 96GB card serving both pipelines), we chose the VL model for **versatility** across both use cases.
- **Result**: 2B VL model achieves Structure F1 77~79%, meeting requirements for both pipelines.
- **Future plan**: Dedicated document parsing line will transition to Qwen3-1.7B-Instruct with Curriculum Learning to improve structuring performance.

### MoC-based Quality Metrics (Label-Free)

Measures the downstream impact of parsing structure preservation on chunking quality. Based on the MoC paper (arXiv:2503.09600v2), using **Semantic Distance** instead of Perplexity:

| Metric | Implementation | Interpretation |
|--------|---------------|----------------|
| **BC (Boundary Clarity)** | `1 - cosine_similarity(chunk_i, chunk_i+1)` | Higher = more independent (good) |
| **CS (Chunk Stickiness)** | Structural Entropy of chunk graph | Lower = less dependency (good) |

**Note**: Semantic Distance BC scores (0.3-0.5) differ from MoC's Perplexity-based scores (0.8+) due to different metric scales.

### Four-Stage Evaluation

| Stage | Role | Metrics | Description |
|-------|------|---------|-------------|
| **Stage 1** | Prerequisite | CER, WER | Text extraction quality check |
| **Stage 2** | Core evaluation | Structure F1 (P, R) | Structure preservation measurement |
| **Stage 3** | Downstream | BC, CS | Chunking quality impact |
| **Stage 4** | Retrieval (WIP) | Hit Rate@k, MRR | End-to-end retrieval validation |

---

## Architecture

```
+--------------------------------------------------------------------+
|                        Document Input (PDF)                        |
+--------------------------------------------------------------------+
                                |
            +-------------------+-------------------+
            |                                       |
            v                                       v
+------------------------+             +------------------------+
|     Digital PDF?       |             |     Scanned PDF?       |
|   (has text layer)     |             |    (image-based)       |
+------------------------+             +------------------------+
            |                                       |
╔════════════════════════════════════════════════════════════════════╗
║              ★ PARSING STAGE (Bottleneck)  ★                     ║
║  Structure preserved here → quality propagates downstream        ║
║  Structure lost here     → no recovery possible                  ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   +------+------+                         +------+------+        ║
║   |             |                         |             |        ║
║   v             v                         v             v        ║
║ +---------+  +-----------+          +-----------+  +----------+  ║
║ |  Text   |  |   Text    |          |   Image   |  |  Image   |  ║
║ | Baseline|  |  Advanced |          |  Baseline |  | Advanced |  ║
║ | PyMuPDF |  | +VLM Struct|         |  RapidOCR |  |+VLM Struct| ║
║ +---------+  +-----------+          +-----------+  +----------+  ║
║   No struct.   Struct.                No struct.    Struct.       ║
║   preserved    preserved              preserved     preserved    ║
╚════════════════════════════════════════════════════════════════════╝
            |                                       |
            v                                       v
+--------------------------------------------------------------------+
|                    Semantic Chunking Layer                         |
|              (LangChain SemanticChunker + BGE-M3)                  |
+--------------------------------------------------------------------+
                                |
                                v
+--------------------------------------------------------------------+
|              Three-Stage Evaluation (Impact Cascade)               |
|   +-----------+   +------------+   +-----------+   +----------+   |
|   | RQ1:      |   | RQ2:       |   | RQ3:      |   | Future:  |   |
|   | CER, WER  | → | Structure  | → | BC, CS    | → | HR@k,MRR |   |
|   |(Prereq.)  |   | F1 (P,R)   |   |(Downstream)|   |(Retrieval)|   |
|   +-----------+   +------------+   +-----------+   +----------+   |
|                                                                    |
|   Parsing quality ──────────────────────→ Downstream quality       |
+--------------------------------------------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.13+
- CUDA-capable GPU (for VLM inference)
- MeCab (for Korean tokenization)

### Hardware Used in Experiments

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition × 2 (96GB VRAM each) |
| RAM | 128GB DDR5 |
| Storage | SSD |

### Using uv (Recommended)

```bash
git clone https://github.com/Hyeongseob91/test-vlm-document-parsing.git
cd test-vlm-document-parsing

# Install with uv
uv sync

# Install Korean NLP support
uv sync --extra korean
```

### Using pip

```bash
pip install -e .

# For Korean support
pip install -e ".[korean]"

# For all features
pip install -e ".[all]"
```

### External Services

```bash
# 1. Embedding API (Infinity with BGE-M3)
infinity_emb v2 --model-id BAAI/bge-m3 --port 8001

# 2. VLM API (for Advanced parsers)
# Deploy Qwen3-VL-2B-Instruct at http://localhost:8005
```

### MeCab Installation (Ubuntu)

```bash
sudo apt-get install mecab mecab-ko mecab-ko-dic libmecab-dev
pip install mecab-python3
```

---

## Usage

### 1. Parser Comparison (CER/WER/Structure F1)

```bash
# Run all parsers on all test data
python -m src.eval_parsers --all

# Single file test
python -m src.eval_parsers --input data/test_1/test.pdf --gt data/test_1/gt.md

# Baseline only (skip VLM-based Advanced parsers)
python -m src.eval_parsers --all --skip-advanced

# Text parsers only (skip Image/RapidOCR parsers)
python -m src.eval_parsers --all --skip-image
```

**Output**: `results/test_X/` with parsed files, `evaluation.json`, and `README.md` summary.

### 2. Chunking Quality Evaluation (BC/CS)

```bash
# Evaluate all parsed results
python -m src.eval_chunking --all

# With specific breakpoint settings
python -m src.eval_chunking --all \
    --breakpoint-type percentile \
    --breakpoint-threshold 90

# Mock mode (no embedding API needed)
python -m src.eval_chunking --all --use-mock

# Evaluate specific directory
python -m src.eval_chunking --parsed-dir results/test_1/
```

**Output**: `chunking.json` in each test folder with BC/CS scores per parser.

### 3. Streamlit Dashboard

```bash
streamlit run src/dashboard_analysis.py --server.port 8501
```

---

## Evaluation Metrics

### Stage 1: Prerequisite Validation (CER, WER)

CER/WER verify that text extraction quality is **sufficient for VLM input**, not the primary VLM performance metric.

| Metric | Formula | Description |
|--------|---------|-------------|
| **CER** | `(S + D + I) / N` | Character Error Rate |
| **WER** | `(S + D + I) / N` | Word Error Rate (with morphological tokenization) |

### Stage 2: Structure Preservation (Structure F1)

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | `TP / (TP + FP)` | Are generated structure elements correct? (monitors hallucination) |
| **Recall** | `TP / (TP + FN)` | Are GT structure elements detected? (monitors omission) |
| **Structure F1** | `2 * P * R / (P + R)` | Harmonic mean of Precision and Recall |

**Why separate Precision/Recall?**: F1 alone cannot distinguish over-generation (FP) from omission (FN). VLMs can hallucinate structure elements, making Precision monitoring critical.

**Structure elements evaluated**: Headings (`^#{1,6}\s+`), Lists (`^[-*+]\s+`, `^\d+\.\s+`), Tables (`^\|.+\|$`)

### Stage 3: Chunking Quality (MoC-based, Semantic Distance)

| Metric | Formula | Description |
|--------|---------|-------------|
| **BC** | `1 - cosine_similarity` | Higher is better (chunks are independent) |
| **CS** | `-Σ (h_i/2m) * log2(h_i/2m)` | Lower is better (Structural Entropy) |

---

## Experimental Results

### RQ1: Prerequisite Check — CER (test_3, Attention Is All You Need)

| Parser | CER | WER | Note |
|--------|-----|-----|------|
| **Image-Advanced** | **33.09%** | **43.48%** | **Best CER** |
| Image-Baseline | 40.79% | 51.55% | |
| Text-Baseline | 51.25% | 62.89% | |
| Text-Advanced | 64.11% | 75.26% | Highest CER (structural markup cost) |

**Warning**: Korean scanned documents (test_1) showed CER 536% hallucination with Image-Advanced.

### RQ2: Structure Preservation — Structure F1 (test_3)

| Parser | Precision | Recall | F1 | TP | FP | FN |
|--------|-----------|--------|-----|----|----|-----|
| Text-Baseline | 0% | 0% | 0% | 0 | 11 | 24 |
| Image-Baseline | 0% | 0% | 0% | 0 | 0 | 24 |
| **Text-Advanced** | **72.41%** | **87.50%** | **79.25%** | 21 | 8 | 3 |
| Image-Advanced | 70.00% | 87.50% | 77.78% | 21 | 9 | 3 |

### RQ3: Downstream Impact

- BC score 0.512 with 18 natural chunk divisions (test_3)
- Markdown headers provide natural semantic boundaries for chunking

### RQ4: Retrieval Impact (In Progress)

Verifying whether improved chunking quality from structure-preserved parsing actually increases retrieval precision (Hit Rate@k, MRR). Expected completion: February 2026.

### Trade-off Summary

| Metric | Text-Baseline | Text-Advanced | Image-Advanced |
|--------|--------------|---------------|----------------|
| CER | 51.25% | 64.11% | **33.09%** |
| Structure F1 | 0% | 79.25% | **77.78%** |
| Latency | 2.31s | 42.92s | **35.75s** |

**Recommendation**: **Image-Advanced is the balanced choice** — Best CER (33%), competitive Structure F1 (78% vs 79%), and 17% faster latency than Text-Advanced. For offline document preprocessing pipelines (one-time indexing), both Advanced parsers are viable, but Image-Advanced offers the best overall trade-off.
- **Speed-critical (real-time)**: Baseline (0.27-2.3s)
- **Structure-critical (RAG indexing)**: Image-Advanced (F1 78%, CER 33%)
- **Hybrid**: Route by document complexity

---

## Project Structure

```
test-vlm-document-parsing/
├── src/
│   ├── eval_parsers.py          # CLI: Parser comparison (CER/WER/F1)
│   ├── eval_chunking.py         # CLI: Chunking evaluation (BC/CS)
│   ├── dashboard_analysis.py    # Streamlit dashboard
│   │
│   ├── parsers/                 # Parser implementations
│   │   ├── ocr_parser.py        # Text-Baseline (PyMuPDF), Image-Baseline (RapidOCR)
│   │   ├── text_structurer.py   # VLM-based text structuring (Qwen3-VL-2B)
│   │   └── two_stage_parser.py  # Advanced parsers (Baseline + VLM)
│   │
│   ├── chunking/                # Semantic chunking module
│   │   ├── chunker.py           # LangChain SemanticChunker wrapper
│   │   ├── embeddings.py        # LangChain-compatible API embeddings
│   │   ├── metrics.py           # BC/CS metrics (Semantic Distance)
│   │   └── dashboard_export.py  # Dashboard data export utilities
│   │
│   ├── dashboard/               # Dashboard components
│   │   └── styles.py            # CSS styles
│   │
│   └── evaluation/              # Unified evaluation interface
│       └── __init__.py
│
├── data/
│   ├── test_1/                  # Korean government document
│   ├── test_2/                  # Receipt image
│   ├── test_3/                  # English academic paper
│   └── test_4/                  # Additional test document
│
├── results/                     # Evaluation outputs
│   ├── test_1/
│   │   ├── text_baseline_output.txt
│   │   ├── image_baseline_output.txt
│   │   ├── text_advanced_output.txt
│   │   ├── image_advanced_output.txt
│   │   ├── evaluation.json      # Parsing metrics
│   │   ├── chunking.json        # Chunking metrics
│   │   └── README.md            # Summary
│   └── ...
│
├── docs/
│   └── tech_report/             # Full technical report (9 sections + appendices)
│
├── tests/
│   └── test_chunking_cli.py     # Unit tests for chunking module
│
├── pyproject.toml               # Project configuration
├── README.md                    # English documentation
└── README.ko.md                 # Korean documentation
```

---

## Configuration

### VLM Configuration

```yaml
vlm_structurer:
  model: "Qwen3-VL-2B-Instruct"
  api_url: "http://localhost:8005/v1/chat/completions"
  temperature: 0.1
  max_tokens: 8192
  prompt_version: "v2"  # CRITICAL RULES + explicit heading mapping
```

### Chunking Configuration

```python
from src.chunking import ChunkerConfig, create_chunker

config = ChunkerConfig(
    breakpoint_threshold_type="percentile",  # percentile, standard_deviation, interquartile, gradient
    breakpoint_threshold_amount=95.0,        # Threshold value
    min_chunk_size=None,                     # Optional minimum chunk size
)

chunker = create_chunker(
    config=config,
    embedding_api_url="http://localhost:8001/embeddings",
    embedding_model="BAAI/bge-m3",
)
```

### Embedding Client Options

```python
from src.chunking.metrics import create_embedding_client

# API-based (Infinity with BGE-M3)
client = create_embedding_client(
    api_url="http://localhost:8001/embeddings",
    model="BAAI/bge-m3"
)

# Local sentence-transformers
client = create_embedding_client(
    model="jhgan/ko-sroberta-multitask"
)

# Mock for testing
client = create_embedding_client(use_mock=True)
```

---

## API Reference

### Parser Classes

```python
from src.parsers import (
    OCRParser,          # Text-Baseline (PyMuPDF)
    RapidOCRParser,     # Image-Baseline (RapidOCR)
    TwoStageParser,     # Advanced parsers (Baseline + VLM)
    TextStructurer,     # VLM text structuring
)

# Two-Stage Parser (Advanced)
parser = TwoStageParser(
    structurer_api_url="http://localhost:8005/v1/chat/completions",
    structurer_model="qwen3-vl-2b-instruct",
)

# Auto-detect PDF type and parse
result = parser.parse_auto(pdf_bytes)
print(f"Content: {result.content}")
print(f"Stage 1 ({result.stage1_parser}): {result.stage1_time:.2f}s")
print(f"Stage 2 (VLM): {result.stage2_time:.2f}s")
```

### Chunking & Evaluation

```python
from src.chunking import (
    SemanticChunker,
    ChunkerConfig,
    evaluate_chunking,
    create_embedding_client,
)

# Chunk text
config = ChunkerConfig(breakpoint_threshold_type="percentile")
chunker = SemanticChunker(config, embedding_api_url="http://localhost:8001/embeddings")
chunks = chunker.chunk(text, document_id="doc1")

# Evaluate chunking quality
client = create_embedding_client(api_url="http://localhost:8001/embeddings")
metrics = evaluate_chunking(chunks, embedding_client=client)

print(f"BC: {metrics.bc_score.score:.4f}")
print(f"CS: {metrics.cs_score.score:.4f}")
```

---

## Dependencies

Core dependencies:

```toml
[project]
dependencies = [
    "PyMuPDF>=1.24.0",           # Text-Baseline parser
    "rapidocr-pdf>=0.4.0",       # Image-Baseline parser
    "rapidocr-onnxruntime>=1.4.0",
    "langchain-experimental>=0.3.0",  # SemanticChunker
    "langchain-core>=0.3.0",
    "httpx>=0.27.0",             # API client
    "jiwer>=3.0.0",              # CER/WER calculation
    "numpy>=1.24.0",
    "streamlit>=1.45.0",         # Dashboard
    "plotly>=5.18.0",            # Visualization
]
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## References

- [MoC Paper](https://arxiv.org/abs/2503.09600) - Mixtures of Chunking (BC/CS metrics)
- [LangChain SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embedding model
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF text extraction
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR engine

---

## License

MIT License

---

## Citation

```bibtex
@software{document_parsing_structure,
  title = {Document Parsing Structure Preservation Test Framework},
  author = {Kim, Hyeongseob},
  year = {2026},
  url = {https://github.com/Hyeongseob91/test-vlm-document-parsing}
}
```
