# Document Parsing Structure Preservation Test Framework

> **"Structure lost at parsing is structure lost forever — no downstream optimization can recover it."**

In enterprise RAG pipelines handling complex documents — nested tables, image-embedded tables, multi-column layouts — we discovered that **the parsing stage is the irreversible bottleneck**. Once document structure is lost during parsing, no amount of downstream optimization (chunking strategies, embedding model swaps, reranker additions) can recover it. This framework quantitatively validates that hypothesis by comparing traditional OCR methods against VLM-based structured parsing across the full pipeline.

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

**Root Cause**: All of these symptoms originate from a single point of failure — **structural information loss at the Data Parsing stage**, the very first step of the pipeline. Changing chunking strategies, swapping embedding models, or adding rerankers cannot recover structure that was already discarded during parsing.

### Hypothesis

> If the first stage of the pipeline (Data Parsing) preserves document structure, then the same downstream processing (chunking, embedding, retrieval) achieves significantly higher quality. Conversely, if structure is lost at parsing, no downstream optimization can overcome that ceiling.

### Core Research Questions

| RQ | Question | Metrics |
|----|----------|---------|
| **RQ1** | How much document structure does traditional OCR lose? | CER, WER, Structure F1 |
| **RQ2** | Can VLM-based parsing preserve that structure? | BC (Boundary Clarity), CS (Chunk Stickiness) |
| **RQ3** | Does structure preservation at parsing determine downstream (chunking, retrieval) quality? | Hit Rate@k, MRR |

---

## Key Features

### 4-Parser Architecture

A controlled experiment design with 4 parsers: **Baseline** (no structure preservation) vs **Advanced** (VLM-based structure preservation), across both **Text** (digital PDF) and **Image** (scanned PDF) extraction paths. This isolates the effect of structure preservation regardless of the extraction method.

| Parser | Description | Stage 1 | Stage 2 |
|--------|-------------|---------|---------|
| **Text-Baseline** | Digital PDF text extraction | PyMuPDF | - |
| **Image-Baseline** | Scanned PDF OCR | RapidOCR | - |
| **Text-Advanced** | Digital + VLM structuring | PyMuPDF | VLM (Qwen3-VL) |
| **Image-Advanced** | Scanned + VLM structuring | RapidOCR | VLM (Qwen3-VL) |

### Semantic Chunking

Uses LangChain's SemanticChunker with embedding-based boundary detection:

- **Breakpoint Types**: percentile, standard_deviation, interquartile, gradient
- **Embedding Backend**: BGE-M3 via Infinity API or local sentence-transformers
- **Automatic Sizing**: Chunk size determined by semantic boundaries

### MoC-based Quality Metrics (Label-Free)

Measures the downstream impact of parsing structure preservation on chunking quality. Based on the MoC paper (arXiv:2503.09600v2), using **Semantic Distance** instead of Perplexity:

| Metric | Implementation | Interpretation |
|--------|---------------|----------------|
| **BC (Boundary Clarity)** | `1 - cosine_similarity(chunk_i, chunk_i+1)` | Higher = more independent (good) |
| **CS (Chunk Stickiness)** | Structural Entropy of chunk graph | Lower = less dependency (good) |

**Note**: Semantic Distance BC scores (0.3-0.5) differ from MoC's Perplexity-based scores (0.8+) due to different metric scales.

### Comprehensive Evaluation

- **Lexical Metrics**: CER, WER with Korean morphological analysis (MeCab)
- **Structure Metrics**: Structure F1 (Precision, Recall for markdown elements)
- **Chunking Metrics**: BC, CS without Ground Truth
- **Retrieval Metrics**: Hit Rate@k, MRR with statistical significance testing

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
|                 Impact Cascade Evaluation                          |
|   +----------+   +-----------+   +----------+   +----------+      |
|   | Lexical  |   | Structure |   | Chunking |   | Retrieval|      |
|   | CER, WER |   |    F1     |   |  BC, CS  |   | HR@k,MRR|      |
|   +----------+   +-----------+   +----------+   +----------+      |
|                                                                    |
|   Parsing quality ──────────────────────→ Downstream quality       |
+--------------------------------------------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for VLM inference)
- MeCab (for Korean tokenization)

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
# Deploy Qwen3-VL or similar VLM at http://localhost:8005
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

### Phase 1: Lexical Accuracy

| Metric | Formula | Description |
|--------|---------|-------------|
| **CER** | `(S + D + I) / N` | Character Error Rate |
| **WER** | `(S + D + I) / N` | Word Error Rate (with morphological tokenization) |
| **Structure F1** | `2 * P * R / (P + R)` | F1 score for markdown structure elements |

### Phase 2: Chunking Quality (MoC-based, Semantic Distance)

| Metric | Formula | Description |
|--------|---------|-------------|
| **BC** | `1 - cosine_similarity` | Higher is better (chunks are independent) |
| **CS** | `-Σ (h_i/2m) * log2(h_i/2m)` | Lower is better (Structural Entropy) |

**Key Advantages**:
- No Ground Truth required
- Repeatable measurements in production
- Strong correlation with RAG performance

**BC Score Interpretation** (Semantic Distance scale):
| Range | Quality |
|-------|---------|
| > 0.5 | Excellent - highly independent chunks |
| 0.3 - 0.5 | Good - normal semantic chunking |
| < 0.3 | Needs review - chunks may be too similar |

### Phase 3: Retrieval Performance

| Metric | Formula | Description |
|--------|---------|-------------|
| **Hit Rate@k** | `hits_in_top_k / total_queries` | Relevant chunk in top-k results |
| **MRR** | `(1/N) * Σ(1/rank_i)` | Mean Reciprocal Rank |

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
│   │   ├── text_structurer.py   # VLM-based text structuring
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
├── tests/
│   └── test_chunking_cli.py     # Unit tests for chunking module
│
├── pyproject.toml               # Project configuration
├── README.md                    # English documentation
└── README.ko.md                 # Korean documentation
```

---

## Configuration

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
  year = {2025},
  url = {https://github.com/Hyeongseob91/test-vlm-document-parsing}
}
```
