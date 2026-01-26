# VLM Document Parsing Quality Test Framework

> **"Does Structural Integrity in Parsing Improve Semantic Retrieval?"**

A comprehensive evaluation framework for quantitatively analyzing the impact of Vision-Language Model (VLM) structured markdown output on semantic chunking and RAG (Retrieval-Augmented Generation) performance compared to traditional OCR methods.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run parser comparison benchmark
python -m src.test_parsers --pdf data/test_1/test_data_1.pdf --gt data/test_1/gt_data_1.md

# Run full evaluation pipeline
python -m src.run_benchmark --config experiments/config.yaml
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
- [Tech Report](#tech-report)
- [Experimental Results](#experimental-results)
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

### Hypothesis

> VLM-based parsing produces structured markdown that preserves document layout, leading to better semantic chunking boundaries (higher Boundary Score) and improved chunk coherence (higher Chunk Score), ultimately resulting in better retrieval accuracy.

### Core Research Questions

| RQ | Question | Metrics |
|----|----------|---------|
| **RQ1** | Does VLM achieve better lexical accuracy? | CER, WER |
| **RQ2** | Does VLM preserve structure better? | Boundary Score, Chunk Score |
| **RQ3** | Does better parsing improve retrieval? | Hit Rate@k, MRR |

---

## Key Features

### Multi-Parser Support
- **VLM Parser** (Qwen3-VL): Structured markdown output with layout understanding
- **pdfplumber**: Fast digital PDF text extraction
- **Docling + RapidOCR**: Scanned document OCR

### Comprehensive Evaluation
- **Lexical Metrics**: CER, WER with Korean morphological analysis (MeCab)
- **Structural Metrics**: Boundary Score, Chunk Score
- **Retrieval Metrics**: Hit Rate@k, MRR with statistical significance testing
- **Error Analysis**: Automatic error detection, categorization, and case study generation

### Research Infrastructure
- **Tech Report Template**: Complete academic report structure
- **Experiment Configuration**: YAML-based reproducible experiments
- **Q&A Generation**: LLM-powered evaluation dataset creation
- **Benchmark Runner**: Automated end-to-end evaluation pipeline

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Document Input                                │
│                     (PDF / Image / Scan)                            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Parsing Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │    VLM      │  │ pdfplumber  │  │   Docling   │                 │
│  │ (Qwen3-VL)  │  │  (Digital)  │  │ (RapidOCR)  │                 │
│  │  Markdown   │  │ Plain Text  │  │ Plain Text  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Chunking Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Fixed     │  │  Recursive  │  │  Semantic   │                 │
│  │    Size     │  │  Character  │  │   (Topic)   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Evaluation Layer                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │  Lexical  │  │ Structural│  │ Retrieval │  │   Error   │       │
│  │ CER, WER  │  │  BS, CS   │  │ HR@k, MRR │  │ Analysis  │       │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (for VLM inference)
- MeCab (for Korean tokenization)

### Using uv (Recommended)

```bash
git clone https://github.com/your-repo/test-vlm-document-parsing.git
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

### MeCab Installation (Ubuntu)

```bash
sudo apt-get install mecab mecab-ko mecab-ko-dic libmecab-dev
pip install mecab-python3
```

---

## Usage

### 1. Basic Parser Comparison

```bash
# Compare all parsers on a document
python -m src.test_parsers \
    --pdf data/test_1/test_data_1.pdf \
    --gt data/test_1/gt_data_1.md \
    --tokenizer mecab

# Skip VLM (when GPU unavailable)
python -m src.test_parsers --pdf document.pdf --gt ground_truth.md --skip-vlm
```

### 2. Chunking Quality Evaluation (BC/CS)

Evaluate chunking quality using MoC-based metrics without Ground Truth:

```bash
# Basic evaluation with PDF input
python -m src.test_chunking --input data/test_1/test_data_1.pdf

# With mock LLM (no API required, for testing)
python -m src.test_chunking --input data/test.pdf --use-mock

# Evaluate pre-parsed files
python -m src.test_chunking --parsed-files data/test_1/gt_data_1.md --use-mock

# Full evaluation with complete graph
python -m src.test_chunking \
    --input data/test.pdf \
    --graph-type complete \
    --threshold-k 0.7 \
    --output-dir results/chunks/

# Skip VLM parser (OCR only)
python -m src.test_chunking --input data/test.pdf --skip-vlm --use-mock
```

**Output**: JSON chunks, evaluation.json, README.md summary in `results/chunks/<timestamp>/`

### 3. Full Benchmark Pipeline

```bash
# Run complete evaluation with config
python -m src.run_benchmark --config experiments/config.yaml

# Run on single document
python -m src.run_benchmark \
    --pdf data/test_1/test_data_1.pdf \
    --gt data/test_1/gt_data_1.md \
    --qa data/qa_pairs.json \
    --output results/
```

### 4. Generate Q&A Dataset

```bash
# Generate Q&A pairs from ground truth documents
python -m experiments.generate_qa \
    --config experiments/config.yaml \
    --output data/qa_pairs.json

# Use specific LLM provider
python -m experiments.generate_qa --provider openai --questions-per-doc 15
```

### 5. Streamlit Web UI

```bash
streamlit run src/app.py --server.port 8501
```

---

## Evaluation Metrics

### Phase 1: Lexical Accuracy

| Metric | Formula | Description |
|--------|---------|-------------|
| **CER** | `(S + D + I) / N` | Character Error Rate |
| **WER** | `(S + D + I) / N` | Word Error Rate (with morphological tokenization) |

- `S`: Substitutions (incorrect characters/words)
- `D`: Deletions (missing content)
- `I`: Insertions (hallucinated content)
- `N`: Total characters/words in ground truth

### Phase 2: Structural Integrity (MoC-based)

Based on the MoC paper (arXiv:2503.09600v2), we use label-free metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **BC (Boundary Clarity)** | `ppl(q\|d) / ppl(q)` | Higher is better - chunks are independent |
| **CS (Chunk Stickiness)** | Structural Entropy | Lower is better - less inter-chunk dependency |

**Key Advantages**:
- No Ground Truth required
- Repeatable measurements in production
- Strong correlation with RAG performance (BC↔ROUGE-L: 0.88)

### Phase 3: Retrieval Performance

| Metric | Formula | Description |
|--------|---------|-------------|
| **Hit Rate@k** | `hits_in_top_k / total_queries` | Relevant chunk in top-k results |
| **MRR** | `(1/N) * Σ(1/rank_i)` | Mean Reciprocal Rank |

### Quality Thresholds

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| CER | < 2% | < 5% | < 10% |
| WER | < 5% | < 10% | < 20% |
| BS | > 0.9 | > 0.8 | > 0.7 |
| Hit Rate@5 | > 90% | > 75% | > 60% |

---

## Project Structure

```
test-vlm-document-parsing/
├── src/
│   ├── app.py                    # Streamlit Web UI
│   ├── test_parsers.py           # CLI parser comparison tool (CER/WER)
│   ├── test_chunking.py          # CLI chunking evaluation (BC/CS)
│   ├── run_benchmark.py          # Full evaluation pipeline
│   │
│   ├── parsers/                  # Parser implementations
│   │   ├── vlm_parser.py         # Qwen3-VL integration
│   │   ├── ocr_parser.py         # pdfplumber parser
│   │   └── docling_parser.py     # Docling + RapidOCR
│   │
│   ├── chunking/                 # Text chunking module
│   │   ├── chunker.py            # Chunking strategies
│   │   └── metrics.py            # BC/CS metrics (MoC-based)
│   │
│   ├── retrieval/                # Retrieval evaluation
│   │   ├── embedder.py           # Text embeddings
│   │   ├── retriever.py          # Semantic search
│   │   └── evaluator.py          # HR@k, MRR metrics
│   │
│   ├── error_analysis/           # Error analysis module
│   │   ├── analyzer.py           # Error detection
│   │   ├── diff_visualizer.py    # HTML diff generation
│   │   └── case_study.py         # Case study generator
│   │
│   └── evaluation/               # Unified evaluation interface
│       └── __init__.py           # Integration module
│
├── experiments/
│   ├── config.yaml               # Experiment configuration
│   └── generate_qa.py            # Q&A dataset generator
│
├── data/
│   ├── test_1/                   # Korean government document
│   ├── test_2/                   # Receipt image
│   ├── test_3/                   # English academic paper
│   └── qa_pairs.json             # Generated Q&A dataset
│
├── docs/
│   └── tech_report/              # Technical report sections
│       ├── 00_Abstract.md
│       ├── 01_Introduction.md
│       ├── 02_Related_Work.md
│       ├── 03_Methodology.md
│       ├── 04_Experimental_Setup.md
│       ├── 05_Results.md
│       ├── 06_Discussion.md
│       ├── 07_Conclusion.md
│       ├── 08_References.md
│       ├── appendix/
│       └── figures/
│
├── results/                      # Evaluation outputs
│   ├── retrieval/
│   ├── structure/
│   ├── errors/
│   └── ablation/
│
├── _drafts/                      # Legacy evaluation modules
├── pyproject.toml
├── README.md
└── README.ko.md                  # Korean documentation
```

---

## Tech Report

The framework includes a complete academic tech report template:

| Section | Content |
|---------|---------|
| **Abstract** | Research summary, key findings |
| **Introduction** | Problem definition, research questions, contributions |
| **Related Work** | VLM, OCR, RAG literature review |
| **Methodology** | Evaluation framework, metric formulas |
| **Experimental Setup** | Dataset, parser configs, parameters |
| **Results** | Tables and analysis (template) |
| **Discussion** | RQ answers, error patterns, limitations |
| **Conclusion** | Findings, hybrid strategy recommendation |
| **References** | 26 academic citations |
| **Appendix** | Prompt variations, full results, case studies |

---

## Experimental Results

### Preliminary Results (test_3 - Scanned PDF)

| Parser | CER | WER | Latency |
|--------|-----|-----|---------|
| **VLM (Qwen3-VL)** | 56.29% | 70.85% | 15.61s |
| pdfplumber | 99.62% | 100% | 18.12s |
| RapidOCR | N/A | N/A | 6.85s |

**Key Finding**: For scanned documents, VLM is the only viable option.

### Hybrid Strategy Recommendation

```
Document Input
     │
     ▼
  Scanned? ──Yes──► VLM (Required)
     │
    No
     │
     ▼
  Complex Layout? ──Yes──► VLM (Recommended)
  (Tables, Multi-column)
     │
    No
     │
     ▼
  pdfplumber (Fast, Sufficient)
```

---

## Prompt Engineering

### Problem: VLM Hallucination

Early prompts caused VLM to add summaries and explanations, increasing insertion errors.

### Solution: Transcription-focused Prompt (v2)

```
You are a document TRANSCRIPTION engine, not a writer.
Your job is to CONVERT the given document image into Markdown by
STRICTLY TRANSCRIBING what is visible in the image.

## Hard Constraints:
- DO NOT add, rephrase, summarize, infer, or translate any text.
- DO NOT explain, comment, or describe what you are doing.
- If something is unreadable, write `[UNREADABLE]` instead of guessing.
- If a value is missing, use `[EMPTY]`. Never invent values.

## Output:
(Write ONLY the transcribed Markdown content here.)
```

---

## Configuration

All experiments are controlled via `experiments/config.yaml`:

```yaml
# Chunking (fixed for fair comparison)
chunking:
  strategy: "recursive_character"
  chunk_size: 500
  chunk_overlap: 50

# Embedding
embedding:
  model: "jhgan/ko-sroberta-multitask"
  device: "cuda"

# Retrieval
retrieval:
  top_k: [1, 3, 5, 10]

# Quality thresholds
thresholds:
  cer:
    excellent: 0.02
    good: 0.05
```

---

## API Reference

### UnifiedEvaluator

```python
from src.evaluation import UnifiedEvaluator

evaluator = UnifiedEvaluator(config)

# Full evaluation
results = evaluator.evaluate_full(
    parsed_text=parsed_content,
    ground_truth=gt_content,
    qa_pairs=qa_data,
    document_id="test_1",
    parser_name="vlm"
)

# Individual metrics
lexical = evaluator.evaluate_lexical(parsed, gt)
structural = evaluator.evaluate_structural(parsed, gt)
retrieval = evaluator.evaluate_retrieval(parsed, qa_pairs)
errors = evaluator.analyze_errors(parsed, gt)
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

- [jiwer](https://github.com/jitsi/jiwer) - CER/WER calculation
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction
- [Docling](https://github.com/DS4SD/docling) - Document understanding
- [KoNLPy](https://konlpy.org/) - Korean NLP toolkit
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - Vision-Language Model
- [sentence-transformers](https://www.sbert.net/) - Text embeddings

---

## License

MIT License

---

## Citation

```bibtex
@software{vlm_document_parsing,
  title = {VLM Document Parsing Quality Test Framework},
  year = {2025},
  url = {https://github.com/your-repo/test-vlm-document-parsing}
}
```
