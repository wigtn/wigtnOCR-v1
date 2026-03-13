# SilverForge

[![한국어](https://img.shields.io/badge/lang-한국어-blue.svg)](README.ko.md)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://upstage-silverforge-agent-yhuqpaatpux2pebsxntupw.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Hyeongseob91/upstage-silverforge-agent)

**Convert PDF to Structured Markdown** - Ground Truth Data Generation Tool for VLM/SLM Training

> Upstage Ambassador Season 2 Project

## Live Demo

**[Upstage-Silverforge-Agent](https://upstage-silverforge-agent-yhuqpaatpux2pebsxntupw.streamlit.app/)**

## Features

- **PDF to Markdown Conversion** - Using Upstage Document Parse API
- **Heading Hierarchy Restoration** - Rule-based approach with 95%+ accuracy
- **Quality Analysis** - Text, Structure, Semantic evaluation with Solar Pro
- **Notion-style UI** - Clean, minimal design
- **Cloud Storage** - Supabase integration for user authentication & data persistence
- **Batch Processing** - Multiple PDF upload and sequential processing

## Problem Statement

High-quality **Ground Truth (GT) data** is essential for training VLM (Vision Language Model) and SLM (Small Language Model). Traditional LaTeX → Pandoc conversion has significant limitations:

| Method | Coverage | Table Recognition | Limitation |
|--------|----------|-------------------|------------|
| LaTeX → Pandoc | 40-50% | Incomplete | Requires LaTeX source |
| PyMuPDF | 60-70% | Not supported | Structure loss |
| **Upstage Document Parse** | **100%** | **Perfect** | - |

## Solution

**SilverForge** leverages [Upstage Document Parse](https://console.upstage.ai/) API to:

1. **Direct PDF Parsing** - 100% coverage without LaTeX source
2. **Perfect Table Recognition** - Traditional: 0 tables → SilverForge: 10 tables
3. **Heading Hierarchy Restoration** - 95%+ accuracy with rule-based approach

```
PDF → parse_pdf() → Raw Markdown → refine_headings() → Silver Markdown
         ↑                              ↑
    Upstage API                   Rule-based (regex)
```

## Benchmark Results

**Test Document**: "Attention Is All You Need" (Vaswani et al., 2017)

| Metric | LaTeX+Pandoc | SilverForge |
|--------|--------------|-------------|
| Table Recognition | 0 | **10** |
| Heading Restoration | Manual required | **95%+ Auto** |
| Formula Preservation | Partial | **Complete** |
| Source Required | LaTeX required | **PDF only** |

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

### Environment Setup

```bash
cp .env.example .env
# Add your UPSTAGE_API_KEY to .env file
```

### Usage

```python
from silverforge import process

# PDF → Silver Markdown (one-step)
silver_md = process("paper.pdf")

# Or step-by-step
from silverforge import parse_pdf, refine_headings

raw_md = parse_pdf("paper.pdf")      # Upstage API call
silver_md = refine_headings(raw_md)  # Heading restoration
```

### Web UI (Streamlit)

```bash
streamlit run src/silverforge/app.py
```

**UI Features:**
- Notion-style clean design
- Email/Password authentication (Supabase)
- Multiple PDF upload (Drag & Drop)
- Real-time processing with progress bar
- Quality score visualization
- Individual/bulk download (ZIP)

## Dataset Curator Agent

An Agent that automatically evaluates the quality of Silver data.

```python
from silverforge import curate

result = curate(silver_md)
print(result)
# {
#   "pass": True,
#   "overall_score": 85,
#   "recommendation": "Ready for chunking - all checks passed"
# }
```

### Quality Check Items

1. **Text Quality**: Character/Word count, CER/WER (if original provided)
2. **Structure Quality**: Heading hierarchy, table structure, equation blocks
3. **Semantic Quality**: Logic/completeness/coherence evaluation via Solar Pro

## Tech Stack

- **Frontend**: Streamlit (Notion-style UI)
- **Backend**: Python
- **PDF Parsing**: Upstage Document Parse API
- **Quality Evaluation**: Upstage Solar Pro
- **Authentication**: Supabase Auth
- **Database**: Supabase PostgreSQL
- **Deployment**: Streamlit Cloud

## Project Structure

```
silverforge/
├── src/silverforge/
│   ├── __init__.py      # exports
│   ├── core.py          # parsing (parse_pdf, refine_headings)
│   ├── curator.py       # quality check (curate)
│   ├── database.py      # Supabase integration
│   └── app.py           # Streamlit UI
├── supabase/
│   └── schema.sql       # Database schema
├── example.py           # parsing example
├── example_curate.py    # quality check example
└── requirements.txt     # dependencies
```

## Deployment (Streamlit Cloud)

### 1. Supabase Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run `supabase/schema.sql`
3. Copy your **Project URL** and **anon key** from Settings > API

### 2. Streamlit Cloud Setup

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with these settings:
   - **Main file path**: `src/silverforge/app.py`
   - **Python version**: 3.11

4. Add secrets in Streamlit Cloud Dashboard:
```toml
UPSTAGE_API_KEY = "your_upstage_api_key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your_supabase_anon_key"
```

## Acknowledgements

- [Upstage](https://upstage.ai/) - Document Parse API, Solar Pro
- [Supabase](https://supabase.com/) - Auth & Database
- Upstage Ambassador Season 2 Program

## License

MIT
