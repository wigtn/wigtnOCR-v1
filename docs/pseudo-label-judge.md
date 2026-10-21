# Pseudo-label judge: rubric, filtering and bias correction

WigtnOCR-2B is trained on Markdown that a teacher model (Qwen3-VL-30B-A3B-Thinking) wrote for each page.
Before training, a second model scores those pseudo-labels, low-scoring pages are removed, and one
over-represented document is capped. This page documents each step as it was run for the released model.

Code: [`training/gt_validator.py`](../training/gt_validator.py), [`training/prompts/templates.py`](../training/prompts/templates.py),
[`training/data_prep.py`](../training/data_prep.py). Commands: [`scripts/02_judge_filter.sh`](../scripts/02_judge_filter.sh).

## The judge

| | |
|---|---|
| Model | Qwen3.5-122B-A10B, NVFP4 build (served name `qwen3.5-122b`) |
| Serving | vLLM nightly, tensor parallel 2, `--reasoning-parser deepseek_r1` ([`scripts/serve_judge.sh`](../scripts/serve_judge.sh)) |
| Input | The Markdown pseudo-label only. The judge never sees the page image. |
| Decoding | temperature 0.1, up to 16,384 tokens (the reasoning parser moves thinking out of the answer) |
| Output | One JSON object per page: an overall score and five sub-scores, all integers 1–5, plus error notes |

The judge reads text only on purpose. The failures that mattered in the teacher output were visible in
the Markdown itself: leaked reasoning text, truncated pages, and broken tables. Comparing against the image
would have needed a vision-language judge.

## Rubric

The system prompt (`GT_VALIDATION_SYSTEM_PROMPT`) asks for five sub-scores and one overall score.

| Dimension | What it checks | Signals of a low score |
|---|---|---|
| `structure_quality` | Headings, lists and nesting are well formed and consistent | Broken `#` hierarchy, orphaned or mis-nested list items |
| `table_quality` | Tables use `\|` separators, a `\|---\|` header row and a constant column count | Missing header separator, rows with different column counts |
| `completeness_signals` | Signs that the page was cut short or filled with placeholders | Abrupt endings, `<!-- MISSING -->` or `<!-- FAILED -->` markers |
| `hallucination_signals` | Signs of model contamination | `<think>` tags, chain-of-thought text, repeated phrases, nonsense, "Here is the markdown" |
| `formatting_consistency` | One Markdown style throughout | Mixed list markers, inconsistent heading style or spacing |

Scoring guide given to the judge, for the overall `score`:

| Score | Meaning |
|:---:|---|
| 5 | Clean, well-structured Markdown with no quality issues |
| 4 | Minor formatting issues that do not affect usability |
| 3 | Noticeable issues but mostly usable |
| 2 | Significant quality problems affecting usability |
| 1 | Major issues: AI contamination, broken structure, or largely unusable |

**Only the overall score gates training.** A page is kept when `score >= 3`. The five sub-scores are stored
in `validation_report.json` for analysis and are not thresholded.

## Sampling and outcome

Judging every page with a 122B model was too slow for the 3,637 KoGovDoc pages, so KoGovDoc was sampled.
Pages that were not sampled, and pages whose judge call failed, enter training without a score.

| Dataset | Pages | Sampled (seed 42) | Scored | Score ≥ 3 | Pass rate | Removed |
|---|---:|---:|---:|---:|---:|---:|
| KoGovDoc (Korean government documents, 10 docs) | 3,637 | 1,047 (30%) | 1,030 | 774 | 75.1% | 256 |
| arXiv papers (39 docs) | 864 | 864 (100%) | 831 | 613 | 73.8% | 218 |

Mean scores over the scored pages:

| Dataset | structure | table | completeness | hallucination | formatting | overall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| KoGovDoc | 3.82 | 3.83 | 3.69 | 4.07 | 3.67 | 3.30 |
| arXiv | 3.67 | 4.22 | 2.71 | 4.14 | 3.61 | 3.04 |

All counts come from the two `validation_report.json` files of the v1 run. For arXiv, the pages that reach
data preparation are 613 passing + 33 unscored = 646; running `prepare_training_data.py --dry-run` on the
v1 data reproduces that number.

## Bias correction: per-document cap

One document, `kogov_008`, holds 53% of all KoGovDoc pages. Trained as is, the student would mostly learn that
document's layout and vocabulary.

`max_doc_ratio = 0.25` caps every document at 25% of its dataset's kept pages. A document above the cap is
randomly down-sampled to the cap (seed 42); others are kept whole. The cap is computed separately for KoGovDoc
and for arXiv, after the score filter (`_downsample_documents` in `training/data_prep.py`).

## From kept pages to the training split

1. Each kept page is rendered at 200 DPI and paired with its pseudo-label as one ms-swift sample. The sample
   uses the student prompts in `templates.py` (`VLM_TRAINING_SYSTEM_PROMPT` with `STUDENT_USER_PROMPT` or
   `STUDENT_USER_PROMPT_EN`).
2. The samples are shuffled and split 90/10 (seed 42). In the v1 run this gave 2,985 samples: 2,339 KoGovDoc
   and 646 arXiv.
3. A manual cleaning pass then removed leftover reasoning text from 283 samples and dropped 24 samples
   (`data_stats.json`). This pass is not scripted in this repository.

The result is **2,667 training and 294 validation pages**. The validation file is public as
[`Wigtn/KoGovDoc-Bench`](https://huggingface.co/datasets/Wigtn/KoGovDoc-Bench) (`val.jsonl`, 229 KoGovDoc + 65 arXiv
pages). The same 294 pages later became the evaluation corpus of the RCPS paper.
