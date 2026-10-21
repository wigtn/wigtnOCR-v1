# Third-party code: CDM

This folder is the CDM (Character Detection Matching) formula metric from
[opendatalab/UniMERNet](https://github.com/opendatalab/UniMERNet/tree/main/cdm),
commit `5a2c80d96b1d2dba447ff18d873e5fb73ba03c35`, licensed under Apache-2.0.
It bundles KaTeX and match-at (MIT) under `modules/tokenize_latex/third_party/`.

Changes from upstream:

- `evaluation.py`: `ransac(..., random_state=42)` became `ransac(..., rng=42)` to match the current
  scikit-image API (0.21 or later; the upstream `requirements.txt` kept here still pins `<=0.20.0`).
  The seed is unchanged.
- `assets/demo/` (README images) is not included.

`scripts/eval_omnidocbench/run_cdm.py` calls `evaluation.py` in this folder. CDM needs Node.js,
ImageMagick and a LaTeX installation with `pdflatex`; see `README.md` here.
