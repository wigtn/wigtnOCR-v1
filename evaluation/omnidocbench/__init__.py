"""OmniDocBench evaluation pipeline — internalized from official code.

References:
- OmniDocBench (Ouyang et al., CVPR 2025): https://arxiv.org/abs/2412.07626
- TEDS (Zhong et al., 2019): Tree Edit Distance based Similarity
"""

from evaluation.omnidocbench.evaluator import OmniDocBenchEvaluator

__all__ = ["OmniDocBenchEvaluator"]
