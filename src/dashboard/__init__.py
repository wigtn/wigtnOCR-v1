"""
VLM Document Parsing Analysis Dashboard

Streamlit-based dashboard for analyzing parser performance metrics.
"""

from .styles import COLORS, apply_dark_theme
from .data_loader import DataLoader, TestResult, ChunkingMetrics
from .charts import (
    create_grouped_bar_chart,
    create_box_plot,
    create_scatter_plot,
    create_heatmap,
)

__all__ = [
    "COLORS",
    "apply_dark_theme",
    "DataLoader",
    "TestResult",
    "ChunkingMetrics",
    "create_grouped_bar_chart",
    "create_box_plot",
    "create_scatter_plot",
    "create_heatmap",
]
