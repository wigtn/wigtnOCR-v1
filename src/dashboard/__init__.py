"""
VLM Document Parsing Analysis Dashboard

Streamlit-based dashboard for analyzing parser performance metrics.
"""

from .styles import COLORS, apply_dark_theme
from .data_loader import (
    load_results,
    load_results_cached,
    get_parsing_data,
    get_chunking_data,
    get_chunking_data_for_parser,
    get_chunking_parsers,
    get_parser_names,
    get_parsing_summary_df,
    get_chunking_summary_df,
    get_aggregated_parser_df,
    paginate_data,
    export_df_to_csv,
    get_chart_download_config,
    get_sample_data,
    ParsedData,
)
from .charts import (
    create_grouped_bar_chart,
    create_box_plot,
    create_scatter_plot,
    create_heatmap,
    # New chunking charts (MoC-based)
    STRATEGY_COLORS,
    get_strategy_color,
    create_parser_chunking_comparison,
    create_bc_document_flow,
    create_cs_mean_std_bar,
)

__all__ = [
    # Styles
    "COLORS",
    "apply_dark_theme",
    # Data loading
    "load_results",
    "load_results_cached",
    "get_parsing_data",
    "get_chunking_data",
    "get_chunking_data_for_parser",
    "get_chunking_parsers",
    "get_parser_names",
    "get_parsing_summary_df",
    "get_chunking_summary_df",
    "get_aggregated_parser_df",
    "paginate_data",
    "export_df_to_csv",
    "get_chart_download_config",
    "get_sample_data",
    "ParsedData",
    # Charts
    "create_grouped_bar_chart",
    "create_box_plot",
    "create_scatter_plot",
    "create_heatmap",
    # Chunking charts (MoC-based)
    "STRATEGY_COLORS",
    "get_strategy_color",
    "create_parser_chunking_comparison",
    "create_bc_document_flow",
    "create_cs_mean_std_bar",
]
