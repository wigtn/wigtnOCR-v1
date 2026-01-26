"""
Plotly chart functions for the dashboard.

All charts follow the dark theme design system.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .styles import COLORS, PLOTLY_LAYOUT, get_parser_color


# =============================================================================
# Utility Functions
# =============================================================================

def apply_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent theme layout to a figure."""
    # Copy layout and override title
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "title"}
    layout["title"] = {"text": title, "x": 0, "xanchor": "left"}
    layout["height"] = height

    fig.update_layout(**layout)
    return fig


def get_parser_color_sequence(parsers: List[str]) -> List[str]:
    """Get ordered color sequence for parsers."""
    return [get_parser_color(p) for p in parsers]


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =============================================================================
# Parser Performance Charts
# =============================================================================

def create_grouped_bar_chart(
    df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Parser Performance Comparison",
    height: int = 450,
) -> go.Figure:
    """Create grouped bar chart comparing parser metrics.

    Args:
        df: DataFrame with columns [parser, metric1, metric2, ...]
        metrics: List of metric columns to display
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly Figure
    """
    if metrics is None:
        metrics = ["wer", "bleu", "char_acc"]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )
        return apply_layout(fig, title, height)

    # Prepare data for grouped bar chart
    parsers = df["parser"].unique().tolist()
    colors = get_parser_color_sequence(parsers)

    fig = go.Figure()

    for i, parser in enumerate(parsers):
        parser_data = df[df["parser"] == parser]
        values = [parser_data[metric].values[0] if metric in parser_data.columns else 0 for metric in metrics]

        fig.add_trace(go.Bar(
            name=parser,
            x=metrics,
            y=values,
            marker_color=colors[i],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"], size=10),
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    # Add direction annotations
    annotations = []
    metric_directions = {
        "wer": ("Lower is better", COLORS["good"]),
        "cer": ("Lower is better", COLORS["good"]),
        "bleu": ("Higher is better", COLORS["good"]),
        "char_acc": ("Higher is better", COLORS["good"]),
        "latency": ("Lower is better", COLORS["good"]),
    }

    for i, metric in enumerate(metrics):
        if metric.lower() in metric_directions:
            direction, color = metric_directions[metric.lower()]
            annotations.append(dict(
                x=i,
                y=-0.15,
                xref="x",
                yref="paper",
                text=direction,
                showarrow=False,
                font=dict(size=9, color=color),
            ))

    fig.update_layout(annotations=annotations)

    return apply_layout(fig, title, height)


def create_metrics_comparison_chart(
    df: pd.DataFrame,
    test_ids: Optional[List[str]] = None,
    parsers: Optional[List[str]] = None,
    title: str = "Metrics by Test Case",
    height: int = 500,
) -> go.Figure:
    """Create faceted bar chart showing metrics per test case.

    Args:
        df: DataFrame with columns [test_id, test_name, parser, wer, bleu, ...]
        test_ids: Filter to specific tests
        parsers: Filter to specific parsers
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    if test_ids:
        df = df[df["test_id"].isin(test_ids)]
    if parsers:
        df = df[df["parser"].isin(parsers)]

    # Melt the dataframe for plotting
    metrics = ["wer", "bleu", "char_acc"]
    available_metrics = [m for m in metrics if m in df.columns]

    df_melted = df.melt(
        id_vars=["test_id", "test_name", "parser"],
        value_vars=available_metrics,
        var_name="metric",
        value_name="value"
    )

    # Create faceted chart
    fig = px.bar(
        df_melted,
        x="parser",
        y="value",
        color="parser",
        facet_col="metric",
        facet_row="test_name",
        color_discrete_map={p: get_parser_color(p) for p in df["parser"].unique()},
        text="value",
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)

    return apply_layout(fig, title, height)


# =============================================================================
# Chunking Metrics Charts
# =============================================================================

def create_box_plot(
    df: pd.DataFrame,
    metric: str,
    title: str = "",
    height: int = 350,
    show_direction: bool = True,
) -> go.Figure:
    """Create box plot for a single metric.

    Args:
        df: DataFrame with metric column
        metric: Column name for the metric
        title: Chart title
        height: Chart height
        show_direction: Show "higher/lower is better" annotation

    Returns:
        Plotly Figure
    """
    if df.empty or metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text_secondary"])
        )
        return apply_layout(fig, title, height)

    fig = go.Figure()

    fig.add_trace(go.Box(
        y=df[metric],
        name=metric.upper(),
        marker_color=COLORS["vlm_color"],
        boxmean=True,
    ))

    # Add individual points
    fig.add_trace(go.Scatter(
        y=df[metric],
        mode="markers",
        name="Values",
        marker=dict(
            color=COLORS["text_secondary"],
            size=8,
            opacity=0.6,
        ),
        showlegend=False,
    ))

    if show_direction:
        direction_map = {
            "bc": "Higher is better",
            "cs": "Lower is better",
        }
        if metric.lower() in direction_map:
            fig.add_annotation(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=direction_map[metric.lower()],
                showarrow=False,
                font=dict(size=11, color=COLORS["good"]),
            )

    return apply_layout(fig, title or f"{metric.upper()} Distribution", height)


def create_dual_box_plot(
    df: pd.DataFrame,
    title: str = "Chunking Metrics Distribution",
    height: int = 400,
) -> go.Figure:
    """Create side-by-side box plots for BC and CS.

    Args:
        df: DataFrame with bc and cs columns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Boundary Clarity (BC)", "Chunk Stickiness (CS)"]
    )

    # BC Box Plot (higher is better)
    if "bc" in df.columns:
        fig.add_trace(go.Box(
            y=df["bc"],
            name="BC",
            marker_color=COLORS["vlm_color"],
            boxmean=True,
        ), row=1, col=1)

    # CS Box Plot (lower is better)
    if "cs" in df.columns:
        fig.add_trace(go.Box(
            y=df["cs"],
            name="CS",
            marker_color=COLORS["docling_color"],
            boxmean=True,
        ), row=1, col=2)

    fig.update_layout(showlegend=False)

    # Add direction annotations
    fig.add_annotation(
        x=0.25, y=-0.1,
        xref="paper", yref="paper",
        text="Higher is better",
        showarrow=False,
        font=dict(size=10, color=COLORS["good"]),
    )
    fig.add_annotation(
        x=0.75, y=-0.1,
        xref="paper", yref="paper",
        text="Lower is better",
        showarrow=False,
        font=dict(size=10, color=COLORS["good"]),
    )

    return apply_layout(fig, title, height)


def create_scatter_plot(
    df: pd.DataFrame,
    x: str = "bc",
    y: str = "cs",
    color_by: str = None,
    title: str = "BC vs CS Correlation",
    height: int = 400,
) -> go.Figure:
    """Create scatter plot for two metrics.

    Args:
        df: DataFrame with x and y columns
        x: X-axis column name
        y: Y-axis column name
        color_by: Column to use for color coding
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    if df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    if color_by and color_by in df.columns:
        fig = px.scatter(
            df, x=x, y=y,
            color=color_by,
            hover_data=["test_name"] if "test_name" in df.columns else None,
            color_discrete_sequence=[COLORS["vlm_color"], COLORS["ocr_color"], COLORS["docling_color"]],
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker=dict(
                color=COLORS["vlm_color"],
                size=12,
                opacity=0.7,
            ),
            text=df["test_name"] if "test_name" in df.columns else None,
            hovertemplate="%{text}<br>BC: %{x:.2f}<br>CS: %{y:.2f}<extra></extra>",
        ))

    # Add trend line
    if len(df) > 1:
        import numpy as np
        z = np.polyfit(df[x], df[y], 1)
        p = np.poly1d(z)
        x_line = [df[x].min(), df[x].max()]
        y_line = [p(x_line[0]), p(x_line[1])]

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color=COLORS["text_muted"], dash="dash"),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis_title=f"{x.upper()} (Higher is better)",
        yaxis_title=f"{y.upper()} (Lower is better)",
    )

    # Add quadrant annotations
    if len(df) > 0:
        x_mid = df[x].median()
        y_mid = df[y].median()

        fig.add_shape(
            type="line", x0=x_mid, x1=x_mid, y0=df[y].min() * 0.9, y1=df[y].max() * 1.1,
            line=dict(color=COLORS["grid"], dash="dot")
        )
        fig.add_shape(
            type="line", x0=df[x].min() * 0.9, x1=df[x].max() * 1.1, y0=y_mid, y1=y_mid,
            line=dict(color=COLORS["grid"], dash="dot")
        )

        # Best quadrant annotation (high BC, low CS)
        fig.add_annotation(
            x=df[x].max(), y=df[y].min(),
            text="Best",
            showarrow=False,
            font=dict(color=COLORS["good"], size=12),
            xanchor="right", yanchor="bottom",
        )

    return apply_layout(fig, title, height)


# =============================================================================
# Document Analysis Charts
# =============================================================================

def create_heatmap(
    df: pd.DataFrame,
    title: str = "Performance Heatmap",
    height: int = 450,
    colorscale: str = "RdYlGn_r",  # Red (bad) to Green (good) reversed for WER
) -> go.Figure:
    """Create heatmap for document x parser x metric visualization.

    Args:
        df: Pivot table DataFrame (rows=documents, cols=parsers)
        title: Chart title
        height: Chart height
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=colorscale,
        text=df.values.round(3),
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Document: %{y}<br>Parser: %{x}<br>Value: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Value", font=dict(color=COLORS["text_secondary"])),
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
    ))

    fig.update_layout(
        xaxis_title="Parser",
        yaxis_title="Document",
    )

    return apply_layout(fig, title, height)


def create_radar_chart(
    df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Parser Performance Radar",
    height: int = 450,
) -> go.Figure:
    """Create radar chart comparing parsers across metrics.

    Args:
        df: DataFrame with parser and metric columns
        metrics: List of metric columns to include
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    if metrics is None:
        metrics = ["wer", "bleu", "char_acc", "latency"]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    fig = go.Figure()

    # Normalize metrics for radar chart (0-1 scale)
    for parser in df["parser"].unique():
        parser_data = df[df["parser"] == parser]
        values = []

        for metric in metrics:
            if metric in parser_data.columns:
                val = parser_data[metric].values[0]
                # Invert metrics where lower is better
                if metric.lower() in ["wer", "cer", "latency"]:
                    val = 1 - min(val, 1)  # Clamp to 0-1 and invert
                values.append(val)
            else:
                values.append(0)

        # Close the radar shape
        values.append(values[0])
        labels = metrics + [metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            name=parser,
            fill="toself",
            fillcolor=hex_to_rgba(get_parser_color(parser), 0.25),
            line=dict(color=get_parser_color(parser)),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=COLORS["grid"],
                tickfont=dict(color=COLORS["text_secondary"]),
            ),
            angularaxis=dict(
                gridcolor=COLORS["grid"],
                tickfont=dict(color=COLORS["text_primary"]),
            ),
            bgcolor=COLORS["bg_secondary"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    return apply_layout(fig, title, height)


# =============================================================================
# Latency Charts
# =============================================================================

def create_latency_comparison(
    df: pd.DataFrame,
    title: str = "Parsing Latency Comparison",
    height: int = 350,
) -> go.Figure:
    """Create bar chart comparing parser latencies.

    Args:
        df: DataFrame with parser and latency columns
        title: Chart title
        height: Chart height

    Returns:
        Plotly Figure
    """
    if df.empty or "latency" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return apply_layout(fig, title, height)

    # Aggregate latency by parser
    latency_df = df.groupby("parser")["latency"].mean().reset_index()
    latency_df = latency_df.sort_values("latency")

    colors = [get_parser_color(p) for p in latency_df["parser"]]

    fig = go.Figure(go.Bar(
        x=latency_df["parser"],
        y=latency_df["latency"],
        marker_color=colors,
        text=[f"{v:.2f}s" for v in latency_df["latency"]],
        textposition="outside",
    ))

    fig.update_layout(
        xaxis_title="Parser",
        yaxis_title="Average Latency (seconds)",
    )

    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text="Lower is better",
        showarrow=False,
        font=dict(size=10, color=COLORS["good"]),
    )

    return apply_layout(fig, title, height)
