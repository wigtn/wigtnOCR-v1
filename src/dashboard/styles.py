"""
Design system constants and styling utilities for the dashboard.

Clean Light Theme inspired by Streamlit Gallery examples.
"""

# =============================================================================
# Color Palette (Clean Light Theme)
# =============================================================================

COLORS = {
    # Background colors - Pure white
    "bg_primary": "#FFFFFF",
    "bg_secondary": "#FFFFFF",
    "bg_card": "#FFFFFF",

    # Parser colors (Vibrant & Distinct)
    # Note: These are legacy colors, use PARSER_COLORS dict for consistency
    "vlm_color": "#4F46E5",       # Indigo - VLM
    "ocr_color": "#059669",       # Emerald - OCR-Text
    "ocr_image_color": "#D97706", # Amber - OCR-Image

    # Chart colors (Multiple for scatter plots)
    "chart_colors": [
        "#2E8B57",  # Sea Green
        "#4169E1",  # Royal Blue
        "#FF8C00",  # Dark Orange
        "#DC143C",  # Crimson
        "#9370DB",  # Medium Purple
        "#20B2AA",  # Light Sea Green
        "#FF69B4",  # Hot Pink
        "#FFD700",  # Gold
    ],

    # Status colors
    "good": "#2E8B57",           # Green
    "warning": "#FF8C00",        # Orange
    "bad": "#DC143C",            # Crimson

    # Text colors - Dark for readability
    "text_primary": "#262730",   # Almost black
    "text_secondary": "#555555", # Dark gray
    "text_muted": "#888888",     # Medium gray
    "text_link": "#1E90FF",      # Dodger blue for links

    # Chart colors
    "grid": "#E8E8E8",
    "border": "#E0E0E0",

    # Bar colors for inline bars
    "bar_positive": "#90EE90",   # Light green
    "bar_negative": "#FFB6C1",   # Light pink
}


# TwoStage parser colors (added to COLORS dict for backward compatibility)
COLORS["twostage_text_color"] = "#7C3AED"   # Purple - TwoStage-Text
COLORS["twostage_image_color"] = "#0891B2"  # Cyan - TwoStage-Image
COLORS["ocr_image_color"] = "#D97706"       # Amber - OCR-Image (alias)

# Parser name to color mapping (Single Source of Truth)
# Colors chosen for clear visual distinction
PARSER_COLORS = {
    # VLM parser (Blue - distinct from others)
    "vlm": "#2563EB",
    "VLM": "#2563EB",
    # Text-Baseline / OCR-Text parser (Emerald Green)
    "ocr": "#059669",
    "OCR": "#059669",
    "OCR-Text": "#059669",
    "ocr-text": "#059669",
    "Text-Baseline": "#059669",
    "text-baseline": "#059669",
    # OCR-Image / Image-Baseline parser (Orange)
    "OCR-Image": "#EA580C",
    "ocr-image": "#EA580C",
    "Image-Baseline": "#EA580C",
    "image-baseline": "#EA580C",
    # Text-Advanced (Red-Orange - clearly distinct from green baseline)
    "Text-Advanced": "#DC2626",
    "text-advanced": "#DC2626",
    "TwoStage-Text": "#DC2626",
    "twostage-text": "#DC2626",
    # Image-Advanced / TwoStage-Image (Cyan/Teal)
    "Image-Advanced": "#0891B2",
    "image-advanced": "#0891B2",
    "TwoStage-Image": "#0891B2",
    "twostage-image": "#0891B2",
}


def get_parser_color(parser_name: str) -> str:
    """Get the color for a parser by name."""
    return PARSER_COLORS.get(parser_name, COLORS["text_secondary"])


# =============================================================================
# Plotly Theme Configuration
# =============================================================================

PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent
    "plot_bgcolor": "rgba(0,0,0,0)",   # Transparent
    "font": {
        "color": COLORS["text_primary"],
        "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "size": 13,
    },
    "title": {
        "font": {
            "size": 16,
            "color": COLORS["text_primary"],
        },
        "x": 0,  # Left align
        "xanchor": "left",
    },
    "xaxis": {
        "gridcolor": COLORS["grid"],
        "linecolor": COLORS["border"],
        "tickfont": {"color": COLORS["text_secondary"], "size": 11},
        "title_font": {"color": COLORS["text_primary"], "size": 12},
        "zeroline": False,
    },
    "yaxis": {
        "gridcolor": COLORS["grid"],
        "linecolor": COLORS["border"],
        "tickfont": {"color": COLORS["text_secondary"], "size": 11},
        "title_font": {"color": COLORS["text_primary"], "size": 12},
        "zeroline": False,
    },
    "legend": {
        "bgcolor": "rgba(255,255,255,0.8)",
        "font": {"color": COLORS["text_primary"], "size": 11},
        "borderwidth": 0,
    },
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
    "hoverlabel": {
        "bgcolor": "white",
        "font_size": 12,
        "font_family": "-apple-system, BlinkMacSystemFont, sans-serif",
    },
}


# =============================================================================
# Streamlit Custom CSS - Minimal Clean Style
# =============================================================================

def apply_dark_theme() -> str:
    """Return custom CSS for clean light theme styling."""
    return """
    <style>
        /* Hide default Streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Main container - Pure white */
        .stApp {
            background-color: #FFFFFF;
        }

        /* Sidebar - Light gray */
        [data-testid="stSidebar"] {
            background-color: #FAFAFA;
            border-right: 1px solid #E8E8E8;
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: #262730;
        }

        /* Headers - Bold and dark */
        h1 {
            color: #262730 !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        h2 {
            color: #262730 !important;
            font-weight: 600 !important;
            font-size: 1.75rem !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }

        h3 {
            color: #262730 !important;
            font-weight: 600 !important;
            font-size: 1.25rem !important;
        }

        /* Paragraphs */
        p, .stMarkdown {
            color: #262730 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
        }

        /* Links */
        a {
            color: #1E90FF !important;
            text-decoration: none !important;
        }

        a:hover {
            text-decoration: underline !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background-color: #FAFAFA;
            padding: 1rem;
            border-radius: 0.5rem;
        }

        [data-testid="stMetricValue"] {
            color: #262730 !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricLabel"] {
            color: #555555 !important;
            font-size: 0.9rem !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.85rem !important;
        }

        /* Tabs - Minimal */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: transparent;
            border-bottom: 1px solid #E8E8E8;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #555555;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }

        .stTabs [aria-selected="true"] {
            color: #262730 !important;
            border-bottom: 2px solid #2E8B57 !important;
            background-color: transparent !important;
        }

        /* DataFrames - Clean table */
        .stDataFrame {
            border: none !important;
        }

        .stDataFrame table {
            border-collapse: collapse;
            width: 100%;
        }

        .stDataFrame th {
            background-color: #FAFAFA !important;
            color: #262730 !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 0.75rem !important;
            border-bottom: 2px solid #E8E8E8 !important;
        }

        .stDataFrame td {
            color: #262730 !important;
            padding: 0.75rem !important;
            border-bottom: 1px solid #F0F0F0 !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #FAFAFA;
            color: #262730 !important;
            font-weight: 500;
        }

        /* Selectbox and inputs */
        .stSelectbox > label,
        .stMultiSelect > label,
        .stRadio > label,
        .stSlider > label {
            color: #262730 !important;
            font-weight: 500 !important;
        }

        /* Button */
        .stButton > button {
            background-color: #2E8B57;
            color: white;
            border: none;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .stButton > button:hover {
            background-color: #247A4A;
        }

        /* Info boxes */
        .stAlert {
            background-color: #F0F7FF;
            border: 1px solid #B8D4FF;
            color: #262730;
        }

        /* Divider */
        hr {
            border: none;
            border-top: 1px solid #E8E8E8;
            margin: 2rem 0;
        }

        /* Caption text */
        .stCaption {
            color: #888888 !important;
            font-size: 0.85rem !important;
        }

        /* Code blocks */
        code {
            background-color: #F5F5F5;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.9rem;
        }

        /* Badge styles */
        .parser-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .parser-vlm {
            background-color: #D1FAE5;
            color: #065F46;
        }

        .parser-ocr {
            background-color: #DBEAFE;
            color: #1E40AF;
        }

        .parser-ocr-image {
            background-color: #FEF3C7;
            color: #92400E;
        }
    </style>
    """


def get_status_color(value: float, metric: str) -> str:
    """Get status color based on metric value and type."""
    lower_is_better = {"wer", "cer", "cs"}
    higher_is_better = {"bleu", "char_acc", "bc", "accuracy"}

    if metric.lower() in lower_is_better:
        if value <= 0.1:
            return COLORS["good"]
        elif value <= 0.3:
            return COLORS["warning"]
        else:
            return COLORS["bad"]
    elif metric.lower() in higher_is_better:
        if value >= 0.9:
            return COLORS["good"]
        elif value >= 0.7:
            return COLORS["warning"]
        else:
            return COLORS["bad"]
    else:
        return COLORS["text_secondary"]


def create_inline_bar_html(value: float, max_value: float = 1.0,
                           color: str = None, width_px: int = 100) -> str:
    """Create HTML for inline bar (like in the reference design)."""
    if color is None:
        color = COLORS["bar_positive"]

    percentage = min(100, (value / max_value) * 100)

    return f'''
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="min-width: 40px; text-align: right; font-weight: 500;">{value:.1f}</span>
        <div style="width: {width_px}px; height: 12px; background-color: #F0F0F0; border-radius: 2px;">
            <div style="width: {percentage}%; height: 100%; background-color: {color}; border-radius: 2px;"></div>
        </div>
    </div>
    '''
