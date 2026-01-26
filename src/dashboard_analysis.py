"""
VLM Document Parsing Quality Analysis Dashboard

CLI 테스트 결과를 시각화하여 Tech Report 작성을 지원하는 정적 대시보드

Features:
- JSON 결과 파일 로드 (results/parsing_results.json)
- @st.cache_data 캐싱 (1시간 TTL)
- 페이지네이션 (10개 테스트 초과 시)
- 차트 PNG 다운로드
- CSV 내보내기

Usage:
    streamlit run src/dashboard_analysis.py
"""

import sys
from pathlib import Path

_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any
import numpy as np

from dashboard.data_loader import (
    load_results_cached,
    get_parsing_data,
    get_chunking_data,
    get_chunking_data_for_parser,
    get_chunking_parsers,
    get_parser_names,
    get_parsing_summary_df,
    get_chunking_summary_df,
    get_chart_download_config,
    export_df_to_csv,
    paginate_data,
    get_sample_data,
)
from dashboard.charts import (
    STRATEGY_COLORS as CHART_STRATEGY_COLORS,
    create_parser_chunking_comparison,
    create_bc_document_flow,
    create_cs_mean_std_bar,
)

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="VLM Document Parsing Quality Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# 스타일 설정
# =============================================================================

st.markdown("""
<style>
    /* Sidebar 완전 숨김 */
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }

    /* 전체 배경 */
    .stApp { background-color: #FAFAFA; }

    /* 헤더 */
    h1, h2, h3 { color: #1a1a2e !important; font-weight: 600 !important; }

    /* 메트릭 카드 */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E5E5;
    }
    [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { color: #666666 !important; }

    /* 탭 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #E5E5E5; }
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a2e !important;
        border-bottom: 2px solid #4F46E5 !important;
    }

    /* 테이블 */
    .stDataFrame { border-radius: 8px; }

    /* 구분선 */
    hr { border-color: #E5E5E5; margin: 2rem 0; }

    /* 다운로드 버튼 */
    .download-btn {
        background-color: #F3F4F6;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 상수
# =============================================================================

VERSION = "v0.4.0"  # Added parser-specific chunking analysis (MoC-based)
PAGE_SIZE = 10  # 페이지네이션 크기

# 동적 색상 생성 (파서 추가 시 자동 확장)
DEFAULT_COLORS = ["#4F46E5", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]

def get_parser_colors(parsers: List[str]) -> Dict[str, str]:
    """파서별 색상 동적 생성"""
    colors = {}
    for i, parser in enumerate(parsers):
        colors[parser] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
    return colors

# 청킹 전략 색상
STRATEGY_COLORS = {
    "Fixed": "#6366F1",
    "Sentence": "#10B981",
    "Semantic": "#F59E0B",
    "Structuring": "#8B5CF6",
}


# =============================================================================
# 데이터 로드
# =============================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load data with caching"""
    data = load_results_cached()
    if "error" in data:
        # Fallback to sample data
        return get_sample_data(), True
    return data, False


# 데이터 로드
raw_data, is_sample = load_data()

# 파서 색상
PARSER_NAMES = get_parser_names(raw_data)
PARSER_COLORS = get_parser_colors(PARSER_NAMES)

# 변환된 데이터
PARSING_DATA = get_parsing_data(raw_data)
CHUNKING_DATA = get_chunking_data(raw_data)


# =============================================================================
# 차트 생성 함수
# =============================================================================

def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Hex 색상을 rgba로 변환"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_thin_bar_chart(data: Dict, metric: str, title: str,
                          lower_is_better: bool = False) -> go.Figure:
    """얇은 가로형 Bar Chart"""
    parsers = list(data["parsers"].keys())
    values = [data["parsers"][p].get(metric, 0) for p in parsers]
    colors = [PARSER_COLORS.get(p, "#888") for p in parsers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=parsers,
        x=values,
        orientation='h',
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" if metric != "latency" else f"{v:,}ms" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#666"),
    ))

    direction = "← Lower is better" if lower_is_better else "Higher is better →"
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#1a1a2e"), x=0),
        height=160,
        margin=dict(l=0, r=60, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#666"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False,
        annotations=[dict(
            text=direction, x=1, y=-0.15, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#999"), xanchor="right"
        )]
    )
    return fig


def create_radar_chart(all_data: Dict) -> go.Figure:
    """파서별 성능 Radar Chart"""
    metrics = ["WER", "CER", "BLEU", "Latency"]
    fig = go.Figure()

    for parser in PARSER_NAMES:
        values = []
        for metric_key in ["wer", "cer", "bleu", "latency"]:
            vals = [
                test["parsers"][parser].get(metric_key, 0)
                for test in all_data.values()
                if parser in test["parsers"]
            ]
            avg = np.mean(vals) if vals else 0

            # 정규화 (낮을수록 좋은 것은 반전)
            if metric_key in ["wer", "cer"]:
                normalized = 1 - min(avg, 1)
            elif metric_key == "latency":
                normalized = 1 - min(avg / 15000, 1)
            else:
                normalized = avg
            values.append(normalized)

        values.append(values[0])  # 닫기

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=parser,
            line=dict(color=PARSER_COLORS.get(parser, "#888"), width=3),
            fill='toself',
            fillcolor=hex_to_rgba(PARSER_COLORS.get(parser, "#888"), 0.1),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#E5E5E5"),
            angularaxis=dict(tickfont=dict(size=11, color="#666"), gridcolor="#E5E5E5"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10)),
        height=350,
        margin=dict(l=60, r=60, t=30, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_bc_cs_scatter(chunking_data: Dict) -> go.Figure:
    """BC vs CS Scatter Plot"""
    fig = go.Figure()

    for strategy, data in chunking_data.items():
        bc_values = [c.get("bc", 0) for c in data.get("chunks", [])]
        cs_values = [c.get("cs", 0) for c in data.get("chunks", [])]

        if not bc_values:
            continue

        fig.add_trace(go.Scatter(
            x=bc_values, y=cs_values, mode='markers', name=strategy,
            marker=dict(
                size=12,
                color=STRATEGY_COLORS.get(strategy, "#888"),
                line=dict(width=1, color="white"),
                opacity=0.8,
            ),
            hovertemplate=f"<b>{strategy}</b><br>BC: %{{x:.2f}}<br>CS: %{{y:.2f}}<extra></extra>",
        ))

    # Quadrant 영역
    fig.add_shape(type="rect", x0=0.5, x1=1, y0=0, y1=0.5,
                  fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=0.5, y0=0.5, y1=1,
                  fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)

    fig.add_hline(y=0.5, line_dash="dot", line_color="#ccc", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#ccc", line_width=1)

    annotations = [
        dict(x=0.75, y=0.25, text="이상적<br>(BC↑ CS↓)", showarrow=False,
             font=dict(size=9, color="#059669"), opacity=0.7),
        dict(x=0.25, y=0.75, text="Over-merge<br>(BC↓ CS↑)", showarrow=False,
             font=dict(size=9, color="#DC2626"), opacity=0.7),
        dict(x=0.75, y=0.75, text="Fragmentation<br>(BC↑ CS↑)", showarrow=False,
             font=dict(size=9, color="#D97706"), opacity=0.7),
        dict(x=0.25, y=0.25, text="Structural<br>Failure", showarrow=False,
             font=dict(size=9, color="#6B7280"), opacity=0.7),
    ]

    fig.update_layout(
        title=dict(text="BC–CS Distribution by Strategy", font=dict(size=14, color="#1a1a2e"), x=0),
        xaxis=dict(title="Boundary Clarity (BC) →", range=[0, 1], gridcolor="#E5E5E5", zeroline=False),
        yaxis=dict(title="Chunk Stickiness (CS) ↓", range=[0, 1], gridcolor="#E5E5E5", zeroline=False),
        height=450, margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        annotations=annotations,
    )
    return fig


def create_grouped_bar(all_data: Dict, metric: str, title: str, lower_is_better: bool = False) -> go.Figure:
    """전체 테스트 비교 Grouped Bar Chart"""
    test_ids = [d["id"] for d in all_data.values()]
    fig = go.Figure()

    for parser in PARSER_NAMES:
        color = PARSER_COLORS.get(parser, "#888")
        values = [test["parsers"].get(parser, {}).get(metric, 0) for test in all_data.values()]
        fig.add_trace(go.Bar(
            name=parser, x=test_ids, y=values,
            marker_color=color, marker_line_width=0,
            text=[f"{v:.2f}" if metric != "latency" else f"{v/1000:.1f}s" for v in values],
            textposition="outside", textfont=dict(size=9), width=0.25,
        ))

    direction = "↓ Lower is better" if lower_is_better else "↑ Higher is better"
    fig.update_layout(
        title=dict(text=f"{title} ({direction})", font=dict(size=13, color="#1a1a2e"), x=0),
        barmode="group", height=280,
        margin=dict(l=40, r=20, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#666"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#E5E5E5", gridwidth=0.5, zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=9)),
        bargap=0.3, bargroupgap=0.1,
    )
    return fig


# =============================================================================
# 메인 대시보드
# =============================================================================

# 헤더
st.title("📄 VLM Document Parsing Quality Analysis")
st.caption(f"CLI 테스트 결과 시각화 | Tech Report 작성 지원 | {VERSION}")

# 샘플 데이터 경고
if is_sample:
    st.warning("⚠️ 결과 파일을 찾을 수 없어 샘플 데이터를 표시합니다. CLI에서 테스트를 실행하세요.")

# 데이터 정보
data_info_cols = st.columns([1, 1, 1, 2])
with data_info_cols[0]:
    st.metric("Total Tests", len(PARSING_DATA))
with data_info_cols[1]:
    st.metric("Parsers", len(PARSER_NAMES))
with data_info_cols[2]:
    st.metric("Strategies", len(CHUNKING_DATA))
with data_info_cols[3]:
    created_at = raw_data.get("created_at", "N/A")
    st.caption(f"Data Version: {raw_data.get('version', 'N/A')} | Created: {created_at}")

st.markdown("---")

# 탭 구성
tab_parsing, tab_chunking, tab_result = st.tabs([
    "🔍 Parsing Test",
    "📦 Chunking Test",
    "📊 종합 분석"
])


# =============================================================================
# TAB 1: Parsing Test
# =============================================================================

with tab_parsing:
    st.markdown("## Parsing Test Results")

    # Metrics 정의
    with st.expander("📐 Metrics 정의", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**WER (Word Error Rate)** · :green[↓ 낮을수록 좋음]")
            st.markdown("Mecab 기반 단어 단위 오류율. 삽입/삭제/대체 오류 종합.")
            st.markdown("**CER (Character Error Rate)** · :green[↓ 낮을수록 좋음]")
            st.markdown("문자 단위 오류율. 누락/추가/변경 문자 추적.")
        with col2:
            st.markdown("**BLEU Score** · :orange[↑ 높을수록 좋음 (보조)]")
            st.markdown("핵심 키워드 포함 여부. n-gram 정밀도.")
            st.markdown("**Latency** · :green[↓ 낮을수록 좋음]")
            st.markdown("문서 1건 Parsing 처리 시간 (ms).")

    st.markdown("---")

    # Global Performance Summary
    st.markdown("### 📈 Global Performance Summary")

    col_table, col_radar = st.columns([1, 1])

    with col_table:
        # DataFrame 생성
        summary_df = get_parsing_summary_df(raw_data)
        display_df = summary_df[["Test", "Parser", "WER", "CER", "BLEU", "Latency (ms)"]].copy()
        display_df["WER"] = display_df["WER"].apply(lambda x: f"{x:.3f}")
        display_df["CER"] = display_df["CER"].apply(lambda x: f"{x:.3f}")
        display_df["BLEU"] = display_df["BLEU"].apply(lambda x: f"{x:.3f}")
        display_df["Latency (ms)"] = display_df["Latency (ms)"].apply(lambda x: f"{x:,.0f}ms")

        st.dataframe(display_df, use_container_width=True, hide_index=True, height=350)

        # CSV 다운로드
        csv_data = export_df_to_csv(summary_df)
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv_data,
            file_name="parsing_summary.csv",
            mime="text/csv",
        )

    with col_radar:
        radar_fig = create_radar_chart(PARSING_DATA)
        st.plotly_chart(
            radar_fig,
            width="stretch",
            config=get_chart_download_config("radar_chart")
        )

    # Metrics Comparison
    st.markdown("#### Metrics Comparison")

    row1 = st.columns(2)
    row2 = st.columns(2)

    with row1[0]:
        st.plotly_chart(
            create_grouped_bar(PARSING_DATA, "wer", "WER", lower_is_better=True),
            width="stretch",
            config=get_chart_download_config("wer_comparison")
        )
    with row1[1]:
        st.plotly_chart(
            create_grouped_bar(PARSING_DATA, "cer", "CER", lower_is_better=True),
            width="stretch",
            config=get_chart_download_config("cer_comparison")
        )
    with row2[0]:
        st.plotly_chart(
            create_grouped_bar(PARSING_DATA, "bleu", "BLEU", lower_is_better=False),
            width="stretch",
            config=get_chart_download_config("bleu_comparison")
        )
    with row2[1]:
        st.plotly_chart(
            create_grouped_bar(PARSING_DATA, "latency", "Latency", lower_is_better=True),
            width="stretch",
            config=get_chart_download_config("latency_comparison")
        )

    st.markdown("---")

    # Detailed Test Analysis with Pagination
    st.markdown("### 🔬 Detailed Test Analysis")

    # 페이지네이션 (10개 초과 시)
    test_items = list(PARSING_DATA.items())
    total_tests = len(test_items)

    if total_tests > PAGE_SIZE:
        # 페이지 선택
        col_page_info, col_page_nav = st.columns([2, 1])

        with col_page_info:
            st.caption(f"총 {total_tests}개 테스트 (페이지당 {PAGE_SIZE}개)")

        # 페이지 상태
        if "parsing_page" not in st.session_state:
            st.session_state.parsing_page = 1

        total_pages = (total_tests + PAGE_SIZE - 1) // PAGE_SIZE

        with col_page_nav:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.parsing_page,
                key="parsing_page_input"
            )
            st.session_state.parsing_page = page

        # 현재 페이지 데이터
        paginated_items, _, _, _ = paginate_data(test_items, page, PAGE_SIZE)
    else:
        paginated_items = test_items

    # 테스트별 상세 (Lazy Loading via Expander)
    for test_id, test_data in paginated_items:
        with st.expander(f"**{test_data['id']}: {test_data['name']}** ({test_data['doc_type']})", expanded=False):
            # 테이블
            detail_rows = []
            for parser, metrics in test_data["parsers"].items():
                detail_rows.append({
                    "Parser": parser,
                    "WER ↓": f"{metrics.get('wer', 0):.3f}",
                    "CER ↓": f"{metrics.get('cer', 0):.3f}",
                    "BLEU ↑": f"{metrics.get('bleu', 0):.3f}",
                    "Latency ↓": f"{metrics.get('latency', 0):,}ms",
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

            # Bar Charts
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "wer", "WER", lower_is_better=True),
                    width="stretch",
                    config=get_chart_download_config(f"{test_id}_wer")
                )
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "bleu", "BLEU", lower_is_better=False),
                    width="stretch",
                    config=get_chart_download_config(f"{test_id}_bleu")
                )
            with chart_cols[1]:
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "cer", "CER", lower_is_better=True),
                    width="stretch",
                    config=get_chart_download_config(f"{test_id}_cer")
                )
                st.plotly_chart(
                    create_thin_bar_chart(test_data, "latency", "Latency", lower_is_better=True),
                    width="stretch",
                    config=get_chart_download_config(f"{test_id}_latency")
                )


# =============================================================================
# TAB 2: Chunking Test
# =============================================================================

with tab_chunking:
    st.markdown("## Chunking Quality Analysis")
    st.markdown("> 파싱 결과가 Semantic Chunking 품질에 미치는 영향을 분석합니다.")

    # Metrics 정의
    with st.expander("📐 BC / CS Metrics 정의 (MoC Paper 기반)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Boundary Clarity (BC)** · :orange[↑ 높을수록 좋음]")
            st.markdown("청크 경계의 의미적 명확성. 문장 단위로 경계 타당성 평가.")
            st.markdown("- 1에 가까울수록 경계가 의미 단위와 일치")
            st.markdown("- MoC Paper: 'Document Flow' 그래프로 시각화")
        with col2:
            st.markdown("**Chunk Stickiness (CS)** · :green[↓ 낮을수록 좋음]")
            st.markdown("청크 내부 문장 간 평균 유사도 (Avg. Intra-chunk Similarity).")
            st.markdown("- 0에 가까울수록 청크 내부가 독립적")
            st.markdown("- Structuring 전략은 N/A (구조 기반 분할)")

    st.markdown("---")

    # =========================================================================
    # 파서 선택 및 비교 섹션
    # =========================================================================
    st.markdown("### 🔄 Parser별 Chunking 품질 비교")

    # 청킹 결과가 있는 파서 목록
    chunking_parsers = get_chunking_parsers(raw_data)

    if not chunking_parsers or chunking_parsers == ["_legacy"]:
        st.info("파서별 청킹 데이터가 없습니다. v1.1 형식의 결과 파일이 필요합니다.")
    else:
        # 파서 비교 차트 (전체 파서 × 전략별 BC/CS)
        st.markdown("#### 전략별 파서 성능 비교")

        # 전략 선택
        all_strategies = set()
        for parser in chunking_parsers:
            strategies = get_chunking_data_for_parser(raw_data, parser)
            for s in strategies:
                all_strategies.add(s.get("strategy", "unknown"))

        strategy_list = sorted(list(all_strategies))
        if strategy_list:
            selected_strategy = st.selectbox(
                "비교할 전략 선택",
                options=strategy_list,
                index=strategy_list.index("Semantic") if "Semantic" in strategy_list else 0,
                key="chunking_strategy_select"
            )

            # Parser Comparison Chart
            comparison_fig = create_parser_chunking_comparison(
                CHUNKING_DATA,
                selected_strategy,
                title=f"{selected_strategy} Chunking: Parser별 BC/CS 비교"
            )
            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
                config=get_chart_download_config(f"parser_comparison_{selected_strategy}")
            )

        st.markdown("---")

        # =========================================================================
        # 파서별 상세 분석 섹션
        # =========================================================================
        st.markdown("### 📊 Parser별 상세 분석")

        # 파서 선택 드롭다운
        selected_parser = st.selectbox(
            "분석할 파서 선택",
            options=chunking_parsers,
            index=0,
            key="chunking_parser_select"
        )

        # 선택된 파서의 전략 데이터
        parser_strategies = get_chunking_data_for_parser(raw_data, selected_parser)

        if not parser_strategies:
            st.warning(f"{selected_parser}의 청킹 데이터가 없습니다.")
        else:
            # KPI 카드
            st.markdown(f"#### {selected_parser} 전략별 요약")
            kpi_cols = st.columns(len(parser_strategies) + 1)

            with kpi_cols[0]:
                total_strategies = len(parser_strategies)
                st.metric("Strategies", total_strategies)

            for i, strategy_data in enumerate(parser_strategies):
                strategy_name = strategy_data.get("strategy", "unknown")
                mean_bc = strategy_data.get("mean_bc", 0)
                mean_cs = strategy_data.get("mean_cs")

                with kpi_cols[i + 1]:
                    # CS가 N/A인 경우 (Structuring)
                    cs_display = f"{mean_cs:.2f}" if mean_cs is not None else "N/A"
                    st.metric(
                        strategy_name,
                        f"BC: {mean_bc:.2f}",
                        f"CS: {cs_display}",
                        delta_color="off"
                    )

            # 2열 레이아웃: BC Document Flow + CS Mean±Std
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("##### BC Document Flow")
                st.caption("문장별 BC 값과 청크 경계 위치 (MoC Paper Fig.2 스타일)")

                bc_flow_fig = create_bc_document_flow(
                    parser_strategies,
                    title=f"{selected_parser}: Boundary Clarity Flow"
                )
                st.plotly_chart(
                    bc_flow_fig,
                    use_container_width=True,
                    config=get_chart_download_config(f"bc_flow_{selected_parser}")
                )

            with chart_col2:
                st.markdown("##### CS Mean ± Std")
                st.caption("전략별 Chunk Stickiness (낮을수록 좋음, Structuring은 N/A)")

                cs_bar_fig = create_cs_mean_std_bar(
                    parser_strategies,
                    title=f"{selected_parser}: Avg. Intra-chunk Similarity"
                )
                st.plotly_chart(
                    cs_bar_fig,
                    use_container_width=True,
                    config=get_chart_download_config(f"cs_bar_{selected_parser}")
                )

            st.markdown("---")

            # 전략별 상세 데이터
            st.markdown("##### 전략별 상세 데이터")

            for strategy_data in parser_strategies:
                strategy_name = strategy_data.get("strategy", "unknown")
                mean_bc = strategy_data.get("mean_bc", 0)
                mean_cs = strategy_data.get("mean_cs")
                std_bc = strategy_data.get("std_bc")
                std_cs = strategy_data.get("std_cs")

                cs_display = f"{mean_cs:.3f}" if mean_cs is not None else "N/A"
                std_bc_display = f"±{std_bc:.3f}" if std_bc is not None else ""
                std_cs_display = f"±{std_cs:.3f}" if std_cs is not None else ""

                with st.expander(
                    f"**{strategy_name}** | BC: {mean_bc:.3f}{std_bc_display} | CS: {cs_display}{std_cs_display}"
                ):
                    # bc_by_sentence 데이터 표시
                    bc_by_sentence = strategy_data.get("bc_by_sentence", [])
                    if bc_by_sentence:
                        bc_df = pd.DataFrame(bc_by_sentence)
                        if "is_boundary" in bc_df.columns:
                            bc_df["is_boundary"] = bc_df["is_boundary"].apply(
                                lambda x: "✓ 경계" if x else ""
                            )
                        st.dataframe(bc_df, use_container_width=True, hide_index=True, height=300)

                        # CSV 다운로드
                        csv_bc = export_df_to_csv(bc_df)
                        st.download_button(
                            label=f"📥 {strategy_name} BC Data CSV",
                            data=csv_bc,
                            file_name=f"bc_{selected_parser}_{strategy_name.lower()}.csv",
                            mime="text/csv",
                            key=f"download_bc_{selected_parser}_{strategy_name}"
                        )
                    else:
                        st.info("bc_by_sentence 데이터 없음 (CLI 연동 필요)")

    st.markdown("---")

    # Quadrant Guide (유지)
    st.markdown("### 🔍 BC / CS 해석 가이드")
    st.markdown("""
    | 지표 | 의미 | 이상적 값 | 해석 |
    |------|------|----------|------|
    | **BC ↑** | Boundary Clarity | > 0.8 | 청크 경계가 의미 단위와 일치 |
    | **CS ↓** | Chunk Stickiness | < 0.3 | 청크 내부 문장들이 독립적 |
    | **std_bc ↓** | BC 표준편차 | < 0.1 | 일관된 경계 품질 |
    | **std_cs ↓** | CS 표준편차 | < 0.1 | 일관된 응집도 |

    > 💡 **Structuring** 전략은 마크다운 구조(헤딩, 리스트 등)를 기반으로 분할하므로 CS 계산이 적용되지 않습니다.
    """)


# =============================================================================
# TAB 3: 종합 분석
# =============================================================================

with tab_result:
    st.markdown("## 📊 종합 분석 결과")
    st.markdown("> Parsing과 Chunking 결과를 종합하여 파이프라인 품질을 진단합니다.")

    st.markdown("---")

    st.markdown("### 🎯 핵심 발견사항")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Parsing 관점

        1. **VLM이 전반적으로 우수**
           - 대부분의 테스트에서 최저 WER 달성
           - 특히 이미지 기반 문서에서 압도적

        2. **Trade-off 존재**
           - 정확도 ↔ 처리 시간
           - 실시간 서비스에는 pdfplumber 고려

        3. **문서 유형별 차이 큼**
           - 스캔 이미지: VLM 필수
           - 디지털 PDF: pdfplumber도 충분
        """)

    with col2:
        st.markdown("""
        #### Chunking 관점

        1. **Semantic Chunking 권장**
           - BC가 가장 높은 경계 명확도
           - CS가 낮은 내부 의존성

        2. **Fixed Chunking 주의**
           - 의미 경계 무시로 BC 낮음
           - RAG 성능 저하 우려

        3. **최적 파라미터**
           - Chunk Size: 400-600
           - Overlap: 50-100
        """)

    st.markdown("---")

    st.markdown("### 🚀 다음 단계")
    st.markdown("""
    | 우선순위 | 작업 | 목적 |
    |---------|------|------|
    | 1 | Golden Dataset 구축 | 평가 신뢰도 향상 |
    | 2 | VLM SFT 학습 | 구조화 성능 개선 |
    | 3 | Semantic Chunking 적용 | RAG 품질 향상 |
    | 4 | 추가 문서 유형 테스트 | 일반화 검증 |
    """)

    st.markdown("---")
    st.caption(f"VLM Document Parsing Quality Analysis | {VERSION}")
