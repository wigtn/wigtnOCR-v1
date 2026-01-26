"""
VLM Document Parsing & Chunking Research Dashboard

"얼마나 잘했는가"가 아닌 "어디서, 왜 구조가 깨지는가"를 진단하는 연구용 도구

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import numpy as np

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="Parsing & Chunking Research",
    page_icon="🔬",
    layout="wide",
)

# =============================================================================
# 스타일 설정
# =============================================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp { background-color: #FAFAFA; }

    /* 헤더 */
    h1, h2, h3 { color: #1a1a2e !important; font-weight: 600 !important; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E5E5; }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem !important; color: #1a1a2e !important; }

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

    /* 캡션 */
    .metric-caption { font-size: 0.75rem; color: #888888; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 상수 및 샘플 데이터
# =============================================================================

VERSION = "v0.1.0"

# 파서 색상 (절제된 팔레트)
PARSER_COLORS = {
    "VLM (Qwen3-VL)": "#4F46E5",      # Indigo
    "pdfplumber": "#059669",           # Emerald
    "Docling (RapidOCR)": "#D97706",  # Amber
}

# 청킹 전략 색상
STRATEGY_COLORS = {
    "Fixed": "#6366F1",
    "Sentence": "#10B981",
    "Semantic": "#F59E0B",
    "MoC-style": "#EF4444",
}

# 샘플 Parsing 테스트 데이터
PARSING_TEST_DATA = {
    "test_1": {
        "id": "Test 1",
        "name": "학술 논문 (Chain-of-Thought)",
        "doc_type": "PDF - 다중 페이지, 수식/표 포함",
        "parsers": {
            "VLM (Qwen3-VL)": {"wer": 0.124, "cer": 0.089, "bleu": 0.847, "latency": 8520},
            "pdfplumber": {"wer": 0.182, "cer": 0.145, "bleu": 0.723, "latency": 1240},
            "Docling (RapidOCR)": {"wer": 0.251, "cer": 0.198, "bleu": 0.652, "latency": 3820},
        },
    },
    "test_2": {
        "id": "Test 2",
        "name": "영수증 (스캔 이미지)",
        "doc_type": "이미지 - 단일 페이지, 짧은 텍스트",
        "parsers": {
            "VLM (Qwen3-VL)": {"wer": 0.082, "cer": 0.054, "bleu": 0.912, "latency": 2140},
            "pdfplumber": {"wer": 0.456, "cer": 0.387, "bleu": 0.345, "latency": 520},
            "Docling (RapidOCR)": {"wer": 0.153, "cer": 0.112, "bleu": 0.781, "latency": 1580},
        },
    },
    "test_3": {
        "id": "Test 3",
        "name": "기술 문서 (코드 블록 포함)",
        "doc_type": "PDF - 다중 페이지, 코드/다이어그램",
        "parsers": {
            "VLM (Qwen3-VL)": {"wer": 0.156, "cer": 0.118, "bleu": 0.819, "latency": 12340},
            "pdfplumber": {"wer": 0.223, "cer": 0.178, "bleu": 0.682, "latency": 1820},
            "Docling (RapidOCR)": {"wer": 0.284, "cer": 0.231, "bleu": 0.618, "latency": 5240},
        },
    },
}

# 샘플 Chunking 테스트 데이터
CHUNKING_TEST_DATA = {
    "Fixed": {
        "chunks": [
            {"id": 1, "bc": 0.72, "cs": 0.85, "length": 512, "tokens": 128},
            {"id": 2, "bc": 0.68, "cs": 0.91, "length": 512, "tokens": 134},
            {"id": 3, "bc": 0.45, "cs": 0.78, "length": 512, "tokens": 121},
            {"id": 4, "bc": 0.81, "cs": 0.65, "length": 512, "tokens": 142},
            {"id": 5, "bc": 0.38, "cs": 0.92, "length": 512, "tokens": 115},
        ],
        "mean_bc": 0.608, "mean_cs": 0.822,
    },
    "Sentence": {
        "chunks": [
            {"id": 1, "bc": 0.85, "cs": 0.72, "length": 324, "tokens": 82},
            {"id": 2, "bc": 0.79, "cs": 0.68, "length": 456, "tokens": 118},
            {"id": 3, "bc": 0.91, "cs": 0.58, "length": 287, "tokens": 71},
            {"id": 4, "bc": 0.76, "cs": 0.74, "length": 512, "tokens": 135},
            {"id": 5, "bc": 0.88, "cs": 0.61, "length": 398, "tokens": 98},
        ],
        "mean_bc": 0.838, "mean_cs": 0.666,
    },
    "Semantic": {
        "chunks": [
            {"id": 1, "bc": 0.92, "cs": 0.45, "length": 623, "tokens": 156},
            {"id": 2, "bc": 0.87, "cs": 0.52, "length": 412, "tokens": 108},
            {"id": 3, "bc": 0.78, "cs": 0.61, "length": 534, "tokens": 142},
            {"id": 4, "bc": 0.94, "cs": 0.38, "length": 378, "tokens": 95},
        ],
        "mean_bc": 0.878, "mean_cs": 0.490,
    },
}


# =============================================================================
# 차트 생성 함수
# =============================================================================

def create_thin_bar_chart(data: Dict, metric: str, title: str,
                          lower_is_better: bool = False) -> go.Figure:
    """얇고 단정한 가로형 Bar Chart"""

    parsers = list(data["parsers"].keys())
    values = [data["parsers"][p][metric] for p in parsers]
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
        title=dict(text=f"{title}", font=dict(size=13, color="#1a1a2e"), x=0),
        height=160,
        margin=dict(l=0, r=60, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#666"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False,
        annotations=[
            dict(
                text=direction, x=1, y=-0.15, xref="paper", yref="paper",
                showarrow=False, font=dict(size=9, color="#999"),
                xanchor="right"
            )
        ]
    )

    return fig


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Hex 색상을 rgba 문자열로 변환"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_radar_chart(all_data: Dict) -> go.Figure:
    """파서별 전체 성능 Radar Chart"""

    metrics = ["WER", "CER", "BLEU", "Latency"]

    fig = go.Figure()

    # 파서별 평균 계산 및 정규화
    for parser in list(PARSER_COLORS.keys()):
        values = []
        for metric_key, metric_name in [("wer", "WER"), ("cer", "CER"),
                                         ("bleu", "BLEU"), ("latency", "Latency")]:
            avg = np.mean([
                test["parsers"][parser][metric_key]
                for test in all_data.values()
            ])

            # 정규화 (0-1 스케일로, 낮을수록 좋은 것은 반전)
            if metric_key in ["wer", "cer"]:
                normalized = 1 - min(avg, 1)  # 낮을수록 좋음 → 반전
            elif metric_key == "latency":
                normalized = 1 - min(avg / 15000, 1)  # 15초 기준 정규화
            else:
                normalized = avg  # BLEU는 그대로

            values.append(normalized)

        values.append(values[0])  # 닫기

        # 90% 투명 (alpha=0.1), 선만 진하게 (width=3)
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=parser,
            line=dict(color=PARSER_COLORS[parser], width=3),
            fill='toself',
            fillcolor=hex_to_rgba(PARSER_COLORS[parser], 0.1),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#E5E5E5"),
            angularaxis=dict(tickfont=dict(size=11, color="#666"), gridcolor="#E5E5E5"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                   font=dict(size=10)),
        height=350,
        margin=dict(l=60, r=60, t=30, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_bc_cs_scatter(chunking_data: Dict) -> go.Figure:
    """BC vs CS Scatter Plot (핵심 시각화)"""

    fig = go.Figure()

    for strategy, data in chunking_data.items():
        bc_values = [c["bc"] for c in data["chunks"]]
        cs_values = [c["cs"] for c in data["chunks"]]

        fig.add_trace(go.Scatter(
            x=bc_values,
            y=cs_values,
            mode='markers',
            name=strategy,
            marker=dict(
                size=12,
                color=STRATEGY_COLORS.get(strategy, "#888"),
                line=dict(width=1, color="white"),
                opacity=0.8,
            ),
            hovertemplate=f"<b>{strategy}</b><br>BC: %{{x:.2f}}<br>CS: %{{y:.2f}}<extra></extra>",
        ))

    # Quadrant 영역 표시 (배경)
    fig.add_shape(type="rect", x0=0.5, x1=1, y0=0, y1=0.5,
                  fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)  # 이상적 영역
    fig.add_shape(type="rect", x0=0, x1=0.5, y0=0.5, y1=1,
                  fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)  # 문제 영역

    # 가이드 라인
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ccc", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#ccc", line_width=1)

    # Quadrant 라벨
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
        xaxis=dict(title="Boundary Clarity (BC) →", range=[0, 1],
                   gridcolor="#E5E5E5", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Chunk Stickiness (CS) ↓", range=[0, 1],
                   gridcolor="#E5E5E5", zeroline=False, tickfont=dict(size=10)),
        height=450,
        margin=dict(l=60, r=30, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                   font=dict(size=10)),
        annotations=annotations,
    )

    return fig


def create_overview_grouped_bar(all_data: Dict, metric: str, title: str,
                                 lower_is_better: bool = False) -> go.Figure:
    """전체 테스트 비교 Grouped Bar Chart"""

    test_ids = [d["id"] for d in all_data.values()]

    fig = go.Figure()

    for parser, color in PARSER_COLORS.items():
        values = [test["parsers"][parser][metric] for test in all_data.values()]

        fig.add_trace(go.Bar(
            name=parser,
            x=test_ids,
            y=values,
            marker_color=color,
            marker_line_width=0,
            text=[f"{v:.2f}" if metric != "latency" else f"{v/1000:.1f}s" for v in values],
            textposition="outside",
            textfont=dict(size=9),
            width=0.25,
        ))

    direction = "↓ Lower is better" if lower_is_better else "↑ Higher is better"

    fig.update_layout(
        title=dict(text=f"{title} ({direction})", font=dict(size=13, color="#1a1a2e"), x=0),
        barmode="group",
        height=280,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#666"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#E5E5E5", gridwidth=0.5, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                   font=dict(size=9)),
        bargap=0.3,
        bargroupgap=0.1,
    )

    return fig


# =============================================================================
# 메인 앱
# =============================================================================

# 상위 탭 구조
tab_parsing, tab_chunking, tab_result = st.tabs([
    "🔍 Parsing Test",
    "📦 Chunking Test",
    "📊 Result (종합 분석)"
])


# =============================================================================
# TAB 1: Parsing Test
# =============================================================================

with tab_parsing:

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("# Data Parsing Research")
        st.markdown("---")

        test_mode = st.selectbox(
            "Test Mode",
            options=["Sample Data", "File Upload"],
            help="Sample Data: 사전 정의된 테스트 3건 사용"
        )

        if test_mode == "File Upload":
            st.markdown("#### Input Document")
            uploaded_file = st.file_uploader(
                "문서 업로드",
                type=["pdf", "png", "jpg"],
                help="PDF 또는 이미지 파일"
            )

            st.markdown("#### Ground Truth")
            gt_text = st.text_area(
                "정답 텍스트 입력",
                height=150,
                placeholder="Ground Truth 텍스트를 입력하세요..."
            )

            if st.button("테스트 실행", type="primary", use_container_width=True):
                st.info("🚧 File Upload 모드는 Golden Dataset 구축 후 활성화 예정")

        st.markdown("---")
        st.caption(f"Version: {VERSION} (Parsing Research)")

    # --- Main Content ---
    st.markdown("## Parsing Test Results")

    # A. Metrics Overview
    st.markdown("### 📐 Metrics Overview")

    st.markdown("""
    <div style="background-color: #F5F5F5; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
            <div>
                <p style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
                    WER (Word Error Rate) <span style="color: #059669; font-size: 0.9rem;">↓ 낮을수록 좋음</span>
                </p>
                <ul style="color: #555; margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                    <li>Mecab Tokenizer 기반 단어 단위 오류율</li>
                    <li>삽입/삭제/대체 오류를 종합 측정</li>
                </ul>

                <p style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-top: 1rem; margin-bottom: 0.5rem;">
                    CER (Character Error Rate) <span style="color: #059669; font-size: 0.9rem;">↓ 낮을수록 좋음</span>
                </p>
                <ul style="color: #555; margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                    <li>문자 단위 오류율</li>
                    <li>Error Injection: 어떤 문자가 누락/추가/변경되었는지 추적</li>
                </ul>
            </div>
            <div>
                <p style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
                    BLEU Score <span style="color: #D97706; font-size: 0.9rem;">↑ 높을수록 좋음 (보조)</span>
                </p>
                <ul style="color: #555; margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                    <li>문장 유사도가 아닌 핵심 키워드 포함 여부 확인용</li>
                    <li>n-gram 기반 정밀도 측정</li>
                </ul>

                <p style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-top: 1rem; margin-bottom: 0.5rem;">
                    Latency <span style="color: #059669; font-size: 0.9rem;">↓ 낮을수록 좋음</span>
                </p>
                <ul style="color: #555; margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                    <li>문서 1건 기준 Parsing 처리 시간</li>
                    <li>단위: milliseconds (ms)</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # B. 테스트 성능 개요
    st.markdown("### 📈 Global Performance Summary")

    col_table, col_radar = st.columns([1, 1])

    with col_table:
        # 전체 결과 테이블
        table_rows = []
        for test_id, test_data in PARSING_TEST_DATA.items():
            for parser, metrics in test_data["parsers"].items():
                table_rows.append({
                    "Test": test_data["id"],
                    "Parser": parser,
                    "WER": f"{metrics['wer']:.3f}",
                    "CER": f"{metrics['cer']:.3f}",
                    "BLEU": f"{metrics['bleu']:.3f}",
                    "Latency": f"{metrics['latency']:,}ms",
                })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=350)

    with col_radar:
        st.plotly_chart(create_radar_chart(PARSING_TEST_DATA), use_container_width=True)

    # 하단 4개 차트 (Z 레이아웃)
    st.markdown("#### Metrics Comparison")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.plotly_chart(
            create_overview_grouped_bar(PARSING_TEST_DATA, "wer", "WER", lower_is_better=True),
            use_container_width=True
        )

    with row1_col2:
        st.plotly_chart(
            create_overview_grouped_bar(PARSING_TEST_DATA, "cer", "CER", lower_is_better=True),
            use_container_width=True
        )

    with row2_col1:
        st.plotly_chart(
            create_overview_grouped_bar(PARSING_TEST_DATA, "bleu", "BLEU", lower_is_better=False),
            use_container_width=True
        )

    with row2_col2:
        st.plotly_chart(
            create_overview_grouped_bar(PARSING_TEST_DATA, "latency", "Latency", lower_is_better=True),
            use_container_width=True
        )

    st.markdown("---")

    # C. 개별 테스트 상세 결과
    st.markdown("### 🔬 Detailed Test Analysis")

    for test_id, test_data in PARSING_TEST_DATA.items():

        st.markdown(f"#### {test_data['id']}: {test_data['name']}")
        st.caption(f"📄 {test_data['doc_type']}")

        # 메트릭 카드 (작게)
        m1, m2, m3, m4 = st.columns(4)

        # VLM 기준으로 표시 (대표 파서)
        vlm_metrics = test_data["parsers"]["VLM (Qwen3-VL)"]

        m1.metric("WER", f"{vlm_metrics['wer']:.3f}", help="VLM 기준")
        m2.metric("CER", f"{vlm_metrics['cer']:.3f}", help="VLM 기준")
        m3.metric("BLEU", f"{vlm_metrics['bleu']:.3f}", help="VLM 기준")
        m4.metric("Latency", f"{vlm_metrics['latency']:,}ms", help="VLM 기준")

        # 가로형 Bar Chart
        chart_cols = st.columns(4)

        with chart_cols[0]:
            st.plotly_chart(
                create_thin_bar_chart(test_data, "wer", "WER", lower_is_better=True),
                use_container_width=True
            )

        with chart_cols[1]:
            st.plotly_chart(
                create_thin_bar_chart(test_data, "cer", "CER", lower_is_better=True),
                use_container_width=True
            )

        with chart_cols[2]:
            st.plotly_chart(
                create_thin_bar_chart(test_data, "bleu", "BLEU", lower_is_better=False),
                use_container_width=True
            )

        with chart_cols[3]:
            st.plotly_chart(
                create_thin_bar_chart(test_data, "latency", "Latency", lower_is_better=True),
                use_container_width=True
            )

        st.markdown("---")

    # D. 결론
    st.markdown("### 📝 Parsing Analysis Conclusion")

    st.markdown("""
    #### 파서별 특징 요약

    | Parser | 강점 | 약점 | 권장 사용처 |
    |--------|------|------|------------|
    | **VLM (Qwen3-VL)** | 구조화 정확도 높음, 이미지 인식 우수 | 처리 시간 김 | 복잡한 문서, 스캔 이미지 |
    | **pdfplumber** | 빠른 처리, 디지털 PDF에 안정적 | 이미지 기반 문서 취약 | 텍스트 기반 디지털 PDF |
    | **Docling (RapidOCR)** | 범용성, 중간 성능 | 특출난 강점 없음 | 일반적인 OCR 필요 시 |

    #### 에러 유형 경향
    - **VLM**: 수식/특수문자에서 간헐적 오류
    - **pdfplumber**: 레이아웃 복잡도에 민감
    - **Docling**: 저해상도 이미지에서 인식률 저하

    #### Golden Dataset 구축 시 기대 효과
    - 문서 유형별 최적 파서 자동 선택 기준 수립
    - SFT를 통한 VLM 성능 개선 목표치 설정
    """)


# =============================================================================
# TAB 2: Chunking Test
# =============================================================================

with tab_chunking:

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("# Chunking Strategy Research")
        st.markdown("---")

        selected_strategies = st.multiselect(
            "Chunking Strategies",
            options=list(CHUNKING_TEST_DATA.keys()),
            default=list(CHUNKING_TEST_DATA.keys()),
        )

        st.markdown("#### Parameters")
        chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64)
        chunk_overlap = st.slider("Overlap", 0, 128, 50, 10)

        st.markdown("---")
        st.caption(f"Version: {VERSION} (Chunking Research)")

    # --- Main Content ---
    st.markdown("## Chunking Quality Analysis")
    st.markdown("""
    > Chunking Test는 성능 비교가 아니라 **"구조 진단"**이 목적입니다.
    > 청크 경계의 타당성(BC)과 내부 응집도(CS)를 분석합니다.
    """)

    st.markdown("---")

    # A. Chunking Quality Metrics
    st.markdown("### 📐 Chunking Quality Metrics")

    with st.expander("BC / CS Metrics 정의", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Boundary Clarity (BC)**
            - 청크 간 **의미적 독립성** 측정
            - 경계 위치가 의미적으로 타당한지 평가
            - PPL(q|d) / PPL(q) 기반 계산
            - *높을수록 좋음 (0~1)*
            """)

        with col2:
            st.markdown("""
            **Chunk Stickiness (CS)**
            - 청크 내부 **의미 응집도** 측정
            - 내부 요소들이 얼마나 긴밀히 연결되어 있는지
            - Structural Entropy 기반 계산
            - *낮을수록 좋음 (이상적: 0에 가까움)*
            """)

    st.markdown("---")

    # B. BC / CS Distribution Overview
    st.markdown("### 📊 BC–CS Distribution Overview")

    # KPI 카드
    kpi_cols = st.columns(len(selected_strategies) + 1)

    with kpi_cols[0]:
        total_chunks = sum(len(CHUNKING_TEST_DATA[s]["chunks"]) for s in selected_strategies)
        st.metric("Total Chunks", total_chunks)

    for i, strategy in enumerate(selected_strategies):
        data = CHUNKING_TEST_DATA[strategy]
        with kpi_cols[i + 1]:
            st.metric(
                strategy,
                f"BC: {data['mean_bc']:.2f}",
                f"CS: {data['mean_cs']:.2f}",
                delta_color="off"
            )

    # BC-CS Scatter Plot (핵심)
    filtered_data = {s: CHUNKING_TEST_DATA[s] for s in selected_strategies}
    st.plotly_chart(create_bc_cs_scatter(filtered_data), use_container_width=True)

    st.markdown("---")

    # C. Chunking Failure Pattern Analysis
    st.markdown("### 🔍 Failure Pattern Analysis")

    st.markdown("""
    **Quadrant 해석 가이드**

    | 영역 | BC | CS | 의미 | 조치 |
    |------|----|----|------|------|
    | 🟢 이상적 | High | Low | 경계 명확, 내부 독립적 | 유지 |
    | 🟡 Fragmentation | High | High | 과도한 분할 | Chunk 크기 증가 |
    | 🔴 Over-merge | Low | High | 과도한 병합 | Chunk 크기 감소 |
    | ⚫ Structural Failure | Low | Low | 구조 자체 문제 | 전략 재검토 |
    """)

    # 전략별 상세 분석
    for strategy in selected_strategies:
        with st.expander(f"📦 {strategy} Strategy Details"):
            chunks = CHUNKING_TEST_DATA[strategy]["chunks"]

            chunk_df = pd.DataFrame(chunks)
            chunk_df["Quadrant"] = chunk_df.apply(
                lambda r: "이상적" if r["bc"] >= 0.5 and r["cs"] < 0.5
                else "Fragmentation" if r["bc"] >= 0.5 and r["cs"] >= 0.5
                else "Over-merge" if r["bc"] < 0.5 and r["cs"] >= 0.5
                else "Structural Failure",
                axis=1
            )

            st.dataframe(chunk_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # D. Chunking Summary
    st.markdown("### 📝 Chunking Strategy Summary")

    st.markdown("""
    #### 전략별 특징

    | Strategy | BC (mean) | CS (mean) | 특징 |
    |----------|-----------|-----------|------|
    | **Fixed** | 0.61 | 0.82 | 일정한 크기, 의미 경계 무시 |
    | **Sentence** | 0.84 | 0.67 | 문장 단위, 적절한 균형 |
    | **Semantic** | 0.88 | 0.49 | 의미 단위, 가장 높은 BC |

    #### Retrieval 영향 해석
    - **BC가 높은 전략**: 검색 시 관련 없는 컨텍스트 혼입 감소
    - **CS가 낮은 전략**: 청크 내 정보 중복 감소, 효율적 인덱싱
    - **권장**: Semantic 또는 Sentence 기반 전략
    """)


# =============================================================================
# TAB 3: Result (종합 분석)
# =============================================================================

with tab_result:

    st.markdown("## 📊 종합 분석 결과")

    st.markdown("""
    > 이 섹션은 Parsing과 Chunking 결과를 종합하여 전체 파이프라인의 품질을 진단합니다.
    """)

    st.markdown("---")

    st.markdown("### 🎯 핵심 발견사항")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Parsing 관점

        1. **VLM이 전반적으로 우수**
           - 3개 테스트 중 3개에서 최저 WER
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
           - BC 0.88로 가장 높은 경계 명확도
           - CS 0.49로 낮은 내부 의존성

        2. **Fixed Chunking 주의**
           - 의미 경계 무시로 BC 낮음
           - RAG 성능 저하 우려

        3. **최적 파라미터**
           - Chunk Size: 400-600
           - Overlap: 50-100
        """)

    st.markdown("---")

    st.markdown("### 🚀 다음 단계 권장사항")

    st.markdown("""
    | 우선순위 | 작업 | 목적 |
    |---------|------|------|
    | 1 | Golden Dataset 구축 | 평가 신뢰도 향상 |
    | 2 | VLM SFT 학습 | 구조화 성능 개선 |
    | 3 | Semantic Chunking 파이프라인 적용 | RAG 품질 향상 |
    | 4 | 추가 문서 유형 테스트 | 일반화 성능 검증 |
    """)

    st.markdown("---")
    st.caption(f"VLM Document Parsing Research Dashboard | {VERSION}")
