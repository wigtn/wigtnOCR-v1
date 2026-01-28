"""
Chunking CLI 단위 테스트

MoC-based chunking quality metrics (BC, CS) 테스트
"""

import pytest

from src.chunking.chunker import (
    ChunkerConfig,
    Chunk,
)
from src.chunking.metrics import (
    BCScore,
    CSScore,
    ChunkingMetrics,
    MockEmbeddingClient,
    calculate_bc,
    calculate_cs,
    calculate_edge_weight_semantic,
    calculate_structural_entropy,
    build_chunk_graph,
    evaluate_chunking,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_text():
    """테스트용 샘플 텍스트"""
    return """
# 제목

첫 번째 섹션입니다. 이 섹션에서는 프로젝트 개요를 설명합니다.
다양한 문서 파싱 방법을 비교 분석합니다.

## 배경

기존 OCR 방식은 구조 정보를 보존하지 못합니다.
VLM 방식은 마크다운 형식으로 구조를 유지합니다.

## 방법론

1. 파싱 단계
2. 청킹 단계
3. 평가 단계

각 단계에서 다양한 메트릭을 측정합니다.
"""


@pytest.fixture
def sample_chunks():
    """테스트용 청크 리스트"""
    return [
        "첫 번째 청크입니다. Python 프로그래밍에 대해 설명합니다.",
        "두 번째 청크입니다. Java 프로그래밍에 대해 설명합니다.",
        "세 번째 청크입니다. 데이터베이스 설계에 대해 설명합니다.",
        "네 번째 청크입니다. 웹 개발 기초에 대해 설명합니다.",
    ]


@pytest.fixture
def mock_client():
    """Mock Embedding 클라이언트"""
    return MockEmbeddingClient()


# =============================================================================
# Chunker Config Tests
# =============================================================================

class TestChunkerConfig:
    """청킹 설정 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = ChunkerConfig()

        assert config.breakpoint_threshold_type == "percentile"
        assert config.breakpoint_threshold_amount == 95.0
        assert config.min_chunk_size is None

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = ChunkerConfig(
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=3.0,
            min_chunk_size=100,
        )

        assert config.breakpoint_threshold_type == "standard_deviation"
        assert config.breakpoint_threshold_amount == 3.0
        assert config.min_chunk_size == 100

    def test_config_to_dict(self):
        """설정 to_dict 테스트"""
        config = ChunkerConfig(
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=90.0,
        )

        config_dict = config.to_dict()

        assert "breakpoint_threshold_type" in config_dict
        assert "breakpoint_threshold_amount" in config_dict
        assert "min_chunk_size" in config_dict


# =============================================================================
# Chunk Tests
# =============================================================================

class TestChunk:
    """Chunk 클래스 테스트"""

    def test_chunk_creation(self):
        """청크 생성 테스트"""
        chunk = Chunk(
            id="doc_chunk_0",
            content="테스트 내용입니다.",
            start_index=0,
            end_index=10,
            metadata={"document_id": "doc"}
        )

        assert chunk.id == "doc_chunk_0"
        assert chunk.content == "테스트 내용입니다."
        assert chunk.start_index == 0
        assert chunk.end_index == 10

    def test_chunk_length(self):
        """청크 길이 속성 테스트"""
        chunk = Chunk(
            id="test",
            content="Hello World",
            start_index=0,
            end_index=11,
        )

        assert chunk.length == 11

    def test_chunk_to_dict(self):
        """Chunk.to_dict() 테스트"""
        chunk = Chunk(
            id="doc_chunk_0",
            content="테스트 내용",
            start_index=0,
            end_index=5,
            metadata={"key": "value"}
        )

        chunk_dict = chunk.to_dict()

        assert "id" in chunk_dict
        assert "content" in chunk_dict
        assert "start_index" in chunk_dict
        assert "end_index" in chunk_dict
        assert "length" in chunk_dict
        assert "metadata" in chunk_dict


# =============================================================================
# MockEmbeddingClient Tests
# =============================================================================

class TestMockEmbeddingClient:
    """Mock Embedding 클라이언트 테스트"""

    def test_embedding_calculation(self, mock_client):
        """기본 embedding 계산 테스트"""
        emb = mock_client.get_embedding("Hello world")

        assert len(emb) > 0
        assert isinstance(emb, type(emb))  # numpy array

    def test_cosine_similarity(self, mock_client):
        """코사인 유사도 테스트"""
        import numpy as np

        text1 = "Python programming language"
        text2 = "Python programming language"  # 동일 텍스트
        text3 = "축구 경기 결과"  # 다른 텍스트

        sim_same = mock_client.cosine_similarity(text1, text2)
        sim_diff = mock_client.cosine_similarity(text1, text3)

        # 동일 텍스트는 유사도 ~1.0 (부동소수점 허용)
        assert np.isclose(sim_same, 1.0)
        # 다른 텍스트는 유사도가 낮음
        assert sim_diff < sim_same

    def test_empty_text_embedding(self, mock_client):
        """빈 텍스트 embedding 테스트"""
        emb = mock_client.get_embedding("")

        # 빈 텍스트는 zero vector
        import numpy as np
        assert np.allclose(emb, np.zeros_like(emb))

    def test_batch_embedding(self, mock_client):
        """배치 embedding 계산 테스트"""
        texts = ["Hello", "World", "Test"]
        embeddings = mock_client.get_embeddings_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(len(e) > 0 for e in embeddings)


# =============================================================================
# BC (Boundary Clarity) Tests
# =============================================================================

class TestBoundaryClarity:
    """Boundary Clarity (BC) 메트릭 테스트"""

    def test_bc_calculation(self, sample_chunks, mock_client):
        """BC 기본 계산 테스트"""
        bc_score = calculate_bc(sample_chunks, mock_client)

        assert isinstance(bc_score, BCScore)
        assert 0 <= bc_score.score <= 2.0  # BC는 보통 0~2 범위
        assert bc_score.pair_count == len(sample_chunks) - 1

    def test_bc_single_chunk(self, mock_client):
        """단일 청크 BC 테스트"""
        single_chunk = ["Only one chunk"]
        bc_score = calculate_bc(single_chunk, mock_client)

        # 단일 청크는 비교할 쌍이 없음
        assert bc_score.score == 1.0
        assert bc_score.pair_count == 0

    def test_bc_independent_chunks(self, mock_client):
        """독립적인 청크들의 BC 테스트"""
        # 서로 다른 주제의 청크들
        independent_chunks = [
            "Python은 프로그래밍 언어입니다.",
            "축구는 스포츠입니다.",
            "음악은 예술의 한 형태입니다.",
        ]
        bc_score = calculate_bc(independent_chunks, mock_client)

        # 독립적인 청크들은 높은 BC를 가져야 함 (약 1.0 근처)
        assert bc_score.score >= 0.8

    def test_bc_score_structure(self, sample_chunks, mock_client):
        """BCScore 구조 테스트"""
        bc_score = calculate_bc(sample_chunks, mock_client)

        assert hasattr(bc_score, "score")
        assert hasattr(bc_score, "pair_scores")
        assert hasattr(bc_score, "min_score")
        assert hasattr(bc_score, "max_score")
        assert hasattr(bc_score, "std_dev")
        assert hasattr(bc_score, "pair_count")

        assert len(bc_score.pair_scores) == bc_score.pair_count

    def test_bc_to_dict(self, sample_chunks, mock_client):
        """BCScore.to_dict() 테스트"""
        bc_score = calculate_bc(sample_chunks, mock_client)
        bc_dict = bc_score.to_dict()

        assert "score" in bc_dict
        assert "pair_count" in bc_dict
        assert "min_score" in bc_dict
        assert "max_score" in bc_dict
        assert "std_dev" in bc_dict


# =============================================================================
# CS (Chunk Stickiness) Tests
# =============================================================================

class TestChunkStickiness:
    """Chunk Stickiness (CS) 메트릭 테스트"""

    def test_cs_calculation(self, sample_chunks, mock_client):
        """CS 기본 계산 테스트"""
        cs_score = calculate_cs(
            sample_chunks, mock_client,
            threshold_k=0.5,  # 낮은 threshold로 테스트
            graph_type="incomplete"
        )

        assert isinstance(cs_score, CSScore)
        assert cs_score.score >= 0  # Entropy는 항상 0 이상
        assert cs_score.node_count == len(sample_chunks)

    def test_cs_single_chunk(self, mock_client):
        """단일 청크 CS 테스트"""
        single_chunk = ["Only one chunk"]
        cs_score = calculate_cs(single_chunk, mock_client)

        assert cs_score.score == 0.0
        assert cs_score.edge_count == 0

    def test_cs_graph_types(self, sample_chunks, mock_client):
        """CS 그래프 타입 테스트"""
        cs_incomplete = calculate_cs(
            sample_chunks, mock_client,
            threshold_k=0.3,
            graph_type="incomplete"
        )
        cs_complete = calculate_cs(
            sample_chunks, mock_client,
            threshold_k=0.3,
            graph_type="complete"
        )

        assert cs_incomplete.graph_type == "incomplete"
        assert cs_complete.graph_type == "complete"

        # Complete 그래프는 더 많은 엣지를 가질 수 있음
        # (같은 threshold에서)
        assert cs_complete.edge_count >= cs_incomplete.edge_count

    def test_cs_threshold_effect(self, sample_chunks, mock_client):
        """CS threshold 영향 테스트"""
        cs_low_threshold = calculate_cs(
            sample_chunks, mock_client,
            threshold_k=0.1,  # 낮은 threshold
            graph_type="incomplete"
        )
        cs_high_threshold = calculate_cs(
            sample_chunks, mock_client,
            threshold_k=0.9,  # 높은 threshold
            graph_type="incomplete"
        )

        # 높은 threshold는 더 적은 엣지
        assert cs_high_threshold.edge_count <= cs_low_threshold.edge_count

    def test_cs_to_dict(self, sample_chunks, mock_client):
        """CSScore.to_dict() 테스트"""
        cs_score = calculate_cs(sample_chunks, mock_client)
        cs_dict = cs_score.to_dict()

        assert "score" in cs_dict
        assert "graph_type" in cs_dict
        assert "node_count" in cs_dict
        assert "edge_count" in cs_dict
        assert "threshold_k" in cs_dict


# =============================================================================
# Edge Weight & Graph Tests
# =============================================================================

class TestGraphConstruction:
    """그래프 구성 테스트"""

    def test_edge_weight_calculation(self, mock_client):
        """엣지 가중치 계산 테스트 (embedding 기반)"""
        import numpy as np

        emb1 = mock_client.get_embedding("Python programming language")
        emb2 = mock_client.get_embedding("Python is a popular programming language")

        weight = calculate_edge_weight_semantic(emb1, emb2)

        assert 0 <= weight <= 1
        assert isinstance(weight, float)

    def test_edge_weight_independent_texts(self, mock_client):
        """독립적인 텍스트의 엣지 가중치 테스트"""
        import numpy as np

        emb1 = mock_client.get_embedding("축구 경기 결과")
        emb2 = mock_client.get_embedding("Python 프로그래밍 언어")

        weight = calculate_edge_weight_semantic(emb1, emb2)

        # 독립적인 텍스트는 낮은 가중치
        assert weight < 0.5

    def test_build_chunk_graph_incomplete(self, sample_chunks, mock_client):
        """Incomplete 그래프 구성 테스트"""
        graph = build_chunk_graph(
            sample_chunks, mock_client,
            threshold_k=0.1,
            graph_type="incomplete"
        )

        assert isinstance(graph, dict)
        assert len(graph) == len(sample_chunks)

        # Incomplete 그래프는 순차적 쌍만
        for node_id, neighbors in graph.items():
            for neighbor_id, weight in neighbors:
                # 인접한 노드만 연결
                assert abs(node_id - neighbor_id) == 1

    def test_build_chunk_graph_complete(self, sample_chunks, mock_client):
        """Complete 그래프 구성 테스트"""
        graph = build_chunk_graph(
            sample_chunks, mock_client,
            threshold_k=0.1,
            graph_type="complete"
        )

        assert isinstance(graph, dict)
        assert len(graph) == len(sample_chunks)

    def test_structural_entropy_empty_graph(self):
        """빈 그래프의 structural entropy 테스트"""
        empty_graph = {}
        entropy = calculate_structural_entropy(empty_graph)

        assert entropy == 0.0

    def test_structural_entropy_no_edges(self):
        """엣지 없는 그래프의 structural entropy 테스트"""
        graph_no_edges = {0: [], 1: [], 2: []}
        entropy = calculate_structural_entropy(graph_no_edges)

        assert entropy == 0.0


# =============================================================================
# Combined Evaluation Tests
# =============================================================================

class TestCombinedEvaluation:
    """통합 평가 테스트"""

    def test_evaluate_chunking(self, sample_chunks, mock_client):
        """evaluate_chunking 통합 테스트"""
        metrics = evaluate_chunking(
            sample_chunks,
            embedding_client=mock_client,
            threshold_k=0.5,
            graph_type="incomplete",
            calculate_cs_flag=True,
            verbose=False
        )

        assert isinstance(metrics, ChunkingMetrics)
        assert metrics.bc_score is not None
        assert metrics.cs_score is not None

    def test_evaluate_chunking_bc_only(self, sample_chunks, mock_client):
        """BC만 계산하는 테스트"""
        metrics = evaluate_chunking(
            sample_chunks,
            embedding_client=mock_client,
            calculate_cs_flag=False,  # CS 스킵
            verbose=False
        )

        assert metrics.bc_score is not None
        assert metrics.cs_score is None

    def test_evaluate_chunking_without_client(self, sample_chunks):
        """Embedding 클라이언트 없이 평가 (Mock 자동 사용)"""
        metrics = evaluate_chunking(
            sample_chunks,
            embedding_client=None,  # Mock 자동 사용
            verbose=False
        )

        assert metrics.bc_score is not None

    def test_chunking_metrics_to_dict(self, sample_chunks, mock_client):
        """ChunkingMetrics.to_dict() 테스트"""
        metrics = evaluate_chunking(
            sample_chunks,
            embedding_client=mock_client,
            verbose=False
        )
        metrics_dict = metrics.to_dict()

        assert "bc" in metrics_dict
        assert "cs" in metrics_dict
        assert metrics_dict["bc"] is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_json_output(self, sample_chunks, mock_client):
        """JSON 출력 형식 테스트"""
        import json

        metrics = evaluate_chunking(
            sample_chunks,
            embedding_client=mock_client,
            verbose=False
        )

        # to_dict()가 JSON 직렬화 가능한지 확인
        metrics_dict = metrics.to_dict()
        json_str = json.dumps(metrics_dict, ensure_ascii=False)

        assert len(json_str) > 0

        # 역직렬화 확인
        parsed = json.loads(json_str)
        assert parsed["bc"]["score"] == metrics.bc_score.score
