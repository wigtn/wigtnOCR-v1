"""
Chunking CLI 단위 테스트

MoC-based chunking quality metrics (BC, CS) 테스트
"""

import pytest

from src.chunking.chunker import (
    ChunkerConfig,
    ChunkingStrategy,
    create_chunker,
    Chunk,
)
from src.chunking.metrics import (
    BCScore,
    CSScore,
    ChunkingMetrics,
    MockLLMClient,
    calculate_bc,
    calculate_cs,
    calculate_edge_weight,
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
    """Mock LLM 클라이언트"""
    return MockLLMClient()


# =============================================================================
# Chunker Tests
# =============================================================================

class TestChunker:
    """청킹 기능 테스트"""

    def test_recursive_character_chunker(self, sample_text):
        """RecursiveCharacterChunker 기본 동작 테스트"""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=200,
            chunk_overlap=20,
        )
        chunker = create_chunker(config)

        chunks = chunker.chunk(sample_text, document_id="test")

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content for c in chunks)

    def test_fixed_size_chunker(self, sample_text):
        """FixedSizeChunker 기본 동작 테스트"""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=100,
            chunk_overlap=10,
        )
        chunker = create_chunker(config)

        chunks = chunker.chunk(sample_text, document_id="test")

        assert len(chunks) > 0
        # 고정 크기 청커는 대부분 청크가 비슷한 크기
        for chunk in chunks[:-1]:  # 마지막 청크 제외
            assert chunk.length <= config.chunk_size + 10  # 약간의 여유

    def test_chunk_overlap(self, sample_text):
        """청크 오버랩 테스트"""
        config = ChunkerConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=100,
            chunk_overlap=20,
        )
        chunker = create_chunker(config)

        chunks = chunker.chunk(sample_text, document_id="test")

        # 최소 2개 청크가 있어야 오버랩 테스트 가능
        if len(chunks) >= 2:
            # 오버랩이 있으면 연속 청크 사이에 공통 텍스트가 있을 수 있음
            # (실제로 공유되는지는 텍스트 분할 방식에 따라 다름)
            assert all(c.start_index >= 0 for c in chunks)

    def test_chunk_metadata(self, sample_text):
        """청크 메타데이터 테스트"""
        config = ChunkerConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER)
        chunker = create_chunker(config)

        chunks = chunker.chunk(sample_text, document_id="my_doc")

        for i, chunk in enumerate(chunks):
            assert chunk.id.startswith("my_doc_chunk_")
            assert "document_id" in chunk.metadata
            assert chunk.metadata["document_id"] == "my_doc"

    def test_empty_text(self):
        """빈 텍스트 처리 테스트"""
        config = ChunkerConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER)
        chunker = create_chunker(config)

        chunks = chunker.chunk("", document_id="empty")

        assert len(chunks) == 0

    def test_chunk_to_dict(self, sample_text):
        """Chunk.to_dict() 테스트"""
        config = ChunkerConfig(strategy=ChunkingStrategy.RECURSIVE_CHARACTER)
        chunker = create_chunker(config)

        chunks = chunker.chunk(sample_text, document_id="test")

        if chunks:
            chunk_dict = chunks[0].to_dict()
            assert "id" in chunk_dict
            assert "content" in chunk_dict
            assert "start_index" in chunk_dict
            assert "end_index" in chunk_dict
            assert "length" in chunk_dict
            assert "metadata" in chunk_dict


# =============================================================================
# MockLLMClient Tests
# =============================================================================

class TestMockLLMClient:
    """Mock LLM 클라이언트 테스트"""

    def test_perplexity_calculation(self, mock_client):
        """기본 perplexity 계산 테스트"""
        ppl = mock_client.calculate_perplexity("Hello world")

        assert ppl > 0
        assert isinstance(ppl, float)

    def test_perplexity_with_context(self, mock_client):
        """컨텍스트가 있는 perplexity 계산 테스트"""
        text = "Python programming"
        context = "Python is a programming language. Python programming"

        ppl_alone = mock_client.calculate_perplexity(text)
        ppl_with_context = mock_client.calculate_perplexity(text, context=context)

        # 컨텍스트와 겹치는 단어가 있으면 perplexity가 낮아져야 함
        assert ppl_with_context <= ppl_alone

    def test_empty_text_perplexity(self, mock_client):
        """빈 텍스트 perplexity 테스트"""
        ppl = mock_client.calculate_perplexity("")

        assert ppl == 1.0

    def test_batch_perplexity(self, mock_client):
        """배치 perplexity 계산 테스트"""
        texts = ["Hello", "World", "Test"]
        ppls = mock_client.calculate_perplexity_batch(texts)

        assert len(ppls) == len(texts)
        assert all(p > 0 for p in ppls)


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
        """엣지 가중치 계산 테스트"""
        q = "Python programming language"
        d = "Python is a popular programming language"

        weight = calculate_edge_weight(q, d, mock_client)

        assert 0 <= weight <= 1
        assert isinstance(weight, float)

    def test_edge_weight_independent_texts(self, mock_client):
        """독립적인 텍스트의 엣지 가중치 테스트"""
        q = "축구 경기 결과"
        d = "Python 프로그래밍 언어"

        weight = calculate_edge_weight(q, d, mock_client)

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
            llm_client=mock_client,
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
            llm_client=mock_client,
            calculate_cs_flag=False,  # CS 스킵
            verbose=False
        )

        assert metrics.bc_score is not None
        assert metrics.cs_score is None

    def test_evaluate_chunking_without_client(self, sample_chunks):
        """LLM 클라이언트 없이 평가 (Mock 자동 사용)"""
        metrics = evaluate_chunking(
            sample_chunks,
            llm_client=None,  # Mock 자동 사용
            verbose=False
        )

        assert metrics.bc_score is not None

    def test_chunking_metrics_to_dict(self, sample_chunks, mock_client):
        """ChunkingMetrics.to_dict() 테스트"""
        metrics = evaluate_chunking(
            sample_chunks,
            llm_client=mock_client,
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

    def test_end_to_end_pipeline(self, sample_text, mock_client):
        """전체 파이프라인 통합 테스트: 텍스트 → 청킹 → 평가"""
        # 1. 청킹
        config = ChunkerConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=150,
            chunk_overlap=20,
        )
        chunker = create_chunker(config)
        chunks = chunker.chunk(sample_text, document_id="integration_test")

        assert len(chunks) >= 2, "통합 테스트를 위해 최소 2개 청크 필요"

        # 2. BC/CS 평가
        metrics = evaluate_chunking(
            chunks,
            llm_client=mock_client,
            threshold_k=0.5,
            graph_type="incomplete",
            verbose=False
        )

        # 3. 결과 검증
        assert metrics.bc_score is not None
        assert metrics.bc_score.score > 0
        assert metrics.bc_score.pair_count == len(chunks) - 1

        assert metrics.cs_score is not None
        assert metrics.cs_score.node_count == len(chunks)

    def test_json_output(self, sample_chunks, mock_client):
        """JSON 출력 형식 테스트"""
        import json

        metrics = evaluate_chunking(
            sample_chunks,
            llm_client=mock_client,
            verbose=False
        )

        # to_dict()가 JSON 직렬화 가능한지 확인
        metrics_dict = metrics.to_dict()
        json_str = json.dumps(metrics_dict, ensure_ascii=False)

        assert len(json_str) > 0

        # 역직렬화 확인
        parsed = json.loads(json_str)
        assert parsed["bc"]["score"] == metrics.bc_score.score
