"""Tests for re-ranking backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from archivist.config import Config, RerankerConfig, RetrievalConfig
from archivist.retrieval.models import SearchResult
from archivist.retrieval.reranker import (
    LocalReranker,
    VoyageReranker,
    get_reranker,
)


def _make_result(index: int, score: float = 0.5) -> SearchResult:
    return SearchResult(
        text=f"Document text number {index}",
        score=score,
        source_file=f"doc{index}.pdf",
        family_slug="test",
        doc_title=f"Doc {index}",
        doc_type="other",
        version=None,
        page_number=None,
        heading_path=None,
        chunk_role="base",
    )


def _make_results(n: int = 5) -> list[SearchResult]:
    return [_make_result(i, score=float(n - i)) for i in range(n)]


class TestLocalReranker:
    def test_rerank_returns_top_k(self) -> None:
        with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
            mock_model = MagicMock()
            mock_ce_cls.return_value = mock_model
            mock_model.predict.return_value = [0.1, 0.3, 0.9, 0.5, 0.2]

            reranker = LocalReranker(model_name="test-model")
            results = _make_results(5)
            reranked = reranker.rerank("test query", results, top_k=3)

            assert len(reranked) == 3
            assert reranked[0].source_file == "doc2.pdf"
            assert reranked[0].score == pytest.approx(0.9)
            assert reranked[1].source_file == "doc3.pdf"
            assert reranked[1].score == pytest.approx(0.5)

    def test_rerank_empty_results(self) -> None:
        reranker = LocalReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_lazy_model_loading(self) -> None:
        reranker = LocalReranker()
        assert reranker._model is None

        with patch("sentence_transformers.CrossEncoder") as mock_ce_cls:
            mock_model = MagicMock()
            mock_ce_cls.return_value = mock_model
            mock_model.predict.return_value = [0.5]

            reranker.rerank("q", [_make_result(0)], top_k=1)
            mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
            assert reranker._model is not None


class TestVoyageReranker:
    def test_rerank_returns_top_k(self) -> None:
        with patch("voyageai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_r0 = MagicMock()
            mock_r0.index = 2
            mock_r0.relevance_score = 0.95

            mock_r1 = MagicMock()
            mock_r1.index = 0
            mock_r1.relevance_score = 0.80

            mock_response = MagicMock()
            mock_response.results = [mock_r0, mock_r1]
            mock_client.rerank.return_value = mock_response

            reranker = VoyageReranker(api_key="test-key", model="rerank-2")
            results = _make_results(5)
            reranked = reranker.rerank("test query", results, top_k=2)

            assert len(reranked) == 2
            assert reranked[0].source_file == "doc2.pdf"
            assert reranked[0].score == pytest.approx(0.95)
            assert reranked[1].source_file == "doc0.pdf"
            assert reranked[1].score == pytest.approx(0.80)

    def test_rerank_empty_results(self) -> None:
        reranker = VoyageReranker(api_key="key")
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


class TestGetReranker:
    def _make_config(
        self,
        enabled: bool = False,
        reranker_type: str = "local",
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        api_key: str | None = None,
    ) -> Config:
        config = Config.default()
        config.retrieval = RetrievalConfig(
            reranker=RerankerConfig(
                enabled=enabled,
                type=reranker_type,
                model=model,
                api_key=api_key,
            ),
        )
        return config

    def test_disabled_returns_none(self) -> None:
        config = self._make_config(enabled=False)
        assert get_reranker(config) is None

    def test_local_returns_local_reranker(self) -> None:
        config = self._make_config(enabled=True, reranker_type="local")
        reranker = get_reranker(config)
        assert isinstance(reranker, LocalReranker)

    def test_api_returns_voyage_reranker(self) -> None:
        config = self._make_config(enabled=True, reranker_type="api", api_key="key")
        reranker = get_reranker(config)
        assert isinstance(reranker, VoyageReranker)

    def test_api_without_key_raises(self) -> None:
        config = self._make_config(enabled=True, reranker_type="api", api_key=None)
        with pytest.raises(ValueError, match="API key"):
            get_reranker(config)

    def test_default_config_disabled(self) -> None:
        config = Config.default()
        assert get_reranker(config) is None
