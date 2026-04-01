"""Re-ranking backends for retrieval results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

from archivist.retrieval.models import SearchResult

if TYPE_CHECKING:
    from archivist.config import Config


class Reranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-rank results and return top_k."""


class LocalReranker(Reranker):
    """Re-ranker using sentence_transformers CrossEncoder.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default.
    The model is lazy-loaded on first call to rerank().
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-rank results using a cross-encoder model."""
        if not results:
            return []

        model = self._load_model()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)  # type: ignore[union-attr]

        reranked = [
            replace(r, score=float(s))
            for r, s in zip(results, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]


class VoyageReranker(Reranker):
    """Re-ranker using the Voyage AI rerank API."""

    def __init__(self, api_key: str, model: str = "rerank-2") -> None:
        self._api_key = api_key
        self._model = model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-rank results using the Voyage rerank API."""
        if not results:
            return []

        import voyageai

        client = voyageai.Client(api_key=self._api_key)
        documents = [r.text for r in results]
        response = client.rerank(query, documents, model=self._model, top_k=top_k)

        reranked: list[SearchResult] = []
        for item in response.results:
            original = results[item.index]
            reranked.append(replace(original, score=float(item.relevance_score)))

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]


def get_reranker(config: Config) -> Reranker | None:
    """Return a Reranker based on config, or None if disabled."""
    reranker_cfg = config.retrieval.reranker

    if not reranker_cfg.enabled:
        return None

    if reranker_cfg.type == "api":
        api_key = reranker_cfg.api_key
        if api_key is None:
            msg = "Voyage reranker requires an API key (retrieval.reranker.api_key)"
            raise ValueError(msg)
        return VoyageReranker(api_key=api_key, model=reranker_cfg.model)

    return LocalReranker(model_name=reranker_cfg.model)
