"""Retrieval result models."""

from dataclasses import dataclass
from datetime import date
import logging
import os
import re

from rank_bm25 import BM25Okapi

from foundation.contract import Evidence
from foundation.store import KnowledgeStore
from retrieval.semantic_query import selected_snapshot
from foundation.source_urls import page_url

logger = logging.getLogger(__name__)
_VECTOR_MODELS = {}
_VECTOR_MODEL_ERRORS = {}
_RERANK_MODELS = {}


@dataclass(frozen=True)
class RetrievalHit:
    evidence: Evidence
    score: float


class EvidenceRetriever:
    def __init__(self, store: KnowledgeStore, *, vector_encoder=None,
                 vector_model_name: str | None = None):
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._hashes: list[str] = []
        self._index_by_hash: dict[str, int] = {}
        self._by_snapshot: dict[str, dict[str, list[Evidence]]] = {}
        self._by_url = {}
        self._built = False
        self._vector_model_name = vector_model_name or os.environ.get(
            'BANKGPT_VECTOR_MODEL', 'sentence-transformers/all-MiniLM-L6-v2',
        )
        self._vector_encoder = vector_encoder
        self._vector_attempted = vector_encoder is not None
        self._vector_disabled = False
        self._vector_cache = {}
        self._vectors = None
        self._numpy = None
        self.vector_error: str | None = None
        self._reranker = None
        self._rerank_attempted = False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(
            r"[a-z0-9]+(?:[.'-][a-z0-9]+)*",
            text.casefold(),
        )

        result = []
        for token in tokens:
            result.append(token)
            if "-" in token:
                result.extend(part for part in token.split("-") if part)

        return result
    
    def build_bm25_index(self) -> None:
        """Build BM25 over unique content hashes."""
        texts: dict[str, str] = {}
        vector_texts: dict[str, str] = {}
        by_snapshot: dict[str, dict[str, list[Evidence]]] = {}
        by_url = {}
        for evidence in self.store.evidence.values():
            texts.setdefault(evidence.content_hash, evidence.text)
            title = evidence.source_title.strip()
            vector_texts.setdefault(
                evidence.content_hash,
                f'{title}\n\n{evidence.text}' if title else evidence.text,
            )
            by_snapshot.setdefault(
                evidence.snapshot_id, {}
            ).setdefault(
                evidence.content_hash, []
            ).append(evidence)
            url = page_url(evidence.source_url)
            if url:
                by_url.setdefault((evidence.snapshot_id, url), []).append(evidence)

        for snapshot in by_snapshot.values():
            for occurrences in snapshot.values():
                occurrences.sort(
                    key=lambda e: (
                        e.in_recommendations,
                        e.source_url,
                        str(e.id),
                    )
                )

        pairs = [
            (key, tokens)
            for key in sorted(texts)
            if (tokens := self._tokenize(texts[key]))
        ]
        self._hashes = [key for key, _ in pairs]
        self._index_by_hash = {key: i for i, key in enumerate(self._hashes)}
        self._bm25 = BM25Okapi([tokens for _, tokens in pairs]) if pairs else None
        self._build_vector_index(vector_texts)
        self._load_reranker()
        self._by_snapshot = by_snapshot
        self._by_url = {key: sorted(records, key=lambda e: (
            e.in_recommendations, e.content_hash, str(e.id))) for key, records in by_url.items()}
        self._built = True

    def _load_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if self._rerank_attempted:
            return None
        self._rerank_attempted = True
        name = os.environ.get('BANKGPT_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        try:
            if name not in _RERANK_MODELS:
                from sentence_transformers import CrossEncoder
                _RERANK_MODELS[name] = CrossEncoder(name, device='cpu', max_length=512)
            self._reranker = _RERANK_MODELS[name]
        except Exception as exc:
            logger.warning('Reranker unavailable: %s', type(exc).__name__)
        return self._reranker

    def _rerank_candidates(self, query, hits, top_k):
        model = self._load_reranker()
        if model is None:
            return hits[:top_k]
        try:
            scores = model.predict(
                [(query, hit.evidence.text) for hit in hits],
                batch_size=16, show_progress_bar=False,
            )
            return [hit for _, hit in sorted(
                zip(scores, hits), key=lambda item: -float(item[0]),
            )[:top_k]]
        except Exception as exc:
            logger.warning('Reranking failed: %s', type(exc).__name__)
            return hits[:top_k]

    def _load_vector_encoder(self) -> bool:
        if self._vector_disabled:
            return False
        if self._vector_encoder is not None:
            return True
        if self._vector_attempted:
            return False

        self._vector_attempted = True
        if self._vector_model_name in _VECTOR_MODELS:
            self._vector_encoder = _VECTOR_MODELS[self._vector_model_name]
            return True
        if self._vector_model_name in _VECTOR_MODEL_ERRORS:
            self.vector_error = _VECTOR_MODEL_ERRORS[self._vector_model_name]
            self._vector_disabled = True
            return False
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._numpy = np
            self._vector_encoder = SentenceTransformer(
                self._vector_model_name, device='cpu',
            )
            _VECTOR_MODELS[self._vector_model_name] = self._vector_encoder
            return True
        except Exception as exc:
            self.vector_error = type(exc).__name__
            _VECTOR_MODEL_ERRORS[self._vector_model_name] = self.vector_error
            self._vector_disabled = True
            logger.warning(
                'Could not load vector model %s (%s); using lexical retrieval',
                self._vector_model_name, self.vector_error,
            )
            return False

    def _build_vector_index(self, texts: dict[str, str]) -> None:
        if not texts or not self._load_vector_encoder():
            self._vectors = None
            return

        try:
            if self._numpy is None:
                import numpy as np
                self._numpy = np
            new_hashes = [h for h in self._hashes if h not in self._vector_cache]
            if new_hashes:
                encoded = self._vector_encoder.encode(
                    [texts[h] for h in new_hashes],
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                if len(encoded) != len(new_hashes):
                    raise ValueError('Vector encoder returned the wrong number of embeddings')
                self._vector_cache.update(
                    (h, self._numpy.asarray(vector, dtype=self._numpy.float32))
                    for h, vector in zip(new_hashes, encoded)
                )
            self._vectors = self._numpy.stack(
                [self._vector_cache[h] for h in self._hashes], axis=0,
            ) if self._hashes else None
        except Exception as exc:
            self.vector_error = type(exc).__name__
            self._vector_disabled = True
            self._vector_encoder = None
            self._vectors = None
            logger.warning('Vector index build failed (%s); using lexical retrieval',
                           self.vector_error)

    def _vector_hits(self, query: str, eligible, *, top_k: int):
        if top_k <= 0 or self._vectors is None or not self._load_vector_encoder():
            return []
        try:
            query_vector = self._vector_encoder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            query_vector = self._numpy.asarray(query_vector, dtype=self._numpy.float32)
            indexed = [(self._index_by_hash[h], h) for h in eligible
                       if h in self._index_by_hash]
            if not indexed:
                return []
            indices = [i for i, _ in indexed]
            scores = self._vectors[indices] @ query_vector
            ranked = sorted(
                ((float(score), h) for score, (_, h) in zip(scores, indexed)),
                key=lambda pair: (-pair[0], pair[1]),
            )
            return [
                RetrievalHit(
                    evidence=min(
                        eligible[content_hash],
                        key=lambda e: (e.in_recommendations, e.source_url, str(e.id)),
                    ),
                    score=score,
                )
                for score, content_hash in ranked[:top_k]
                if score > 0.
            ]
        except Exception as exc:
            self.vector_error = type(exc).__name__
            self._vector_disabled = True
            self._vector_encoder = None
            self._vectors = None
            logger.warning('Vector query failed (%s); using lexical retrieval',
                           self.vector_error)
            return []

    def lookup_url(self, url, *, as_of=None):
        """All source occurrences at the exact target in the selected snapshot.

        Fragment selects a browser region, not another fetched page. Query
        strings, host and scheme remain significant; no fabricated redirects.
        """
        if not self._built:
            self.build_bm25_index()
        snapshot = selected_snapshot(self.store, as_of)
        key = page_url(url)
        if snapshot is None or key is None:
            return []
        return [e for e in self._by_url.get((snapshot.snapshot_id, key), ())
                if not e.in_recommendations]

    def search(
        self,
        query: str,
        *,
        as_of: str | None = None,
        top_k: int = 10,
    ) -> list[RetrievalHit]:
        """Search the latest complete snapshot at or before as_of."""
        if not self._built:
            self.build_bm25_index()

        if top_k <= 0:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        cutoff = date.fromisoformat(as_of) if as_of else date.max
        snapshots = [
            s for s in self.store.snapshots.values()
            if date.fromisoformat(s.taken_on) <= cutoff
        ]
        if not snapshots:
            return []

        snapshot = max(
            snapshots,
            key=lambda s: (date.fromisoformat(s.taken_on), s.snapshot_id),
        )
        eligible = self._by_snapshot.get(snapshot.snapshot_id, {})
        lexical = []
        if self._bm25 is not None:
            indices = [
                self._index_by_hash[h]
                for h in eligible
                if h in self._index_by_hash
            ]
            indices = [
                i for i in indices
                if any(t in self._bm25.doc_freqs[i] for t in tokens)
            ]
            scores = self._bm25.get_batch_scores(tokens, indices)
            ranked = sorted(
                (
                    (i, float(score))
                    for i, score in zip(indices, scores)
                    if score > 0
                ),
                key=lambda pair: (-pair[1], self._hashes[pair[0]]),
            )
            lexical = [
                RetrievalHit(
                    evidence=min(
                        eligible[self._hashes[i]],
                        key=lambda e: (
                            e.in_recommendations,
                            e.source_url,
                            str(e.id),
                        ),
                    ),
                    score=score,
                )
                for i, score in ranked
            ]
        vector = self._vector_hits(query, eligible, top_k=max(top_k, 30))
        # Fuse the lexical and vector Evidence rankings.
        fused, sources = {}, {}
        for hits in (lexical, vector):
            for rank, hit in enumerate(hits, 1):
                h = hit.evidence.content_hash
                fused[h] = fused.get(h, 0.) + 1. / (60 + rank)
                sources[h] = hit.evidence
        candidates = [RetrievalHit(sources[h], fused[h]) for h in sorted(
            fused, key=lambda h: (-fused[h], h))[:max(top_k, 30)]]
        return self._rerank_candidates(query, candidates, top_k)
