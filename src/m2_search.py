from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re
from dataclasses import dataclass
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
        return re.sub(r"\s+", " ", normalized).strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenized_query)
        else:
            query_terms = set(tokenized_query)
            scores = [
                float(sum(1 for token in tokens if token in query_terms))
                for tokens in self.corpus_tokens
            ]

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[idx]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None
        self._fallback_documents: list[dict] = []
        self._fallback_vectors: list[dict[str, int]] = []

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = False
        return self._encoder if self._encoder is not False else None

    def _fallback_vectorize(self, text: str) -> dict[str, int]:
        tokens = segment_vietnamese(text).split()
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _fallback_similarity(self, a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        numerator = sum(a[token] * b.get(token, 0) for token in a)
        norm_a = sqrt(sum(value * value for value in a.values()))
        norm_b = sqrt(sum(value * value for value in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return numerator / (norm_a * norm_b)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._fallback_documents = chunks
        self._fallback_vectors = [self._fallback_vectorize(chunk["text"]) for chunk in chunks]

        encoder = self._get_encoder()
        if self.client is None or encoder is None or not chunks:
            return

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

            texts = [chunk["text"] for chunk in chunks]
            vectors = encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunk.get("metadata", {}), "text": chunk["text"]},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception:
            # Keep fallback corpus in memory even if Qdrant/indexing fails.
            return

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        if self.client is not None and encoder is not None:
            try:
                query_vector = encoder.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                )
                points = getattr(response, "points", response)
                results: list[SearchResult] = []
                for point in points:
                    payload = dict(point.payload or {})
                    results.append(SearchResult(
                        text=payload.get("text", ""),
                        score=float(getattr(point, "score", 0.0)),
                        metadata={k: v for k, v in payload.items() if k != "text"},
                        method="dense",
                    ))
                if results:
                    return results
            except Exception:
                pass

        query_vector = self._fallback_vectorize(query)
        scored = []
        for doc, doc_vector in zip(self._fallback_documents, self._fallback_vectors):
            score = self._fallback_similarity(query_vector, doc_vector)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            SearchResult(
                text=doc["text"],
                score=float(score),
                metadata=doc.get("metadata", {}),
                method="dense",
            )
            for score, doc in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict[str, object]] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] = float(rrf_scores[result.text]["score"]) + 1.0 / (k + rank + 1)

    fused = sorted(
        rrf_scores.values(),
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:top_k]

    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in fused
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
