"""Dense FAISS + sparse BM25 retrieval with weighted Reciprocal Rank Fusion."""
import json
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    FAISS_DIR, CHUNKS_FILE, EMBEDDING_MODEL, DENSE_TOP_K,
    SPARSE_TOP_K, RRF_C, RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT,
    FINAL_TOP_K, ENABLE_RERANKER, RERANKER_MODEL, validate_config
)


def tokenize(text: str):
    # Keep technical identifiers such as ERR-504 and XJ_1029 intact.
    return re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())


class HybridRetriever:
    
    def __init__(self):
        validate_config()
        if not FAISS_DIR.exists() or not CHUNKS_FILE.exists():
            raise RuntimeError("Index not found. Run: python -m src.ingestion")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
        # This is a locally generated FAISS index. Only load indexes you trust.
        self.vector_store = FAISS.load_local(
            str(FAISS_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.documents = []
        with CHUNKS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.documents.append(
                    Document(
                        page_content=row["page_content"],
                        metadata=row["metadata"]
                    )
                )

        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.documents])
        self.reranker = None
        if ENABLE_RERANKER:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKER_MODEL)

    @staticmethod
    def _key(doc: Document):
        return str(doc.metadata.get("chunk_id") or (
            doc.metadata.get("source", ""),
            doc.metadata.get("start_index", ""),
            doc.page_content[:100]
        ))

    def retrieve(self, query: str, final_k: int = FINAL_TOP_K) -> List[Document]:
        dense = self.vector_store.similarity_search(query, k=DENSE_TOP_K)

        sparse_scores = self.bm25.get_scores(tokenize(query))
        sparse_indices = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True
        )[:SPARSE_TOP_K]
        sparse = [self.documents[i] for i in sparse_indices]

        # Weighted RRF: score = sum(weight / (c + 1-based rank)).
        fused_scores = {}
        docs_by_key = {}
        for weight, ranked_docs in [
            (RRF_DENSE_WEIGHT, dense),
            (RRF_SPARSE_WEIGHT, sparse)
        ]:
            for rank, doc in enumerate(ranked_docs, start=1):
                key = self._key(doc)
                docs_by_key[key] = doc
                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    weight / (RRF_C + rank)
                )

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )
        candidates = [docs_by_key[k] for k in ranked_keys]

        if self.reranker and candidates:
            pairs = [(query, d.page_content) for d in candidates]
            scores = self.reranker.predict(pairs)
            candidates = [
                d for d, _ in sorted(
                    zip(candidates, scores),
                    key=lambda pair: float(pair[1]),
                    reverse=True
                )
            ]

        return candidates[:final_k]
