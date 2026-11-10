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

from src.logging_config import get_logger

#create module specific logger
logger = get_logger(__name__)


def tokenize(text: str):
    """Tokenize text while preserving technical identifiers."""

    # Keep technical identifiers such as ERR-504 and XJ_1029 intact.
    return re.findall(
        r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*",
        text.lower()
        )


class HybridRetriever:

    def __init__(self):

        validate_config()

        logger.info("Initializing HybridRetriever")

        try:

            #2. verify index availability
            if not FAISS_DIR.exists() or not CHUNKS_FILE.exists():

                raise RuntimeError(
                    "Index not found. Run: python -m src.ingestion"
                    )

            #3. initialize embeddings
            logger.info(
                "initializing embedding model: %s",
                EMBEDDING_MODEL
            )


            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True}
            )

            #4. Load FAISS index
            # This is a locally generated FAISS index. Only load indexes you trust.
            logger.info("Loading FAISS index")

            self.vector_store = FAISS.load_local(
                str(FAISS_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info("FAISS index loaded successfully")

            #5.l load chunk metadata

            logger.info("Loading chunk metadata")

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

            logger.info(
                "Loaded %d documents for BM25 retrieval",
                len(self.documents)
            )

            if not self.documents:
                raise RuntimeError(
                    "No documents found in chunk metadata."
                )

            #6. Initialize BM25

            logger.info("Initializing BM25 sparse retriever")


            self.bm25 = BM25Okapi(
                [tokenize(d.page_content) for d in self.documents]
                )

            logger.info("BM25 retriever initialized successfully")

            #7. initialize optional reranker

            self.reranker = None

            if ENABLE_RERANKER:
                logger.info(
                    "Initialized cross-encoder reranker: %s",
                    RERANKER_MODEL
                )

                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(RERANKER_MODEL)

                logger.info("Reranker initialized successfully")

            else:

                logger.info(
                    "Cross-encoder reranking is disabled"
                )

            logger.info(
                "HybridRetriever initialized successfully"
            )

        except Exception:

            logger.exception(
                "HybridRetriever initialization failed"
            )

            raise



    @staticmethod
    def _key(doc: Document):

        return str(
            doc.metadata.get("chunk_id")
            or (
            doc.metadata.get("source", ""),
            doc.metadata.get("start_index", ""),
            doc.page_content[:100]
            )
        )

    def retrieve(self,
                 query: str,
                 final_k: int = FINAL_TOP_K
    ) -> List[Document]:

        logger.info(
            "starting hybrid retrieval. requested final_k: %d",
            final_k
        )

        try:

            #1. Dense retrieval using FAISS

            dense = self.vector_store.similarity_search(
                query,
                k=DENSE_TOP_K
            )

            logger.debug(
                "Dense retrieval returned %d documents",
                len(dense)

            )

            #2. sparse retrieval using BM25

            sparse_scores = self.bm25.get_scores(
                tokenize(query)
            )

            sparse_indices = sorted(
                range(len(sparse_scores)),
                key=lambda i: sparse_scores[i],
                reverse=True
            )[:SPARSE_TOP_K]

            sparse = [
                self.documents[i]
                for i in sparse_indices
            ]

            logger.debug(
                "sparse retrieval returned %d documents",
                len(sparse)
            )

            #3. weighted Reciprocal Rank fusion
            # Weighted RRF: score = sum(weight / (c + 1-based rank)).
            fused_scores = {}
            docs_by_key = {}

            for weight, ranked_docs in [
                (RRF_DENSE_WEIGHT, dense),
                (RRF_SPARSE_WEIGHT, sparse)
            ]:

                for rank, doc in enumerate(
                    ranked_docs,
                    start=1
                    ):

                    key = self._key(doc)

                    docs_by_key[key] = doc

                    fused_scores[key] = (
                        fused_scores.get(key, 0.0)
                        + weight / (RRF_C + rank)
                    )


            ranked_keys = sorted(
                fused_scores,
                key=fused_scores.get,
                reverse=True
            )

            candidates = [
                docs_by_key[k]
                for k in ranked_keys
                ]

            logger.info(
                "RRF fusion completed. Candidate count: %d",
                len(candidates)
            )

                #4. optional cross encoder reranking

            if self.reranker and candidates:

               logger.info("starting cross encoder reranking")

               pairs = [
                    (query, d.page_content)
                    for d in candidates
               ]

               scores = self.reranker.predict(pairs)

               candidates = [
                    d
                    for d, _ in sorted(
                        zip(candidates, scores),
                        key=lambda pair: float(pair[1]),
                        reverse=True
                    )
                ]

               logger.info("cross encoder reranking completed")

            #5. return funal documents

            final_results = candidates[:final_k]

            logger.info(
                "Hybrif retriebal completed final documents: %d",
                len(final_results)
            )

            return final_results

        except Exception :

            logger.exception("hybrid retrieval failed")

            raise


