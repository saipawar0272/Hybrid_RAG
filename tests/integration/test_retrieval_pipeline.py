
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

import src.hybrid_retriever as retriever_module


def test_hybrid_retrieval_pipeline_end_to_end(monkeypatch, tmp_path):
    """
    Integration test:
    Load persisted chunk metadata -> initialize BM25 ->
    perform dense retrieval -> perform sparse retrieval ->
    apply RRF fusion -> return final ranked documents.
    """

    # --------------------------------------------------
    # 1. ARRANGE: CREATE PERSISTED TEST ARTIFACTS
    # --------------------------------------------------

    artifacts_dir = tmp_path / "artifacts"
    faiss_dir = artifacts_dir / "faiss_index"
    chunks_file = artifacts_dir / "chunks.jsonl"

    faiss_dir.mkdir(parents=True)

    documents = [
        Document(
            page_content=(
                "Employee annual leave policy vacation entitlement "
                "annual leave policy employee vacation days"
            ),
            metadata={
                "source": "company_policy.md",
                "chunk_id": "chunk-1"
            }
        ),
        Document(
            page_content=(
                "Employee leave policy annual leave and vacation days"
            ),
            metadata={
                "source": "employee_handbook.md",
                "chunk_id": "chunk-2"
            }
        ),
        Document(
            page_content=(
                "Database connection timeout error troubleshooting guide"
            ),
            metadata={
                "source": "engineering_handbook.md",
                "chunk_id": "chunk-3"
            }
        )
    ]

    with chunks_file.open("w", encoding="utf-8") as file:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }

            file.write(json.dumps(record) + "\n")

    # --------------------------------------------------
    # 2. CONFIGURE RETRIEVER
    # --------------------------------------------------

    monkeypatch.setattr(
        retriever_module,
        "FAISS_DIR",
        faiss_dir
    )

    monkeypatch.setattr(
        retriever_module,
        "CHUNKS_FILE",
        chunks_file
    )

    monkeypatch.setattr(
        retriever_module,
        "validate_config",
        lambda: None
    )

    monkeypatch.setattr(
        retriever_module,
        "DENSE_TOP_K",
        1
    )

    monkeypatch.setattr(
        retriever_module,
        "SPARSE_TOP_K",
        3
    )

    monkeypatch.setattr(
        retriever_module,
        "RRF_C",
        60
    )

    monkeypatch.setattr(
        retriever_module,
        "RRF_DENSE_WEIGHT",
        0.5
    )

    monkeypatch.setattr(
        retriever_module,
        "RRF_SPARSE_WEIGHT",
        0.5
    )

    monkeypatch.setattr(
        retriever_module,
        "ENABLE_RERANKER",
        False
    )

    # --------------------------------------------------
    # 3. MOCK EMBEDDING MODEL
    # --------------------------------------------------

    class FakeEmbeddings:

        def __init__(self, **kwargs):
            self.model_name = kwargs["model_name"]

    monkeypatch.setattr(
        retriever_module,
        "HuggingFaceEmbeddings",
        FakeEmbeddings
    )

    # --------------------------------------------------
    # 4. MOCK FAISS VECTOR STORE
    # --------------------------------------------------

    class FakeVectorStore:

        def similarity_search(self, query, k):
            assert query == "annual leave policy"
            assert k == 1

            # Simulate dense retrieval returning chunk-3.
            # This intentionally differs from BM25's top result.
            return [documents[2]]

    class FakeFAISS:

        @staticmethod
        def load_local(
            path,
            embeddings,
            allow_dangerous_deserialization
        ):
            assert Path(path) == faiss_dir
            assert isinstance(embeddings, FakeEmbeddings)
            assert allow_dangerous_deserialization is True

            return FakeVectorStore()

    monkeypatch.setattr(
        retriever_module,
        "FAISS",
        FakeFAISS
    )

    # --------------------------------------------------
    # 5. ACT: INITIALIZE HYBRID RETRIEVER
    # --------------------------------------------------

    retriever = retriever_module.HybridRetriever()

    # --------------------------------------------------
    # 6. ACT: EXECUTE HYBRID RETRIEVAL
    # --------------------------------------------------

    results = retriever.retrieve(
        query="annual leave policy",
        final_k=2
    )

    # --------------------------------------------------
    # 7. ASSERT: VERIFY RETRIEVAL RESULTS
    # --------------------------------------------------

    assert len(results) == 2

    result_ids = [
        doc.metadata["chunk_id"]
        for doc in results
    ]

    # Dense retrieval contributes chunk-3 at rank 1.
    # BM25 contributes chunk-1, chunk-2, chunk-3.
    # RRF combines these rankings.
    assert result_ids == ["chunk-3", "chunk-1"]

    # Verify document content and source metadata.
    assert results[0].metadata["source"] == (
        "engineering_handbook.md"
    )

    assert results[1].metadata["source"] == (
        "company_policy.md"
    )

    assert "Database connection timeout" in results[0].page_content
    assert "annual leave policy" in results[1].page_content.lower()

    # Verify BM25 was initialized with all persisted documents.
    assert len(retriever.documents) == 3


