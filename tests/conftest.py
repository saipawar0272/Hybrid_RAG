
import json
import pytest


from langchain_core.documents import Document

import src.hybrid_retriever as retriever_module


@pytest.fixture
def sample_document():
    return Document(
        page_content="Company leave policy",
        metadata={
            "chunk_id": "chunk-101",
            "source" : "company_policy.md",
            "start_index": 0
        }
    )

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Company leave policy",
            metadata={"chunk_id":"A", "source": "policy.md"}
        ),
        Document(
            page_content="Employee salary policy",
            metadata={"chunk_id": "B", "source": "salary.md"}
        ),
        Document(
            page_content="IT support policy",
            metadata ={"chunk_id": "C", "source": "it_support.md"}
        )
    ]


@pytest.fixture
def constructor_setup(monkeypatch, tmp_path):
    """
    Prepare fake dependencies and temporary index files
    for HybridRetriever constructor tests.
    """

    # Create temporary FAISS directory
    faiss_dir = tmp_path / "faiss_index"
    faiss_dir.mkdir()

    # Create temporary chunk metadata file
    chunks_file = tmp_path / "chunks.jsonl"

    metadata = {
        "page_content": "Company leave policy",
        "metadata": {
            "chunk_id": "chunk-101",
            "source": "company_policy.md"
        }
    }

    chunks_file.write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8"
    )

    # Fake embedding model
    class FakeEmbeddings:
        def __init__(self, model_name, encode_kwargs):
            self.model_name = model_name
            self.encode_kwargs = encode_kwargs

    # Fake FAISS implementation
    class FakeFAISS:

        @staticmethod
        def load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization
        ):
            return "FAISS_INDEX_LOADED"

    # Fake BM25 implementation
    class FakeBM25:

        def __init__(self, corpus):
            self.corpus = corpus

    # Disable config validation for isolated constructor tests.
    monkeypatch.setattr(
        retriever_module,
        "validate_config",
        lambda: None
    )

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
        "HuggingFaceEmbeddings",
        FakeEmbeddings
    )

    monkeypatch.setattr(
        retriever_module,
        "FAISS",
        FakeFAISS
    )

    monkeypatch.setattr(
        retriever_module,
        "BM25Okapi",
        FakeBM25
    )

    monkeypatch.setattr(
        retriever_module,
        "ENABLE_RERANKER",
        False
    )

    return {
        "faiss_dir": faiss_dir,
        "chunks_file": chunks_file,
        "metadata": metadata
    }

