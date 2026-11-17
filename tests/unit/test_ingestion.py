import pytest

from pathlib import Path
from langchain_core.documents import Document

import src.ingestion as ingestion

def test_load_documents_loads_markdown_and_text_files(monkeypatch, tmp_path):

    #Arrange

    markdown_file = tmp_path / "policy.md"
    text_file = tmp_path / "support.txt"

    markdown_file.write_text("# Leav policy", encoding="utf-8")
    text_file.write_text("IT Support Guide", encoding="utf-8")

    monkeypatch.setattr(ingestion, "DATA_DIR", tmp_path)

    def fake_loader(path, encoding=None):

        class FakeLoader:
            def load(self):
                return [
                    Document(
                        page_content=Path(path).read_text(encoding="utf-8"),
                        metadata ={}
                    )
                ]

        return FakeLoader()

    monkeypatch.setattr(
        ingestion,
        "TextLoader",
        fake_loader
    )

    #Act

    documents = ingestion.load_documents()

    #assert

    assert len(documents) == 2

    assert documents[0].metadata["source"] == "policy.md"
    assert documents[1].metadata["source"] == "support.txt"


def test_load_documents_skip_unsupported_files(monkeypatch, tmp_path):

    #arrange

    csv_file = tmp_path / "employees.csv"
    csv_file.write_text("id, name\n1,John", encoding="utf-8")

    monkeypatch.setattr(ingestion, "DATA_DIR", tmp_path)

    #Act

    documents = ingestion.load_documents()

    #Assert
    assert documents == []


def test_load_documents_propogates_loader_errors(monkeypatch, tmp_path):

    #arranhe

    markdown_file = tmp_path / "policy.md"
    markdown_file.write_text("Policy content", encoding="utf-8")

    monkeypatch.setattr(ingestion, "DATA_DIR", tmp_path)

    def failing_loader(path, encoding=None):

        class FakeLoader:

            def load(self):
                raise RuntimeError("Document loading failed")

        return FakeLoader()

    monkeypatch.setattr(
        ingestion,
        "TextLoader",
        failing_loader
    )

    # Act and assert

    with pytest.raises(RuntimeError, match="Document loading failed"):
        ingestion.load_documents()


def test_build_index_creates_faiss_and_metadata(monkeypatch, tmp_path):

    #arrange

    source_doc = Document(
        page_content = "Company leave policy " * 30,
        metadata={"source": "company_policy.md"}
    )

    artifacts_dir = tmp_path / "artifacts"
    faiss_dir = artifacts_dir / "faiss-index"
    chunks_file = artifacts_dir / "chunks.jsonl"

    monkeypatch.setattr(ingestion, "validate_config", lambda: None)
    monkeypatch.setattr(ingestion, "load_documents", lambda: [source_doc])

    monkeypatch.setattr(ingestion, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion, "FAISS_DIR", faiss_dir)
    monkeypatch.setattr(ingestion, "CHUNKS_FILE", chunks_file)

    #fake embedding model

    class FakeEmbeddings:

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        ingestion,
        "HuggingFaceEmbeddings",
        FakeEmbeddings
    )

    #Fake FAISS vector store

    class FakeVectorStore:

        def save_local(save, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeFAISS:

        @staticmethod
        def from_documents(documents, embeddings):
            return FakeVectorStore()

    monkeypatch.setattr(ingestion, "FAISS", FakeFAISS)

    #act

    ingestion.build_index()

    #Assert

    assert faiss_dir.exists()
    assert chunks_file.exists()

    import json

    records = [
        json.loads(line)
        for line in chunks_file.read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) > 0

    for record in records:
        assert "page_content" in record
        assert "metadata" in record
        assert "chunk_id" in record["metadata"]
        assert record["metadata"]["source"] == "company_policy.md"


def test_build_index_raises_error_when_no_documents(monkeypatch):

    #arrange

    monkeypatch.setattr(ingestion, "validate_config", lambda: None)
    monkeypatch.setattr(ingestion, "load_documents", lambda: [])

    # Act and assert

    with pytest.raises(
        RuntimeError, 
        match=r"No .pdf, .txt or .md documents found"
        ):

            ingestion.build_index()


def test_build_index_propogates_embedding_errors(monkeypatch, tmp_path):

    #arrange
    source_doc = Document(
        page_content="Company policy" * 30,
        metadata={"source": "company_policy.md"}
    )

    monkeypatch.setattr(ingestion, "validate_config", lambda: None)
    monkeypatch.setattr(ingestion, "load_documents", lambda: [source_doc])

    monkeypatch.setattr(
        ingestion,
        "HuggingFaceEmbeddings",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Embedding initialization failed")
        )
    )

    monkeypatch.setattr(
        ingestion,
        "ARTIFACTS_DIR",
        tmp_path / "artifacts"
    )

    #act and assert

    with pytest.raises(
        RuntimeError,
        match = "Embedding initialization failed"
    ):
        ingestion.build_index()



    