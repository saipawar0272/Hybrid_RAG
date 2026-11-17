
import json
from pathlib import Path

import pytest

import src.ingestion as ingestion


def test_ingestion_pipeline_end_to_end(monkeypatch, tmp_path):
    """
    Integration test:
    Real document loading -> real chunking ->
    mocked embeddings -> mocked FAISS persistence ->
    JSONL metadata verification.
    """

    # --------------------------------------------------
    # 1. ARRANGE: CREATE TEST DOCUMENTS
    # --------------------------------------------------

    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    faiss_dir = artifacts_dir / "faiss_index"
    chunks_file = artifacts_dir / "chunks.jsonl"

    data_dir.mkdir(parents=True)

    markdown_file = data_dir / "company_policy.md"
    text_file = data_dir / "it_support.txt"
    csv_file = data_dir / "employees.csv"

    markdown_file.write_text(
        "# Company Leave Policy\n"
        + "Employees receive annual leave according to company policy. " * 15,
        encoding="utf-8"
    )

    text_file.write_text(
        "IT Support Handbook\n"
        + "Employees can contact IT support for technical assistance. " * 15,
        encoding="utf-8"
    )

    csv_file.write_text(
        "id,name\n1,Rahul\n",
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 2. CONFIGURE TEMPORARY DIRECTORIES
    # --------------------------------------------------

    monkeypatch.setattr(ingestion, "DATA_DIR", data_dir)
    monkeypatch.setattr(ingestion, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(ingestion, "FAISS_DIR", faiss_dir)
    monkeypatch.setattr(ingestion, "CHUNKS_FILE", chunks_file)

    monkeypatch.setattr(ingestion, "CHUNK_SIZE", 40)
    monkeypatch.setattr(ingestion, "CHUNK_OVERLAP", 5)

    monkeypatch.setattr(
        ingestion,
        "validate_config",
        lambda: None
    )

    # --------------------------------------------------
    # 3. MOCK EMBEDDING MODEL
    # --------------------------------------------------

    class FakeEmbeddings:

        def __init__(self, **kwargs):
            self.model_name = kwargs["model_name"]

    monkeypatch.setattr(
        ingestion,
        "HuggingFaceEmbeddings",
        FakeEmbeddings
    )

    # --------------------------------------------------
    # 4. MOCK FAISS PERSISTENCE
    # --------------------------------------------------

    class FakeVectorStore:

        def __init__(self, documents):
            self.documents = documents

        def save_local(self, path):
            index_dir = Path(path)
            index_dir.mkdir(parents=True, exist_ok=True)

            # Simulate a persisted index artifact.
            (index_dir / "index_test.marker").write_text(
                "Mock FAISS index",
                encoding="utf-8"
            )

    class FakeFAISS:

        @staticmethod
        def from_documents(documents, embeddings):

            assert isinstance(embeddings, FakeEmbeddings)

            return FakeVectorStore(documents)

    monkeypatch.setattr(
        ingestion,
        "FAISS",
        FakeFAISS
    )

    # --------------------------------------------------
    # 5. ACT: EXECUTE COMPLETE INGESTION PIPELINE
    # --------------------------------------------------

    ingestion.build_index()

    # --------------------------------------------------
    # 6. ASSERT: VERIFY GENERATED ARTIFACTS
    # --------------------------------------------------

    assert artifacts_dir.exists()
    assert faiss_dir.exists()
    assert chunks_file.exists()

    assert (faiss_dir / "index_test.marker").exists()

    # --------------------------------------------------
    # 7. VERIFY CHUNK METADATA
    # --------------------------------------------------

    records = [
        json.loads(line)
        for line in chunks_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) > 0

    # Verify chunk structure.
    for record in records:

        assert "page_content" in record
        assert "metadata" in record

        assert record["page_content"].strip()

        assert "chunk_id" in record["metadata"]
        assert "source" in record["metadata"]

    # Verify unique chunk IDs.
    chunk_ids = [
        record["metadata"]["chunk_id"]
        for record in records
    ]

    assert len(chunk_ids) == len(set(chunk_ids))

    # Verify both supported documents were processed.
    sources = {
        record["metadata"]["source"]
        for record in records
    }

    assert sources == {
        "company_policy.md",
        "it_support.txt"
    }

    # Verify unsupported CSV was ignored.
    assert "employees.csv" not in sources

    # Verify actual chunking occurred.
    assert all(
        len(record["page_content"]) <= 40
        for record in records
    )

