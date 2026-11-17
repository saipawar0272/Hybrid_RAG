import pytest

from langchain_core.documents import Document

import src.hybrid_retriever as retriever_module


#--------------------------------
# 1. Tokenization tests
#--------------------------------

def test_tokenize_preserves_technical_identifiers():
    text = "Error ERR-504 occured in service XJ_1029"

    result = retriever_module.tokenize(text)

    assert result == [
        "error",
        "err-504",
        "occured",
        "in",
        "service",
        "xj_1029"
    ]


def test_tokenize_handles_empty_string():
    result = retriever_module.tokenize("")

    assert result == []

#-----------------------
# 2. Document key tests
#----------------------

def test_key_uses_chunk_id():
    doc = Document(
        page_content="Company leave policy",
        metadata ={"chunk_id": "chunk-101"}
    )

    result = retriever_module.HybridRetriever._key(doc)

    assert result == "chunk-101"

def test_key_uses_fallback_when_chunk_id_missing():
    doc = Document(
        page_content="Company leave policy",
        metadata={
            "source": "company_policy.md",
            "start_index": 100
        }
    )

    result = retriever_module.HybridRetriever._key(doc)

    assert result == str(
        ("company_policy.md", 100, "Company leave policy")
    )

#-------------------
# 3. Hybrud retriever and RRF tests
#-------------------

def test_retrieve_combines_dense_and_sparse_results(monkeypatch):
    doc_a = Document(
        page_content="Company leave policy",
        metadata = {"chunk_id":"A"}
    )

    doc_b = Document(
        page_content="Employee Salary policy",
        metadata={"chunk_id": "B"}
    )

    doc_c = Document(
        page_content="IT support policy",
        metadata={"chunk_id": "C"}
    )

    #create retriever without running its expensive constructor

    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    #Mock FAISS dense retrieval

    class FakeVectorStore:

        def similarity_search(self, query, k):
            return [doc_a, doc_b]

    #mock BM25 sparse retrieval

    class FakeBM25:
        def get_scores(self, query_tokens):
            return [1.0, 3.0, 2.0]

    retriever.vector_store = FakeVectorStore()
    retriever.bm25 = FakeBM25()
    retriever.documents = [doc_a, doc_b, doc_c]
    retriever.reranker = None

    #control configuration for deterministic testing

    monkeypatch.setattr(
        retriever_module, "DENSE_TOP_K", 2
    )

    monkeypatch.setattr(
        retriever_module, "SPARSE_TOP_K", 3
    )

    monkeypatch.setattr(
        retriever_module, "RRF_C", 60
    )

    monkeypatch.setattr(
        retriever_module, "RRF_DENSE_WEIGHT", 0.5
    )

    monkeypatch.setattr(
        retriever_module, "RRF_SPARSE_WEIGHT", 0.5
    )

    result = retriever.retrieve(
        query="company policy",
        final_k = 3
    )

    #B ranks first because it ranks first in BM25
    # and second in dense retrieval

    assert [
        doc.metadata["chunk_id"] for doc in result
    ] == ["B", "A", "C"]


def test_retrieve_respects_final_k(monkeypatch):
    doc_a = Document(
        page_content="Document A",
        metadata ={"chunk_id":"A"}
    )

    doc_b = Document(
        page_content="Document B",
        metadata={"chunk_id":"B"}
    )

    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    class FakeVectorStore:

        def similarity_search(self, query, k):
            return [doc_a, doc_b]

    class FakeBM25:

        def get_scores(self, query_token):
            return [2.0, 1.0]


    retriever.vector_store = FakeVectorStore()
    retriever.bm25 = FakeBM25()
    retriever.documents = [doc_a, doc_b]
    retriever.reranker = None

    result = retriever.retrieve(
        query="test query",
        final_k = 1
    )

    assert len(result) ==1
    assert result[0].metadata["chunk_id"] == "A"


def test_retrieve_propogates_retrieval_errors():
    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    class FailingVectorStore:

        def similarity_search(self, query, k):
            raise RuntimeError("FAISS retrieval failed")

    retriever.vector_store = FailingVectorStore()

    with pytest.raises(
        RuntimeError,
        match="FAISS retrieval failed"
    ):
        retriever.retrieve("test query")


def test_sample_document_fixture(sample_document):

    assert sample_document.page_content == "Company leave policy"

    assert sample_document.metadata["chunk_id"] == "chunk-101"

    assert sample_document.metadata["source"] == "company_policy.md"



# --------------------------------
# 4. Input validation tests
# --------------------------------

@pytest.mark.parametrize(
    "query",
    ["", "   ", "\n", None, 123]
)
def test_retrieve_rejects_invalid_query(query):
    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string"
    ):
        retriever.retrieve(query)


@pytest.mark.parametrize(
    "final_k",
    [0, -1, -10, True, 1.5, None]
)
def test_retrieve_rejects_invalid_final_k(final_k):
    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    with pytest.raises(
        ValueError,
        match="final_k must be a positive integer"
    ):
        retriever.retrieve("company policy", final_k=final_k)



# --------------------------------
# 5. Dense retrieval edge cases
# --------------------------------

def test_retrieve_returns_sparse_results_when_dense_is_empty(
    monkeypatch
):
    doc_a = Document(
        page_content="Company leave policy",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="Employee salary policy",
        metadata={"chunk_id": "B"}
    )

    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    class FakeVectorStore:
        def similarity_search(self, query, k):
            return []

    class FakeBM25:
        def get_scores(self, query_tokens):
            return [3.0, 1.0]

    retriever.vector_store = FakeVectorStore()
    retriever.bm25 = FakeBM25()
    retriever.documents = [doc_a, doc_b]
    retriever.reranker = None

    monkeypatch.setattr(
        retriever_module, "SPARSE_TOP_K", 2
    )

    result = retriever.retrieve(
        "company policy",
        final_k=2
    )

    assert len(result) == 2

    assert [
        doc.metadata["chunk_id"] for doc in result
    ] == ["A", "B"]



# --------------------------------
# 6. RRF duplicate handling
# --------------------------------

def test_retrieve_deduplicates_documents_across_retrievers(
    monkeypatch
):
    doc_a = Document(
        page_content="Company leave policy",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="Employee salary policy",
        metadata={"chunk_id": "B"}
    )

    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    class FakeVectorStore:
        def similarity_search(self, query, k):
            return [doc_a, doc_b]

    class FakeBM25:
        def get_scores(self, query_tokens):
            return [1.0, 2.0]

    retriever.vector_store = FakeVectorStore()
    retriever.bm25 = FakeBM25()
    retriever.documents = [doc_a, doc_b]
    retriever.reranker = None

    monkeypatch.setattr(
        retriever_module, "DENSE_TOP_K", 2
    )

    monkeypatch.setattr(
        retriever_module, "SPARSE_TOP_K", 2
    )

    monkeypatch.setattr(
        retriever_module, "RRF_C", 60
    )

    monkeypatch.setattr(
        retriever_module, "RRF_DENSE_WEIGHT", 0.5
    )

    monkeypatch.setattr(
        retriever_module, "RRF_SPARSE_WEIGHT", 0.5
    )

    result = retriever.retrieve(
        "company policy",
        final_k=2
    )

    assert len(result) == 2

    assert {
        doc.metadata["chunk_id"] for doc in result
    } == {"A", "B"}


# --------------------------------
# 7. CrossEncoder reranking tests
# --------------------------------

def test_retrieve_applies_cross_encoder_reranking(monkeypatch):

    doc_a = Document(
        page_content="Company leave policy",
        metadata={"chunk_id": "A"}
    )

    doc_b = Document(
        page_content="Employee salary policy",
        metadata={"chunk_id": "B"}
    )

    retriever = object.__new__(
        retriever_module.HybridRetriever
    )

    class FakeVectorStore:
        def similarity_search(self, query, k):
            return [doc_a, doc_b]

    class FakeBM25:
        def get_scores(self, query_tokens):
            return [2.0, 1.0]

    class FakeReranker:
        def predict(self, pairs):
            assert len(pairs) == 2

            return [0.2, 0.9]

    retriever.vector_store = FakeVectorStore()
    retriever.bm25 = FakeBM25()
    retriever.documents = [doc_a, doc_b]
    retriever.reranker = FakeReranker()

    result = retriever.retrieve(
        "company policy",
        final_k=2
    )

    assert [
        doc.metadata["chunk_id"] for doc in result
    ] == ["B", "A"]


# --------------------------------
# 8. Constructor initialization tests
# --------------------------------


def test_constructor_initializes_successfully(constructor_setup):

    retriever = retriever_module.HybridRetriever()

    assert retriever.embeddings is not None

    assert retriever.vector_store == "FAISS_INDEX_LOADED"

    assert len(retriever.documents) == 1

    assert retriever.documents[0].metadata["chunk_id"] == "chunk-101"

    assert retriever.bm25 is not None

    assert retriever.reranker is None


def test_constructor_raises_error_when_faiss_directory_missing(
    constructor_setup,
    monkeypatch
):

    missing_dir = constructor_setup["faiss_dir"].parent / "missing_index"

    monkeypatch.setattr(
        retriever_module,
        "FAISS_DIR",
        missing_dir
    )

    with pytest.raises(
        RuntimeError,
        match="Index not found"
    ):
        retriever_module.HybridRetriever()


def test_constructor_raises_error_when_metadata_file_missing(
    constructor_setup,
    monkeypatch
):

    missing_file = constructor_setup["chunks_file"].parent / "missing.jsonl"

    monkeypatch.setattr(
        retriever_module,
        "CHUNKS_FILE",
        missing_file
    )

    with pytest.raises(
        RuntimeError,
        match="Index not found"
    ):
        retriever_module.HybridRetriever()


def test_constructor_raises_error_when_metadata_is_empty(
    constructor_setup
):

    constructor_setup["chunks_file"].write_text(
        "",
        encoding="utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match="No documents found"
    ):
        retriever_module.HybridRetriever()


def test_constructor_propagates_invalid_json_error(
    constructor_setup
):

    constructor_setup["chunks_file"].write_text(
        "{invalid json}\n",
        encoding="utf-8"
    )

    with pytest.raises(
        ValueError
    ):
        retriever_module.HybridRetriever()


def test_constructor_propagates_embedding_initialization_error(
    constructor_setup,
    monkeypatch
):

    def failing_embeddings(*args, **kwargs):
        raise RuntimeError("Embedding initialization failed")

    monkeypatch.setattr(
        retriever_module,
        "HuggingFaceEmbeddings",
        failing_embeddings
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding initialization failed"
    ):
        retriever_module.HybridRetriever()


# --------------------------------
# 9. CrossEncoder constructor tests
# --------------------------------


def test_constructor_initializes_cross_encoder(
    constructor_setup,
    monkeypatch
):
    class FakeCrossEncoder:

        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(
        retriever_module,
        "ENABLE_RERANKER",
        True
    )

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        FakeCrossEncoder
    )

    retriever = retriever_module.HybridRetriever()

    assert retriever.reranker is not None

    assert retriever.reranker.model_name == retriever_module.RERANKER_MODEL


def test_constructor_propagates_cross_encoder_initialization_error(
    constructor_setup,
    monkeypatch
):
    def failing_cross_encoder(model_name):
        raise RuntimeError("CrossEncoder initialization failed")

    monkeypatch.setattr(
        retriever_module,
        "ENABLE_RERANKER",
        True
    )

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        failing_cross_encoder
    )

    with pytest.raises(
        RuntimeError,
        match="CrossEncoder initialization failed"
    ):
        retriever_module.HybridRetriever()



