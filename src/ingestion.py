"""Offline ingestion: load -> split -> embed -> save FAISS + chunk metadata."""
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    DATA_DIR, ARTIFACTS_DIR, FAISS_DIR, CHUNKS_FILE,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, validate_config
)


def load_documents():
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def build_index():
    validate_config()

    source_docs = load_documents()
    if not source_docs:
        raise RuntimeError(f"No .pdf, .txt or .md documents found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = splitter.split_documents(source_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{i}"
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(FAISS_DIR))

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(f"Loaded {len(source_docs)} source documents")
    print(f"Created {len(chunks)} chunks")
    print(f"FAISS index: {FAISS_DIR}")
    print(f"Chunk records: {CHUNKS_FILE}")


if __name__ == "__main__":
    build_index()
