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


from src.logging_config import get_logger

logger = get_logger(__name__)




def load_documents():
    docs = []

    logger.info("Starting document loading fromn %s", DATA_DIR)

    for path in sorted(DATA_DIR.rglob("*")):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:

            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()

            elif suffix in {".txt", ".md"}:
                loaded = TextLoader(
                    str(path),
                    encoding="utf-8"
                    ).load()
            else:
                logger.debug(
                    "Skipping unsupported file: %s",
                    path.name
                )
                continue

            for doc in loaded:
                doc.metadata["source"] = path.name

            docs.extend(loaded)

            logger.info(
                "Successfully loaded document: %s",
                path.name
            )

        except Exception:
            logger.exception(
                "Failed to load document: %s",
                path.name
            )
            raise

    logger.info(
        "Document loading completed. Total documents: %d",
        len(docs)
    )

    return docs


def build_index():
    """Build and persist the FAISS index and chunk metadat."""

    validate_config()

    logger.info("starting ingestion pipeline")

    try:

        #1. Load documents

        source_docs = load_documents()

        if not source_docs:
            raise RuntimeError(
                f"No .pdf, .txt or .md documents found in {DATA_DIR}"
                )

        logger.info(
            "Loaded %d source documents",
            len(source_docs)
        )

        #2. split documents into chunks
        logger.info(
            "Starting document chunking. Chunk size: %d, overlap: %d",
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )

        chunks = splitter.split_documents(source_docs)

        if not chunks:
            raise RuntimeError("Documents splitting produced no chunks.")

        #3. Add chunks metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk-{i}"
            chunk.metadata["source"] = chunk.metadata.get(
                "source",
                "unknown"
            )

        logger.info(
            "Document chunking completed. Total chunks: %d",
            len(chunks)
        )

        #4. initialize embedding model

        logger.info(
            "initialized embedding model: %s",
            EMBEDDING_MODEL
        )


        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )

        logger.info("Embedding model initialized successfully")

        #5. create artifact directory


        ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True
            )

        #6. Create FAISS vector databse
        logger.info("creating FAISS vector index")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
            )

        #7. save FAISS index
        vector_store.save_local(str(FAISS_DIR))

        logger.info(
            "FAISS index saved successfully at: %s",
            FAISS_DIR
        )

        #8. save chunk metadat at JSONL
        logger.info("Saving chunk metadata")


        with CHUNKS_FILE.open("w", encoding="utf-8") as f:

            for chunk in chunks:

                record = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str
                        ) + "\n"
                    )

        logger.info(
            "Chunk metadata saved successfully at: %s",
            CHUNKS_FILE
        )

        logger.info(
            "Ingestion pipeline completed sucessfully. "
            "Documents: %d | chunks : %d",
            len(source_docs),
            len(chunks)
        )

    except Exception:
        logger.exception("Ingestion pipeline failed")
        raise

if __name__ == "__main__":
    build_index()
