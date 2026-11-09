import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FAISS_DIR = ARTIFACTS_DIR / "faiss_index"
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.jsonl"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "8"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "8"))
RRF_C = int(os.getenv("RRF_C", "60"))
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.5"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.5"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


#---------------------

# Configuration validation
# ---------------------------

def validate_config():
    """validate apllication configuration at startup."""

    errors = []

    #1. chunking validation

    if CHUNK_SIZE <= 0 :
        errors.append("CHUNK_SIZE must be greater than 0")

    if CHUNK_OVERLAP < 0:
        errors.append("CHUNK_OVERLAP cannot be negative")


    if CHUNK_OVERLAP >= CHUNK_SIZE :
        errors.append("CHUNK-OVERLAP must be smaller than CHUNK-SIZE")


    # Retrieval validation ---

    if DENSE_TOP_K <= 0 :
        errors.append('DENSE_TOP_K must be greater than 0')

    if SPARSE_TOP_K <= 0 :
        errors.append('sparse top-k must be greater than 0')

    if FINAL_TOP_K <= 0:
        errors.append('final top-k must be gt than 0')


    #3. RRF validation

    if RRF_C <= 0 :
        errors.append('RRF_C must be gt than 0')

    if not 0 <= RRF_DENSE_WEIGHT <= 1 :
        errors.append('RRF_DENSE_WEIGHT must be in between 0 and 1')

    if not 0 <= RRF_SPARSE_WEIGHT <= 1 :
        errors.append('RRF_SPARSE_WEIGHT must be in between 0 and 1')

    if RRF_DENSE_WEIGHT + RRF_SPARSE_WEIGHT < 0:
        errors.append(' RRF value should not be negative')


    #4. Required configuration

    if not EMBEDDING_MODEL.strip() :
        errors.append("Embedding model cannot be empty")


    if not GROQ_MODEL.strip():
        errors.append('groq model cannot be empty')

    #report configuration erros

    if errors :
        raise ValueError(
            "Invalid application configuration\n- "
            + "\n- ".join(errors)
        )



