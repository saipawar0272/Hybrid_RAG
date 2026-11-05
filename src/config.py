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
