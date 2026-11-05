# Enterprise AI Knowledge Assistant
A learning-first, production-style starter project combining:
- Hybrid RAG: FAISS dense retrieval + BM25 sparse retrieval + Reciprocal Rank Fusion (RRF)
- Conversational RAG: history-aware routing and follow-up questions
- LangGraph: explicit state graph and conditional routing
- Function calling: calculator and demo weather tools
- Contextual compression: LLM extracts query-relevant evidence before answer generation
- Streamlit chat UI and CLI

## 1. Architecture

### Offline ingestion
```text
data/*.md, *.txt, *.pdf
        |
Document loading -> chunking -> embeddings
        |                         |
   chunks.jsonl                FAISS index
        |
   BM25 index is built in memory at app startup
```

### Online request
```text
User message + thread_id
        |
LangGraph router (uses recent conversation)
        |
  +-----+-----------+
  |                 |
RAG route         Tool route / General chat
  |                 |
FAISS + BM25       LLM bound to tools
  |                 |
RRF               ToolNode executes tools
  |                 |
Optional rerank    Tool result -> LLM
  |
LLM compression
  |
Answer with source filenames
  |
LangGraph checkpointer saves thread state
```

## 2. Requirements
- Python 3.11 recommended
- A Groq API key for the chat model
- Internet access for the first Hugging Face embedding-model download
- About 1-2 GB free disk space is recommended for Python packages and model cache

The demo weather tool returns **mock data**, not live weather.

## 3. Setup (Windows PowerShell)

```powershell
cd enterprise_ai_knowledge_assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_key_here
```

Never commit `.env` or expose API keys in screenshots or GitHub.

## 4. Build the knowledge index

```powershell
python -m src.ingestion
```

This reads the sample files under `data/`, splits them into chunks, creates embeddings, saves the FAISS index under `artifacts/faiss_index/`, and writes chunk text/metadata to `artifacts/chunks.jsonl`.

Add your own `.pdf`, `.txt`, or `.md` files to `data/` and rerun ingestion whenever source documents change.

## 5. Run the application

### Streamlit UI
```powershell
streamlit run app.py
```

### CLI
```powershell
python -m src.cli
```

Try these questions in the UI:
1. `How many annual leaves do employees receive?`
2. `What about sick leaves?`  (tests conversational follow-up)
3. `What is the notice period?`
4. `What does ERR-504 mean?`
5. `What is the weather in Pune?` (uses demo tool)
6. `Calculate 125 * 8 + 40` (uses calculator tool)
7. `Hello, how are you?` (general chat)

Use the same Thread ID to continue a conversation. Change the Thread ID to start a separate conversation.

## 6. Project structure

```text
enterprise_ai_knowledge_assistant/
├── app.py                    # Streamlit chat interface
├── requirements.txt
├── .env.example
├── data/                     # Put source documents here
├── artifacts/                # Generated FAISS index and chunks
└── src/
    ├── config.py              # Environment configuration
    ├── ingestion.py           # Load -> split -> embed -> index
    ├── hybrid_retriever.py    # Dense + BM25 + RRF
    ├── tools.py               # Function-calling tools
    ├── graph.py               # LangGraph orchestration + memory
    └── cli.py                 # Terminal interface
```

## 7. Understand each component

### Hybrid retrieval
- Dense retrieval: FAISS searches embeddings for semantic similarity.
- Sparse retrieval: BM25 searches exact terms such as `ERR-504`.
- Query rewriting: uses recent thread messages to turn follow-ups into standalone retrieval queries.\n- RRF: combines the two ranked lists using `1 / (c + rank)`.
- The two retrieval systems use the same source chunks, but different search representations.

### Conversational memory
LangGraph's checkpointer stores graph state by `thread_id`. This demo uses `InMemorySaver`, so state is lost when the Python process stops. It is for local learning only. For persistent production memory, use a database-backed checkpointer such as PostgreSQL.

This is separate from a Transformer KV cache. KV cache is an inference optimization; it is not the application's durable chat-history database.

### Function calling
The model receives tool schemas. It can request `get_weather_demo` or `calculate`. LangGraph's `ToolNode` executes the selected Python function, returns the tool result to the model, and the model writes a user-facing response. The weather function is deliberately mocked.

### Contextual compression
After retrieval, an LLM extracts only the parts relevant to the current question. This can reduce prompt size, but it is another model call and may omit evidence. In production, evaluate compression faithfulness and retain source metadata.

### Reranking
An optional cross-encoder can rerank RRF candidates. Set `ENABLE_RERANKER=true` in `.env`. It downloads a separate model and increases CPU/RAM use. Start with it disabled, establish a baseline, then compare NDCG@K / Recall@K.

## 8. Important production upgrades

This is an educational starter, not a secure production deployment. Before production:
1. Replace `InMemorySaver` with a PostgreSQL checkpointer.
2. Add authentication and enforce tenant/user-level document permissions before retrieval.
3. Replace the mock weather tool with a real API and validate tool inputs.
4. Add timeouts, retries, rate limits, tracing, structured logs and exception handling.
5. Add a labelled retrieval evaluation set; measure Recall@K, Precision@K, MRR and NDCG.
6. Evaluate answer faithfulness, citation correctness and refusal behavior.
7. Add document versioning, deletion/re-indexing and access-control metadata.
8. Store secrets in a secret manager, not `.env`, in deployed environments.
9. Use persistent vector storage and backups.
10. Add human approval for tools that create side effects (payments, emails, account changes).

## 9. Learning exercises

1. Change `k_dense` and `k_sparse`. Observe retrieved chunks.
2. Change RRF weights. Test exact identifier queries versus paraphrased queries.
3. Disable BM25 and compare results with dense-only retrieval.
4. Disable dense retrieval and compare with BM25-only retrieval.
5. Enable the cross-encoder and compare the final ordering.
6. Disable compression and compare context length and answer quality.
7. Ask `What about sick leaves?` after asking about annual leave. Inspect the LangGraph state and router behavior.
8. Use two different thread IDs. Confirm that their histories do not mix.
9. Add a new policy document and rebuild the index.
10. Add a new tool with `@tool`, bind it to the model, and test the tool loop.

## 10. Troubleshooting

- `GROQ_API_KEY` missing: check `.env` and restart Streamlit.
- Index not found: run `python -m src.ingestion`.
- Hugging Face download fails: check internet access and retry; the model is cached after the first download.
- FAISS install issues on Windows: use Python 3.11 in a clean venv and install from `requirements.txt`.
- Model/tool incompatibility: choose a Groq model that supports tool calling and structured output, then set `GROQ_MODEL` in `.env`.
- Memory seems lost: `InMemorySaver` is process-local; do not stop/restart the app if you want to preserve the demo thread.
