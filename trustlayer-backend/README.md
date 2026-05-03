# TrustLayer AI — Backend
> "Reliable AI, not just smart AI."

A production-ready, hallucination-aware RAG backend built with FastAPI, FAISS, and OpenAI.  
Answers **only** from trusted documents, verifies every response with a second LLM pass, and returns a structured confidence score.

---

## Quick Start

```bash
# 1. Clone & enter the project
git clone <your-repo>
cd trustlayer-backend

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...

# 5. Run the server
uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**  
Swagger UI at **http://localhost:8000/docs**

---

## Project Structure

```
trustlayer-backend/
│
├── app/
│   ├── main.py                 # FastAPI app, middleware, routers
│   │
│   ├── routes/
│   │   ├── upload.py           # POST /upload
│   │   └── query.py            # POST /ask
│   │
│   ├── services/
│   │   ├── rag_service.py      # FAISS store + retrieval
│   │   ├── llm_service.py      # Grounded answer generation
│   │   └── verifier_service.py # Hallucination detection
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── core/
│   │   └── config.py           # Settings loaded from .env
│   │
│   └── utils/
│       ├── embeddings.py       # OpenAI embedding wrapper
│       └── chunking.py         # Text extraction + chunking
│
├── data/                       # FAISS index + uploaded files (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Reference

### `POST /upload`

Upload a document to be indexed.

**Request:** multipart/form-data with `file` field (`.pdf`, `.txt`, `.md`)

**Response:**
```json
{
  "filename": "my_doc.pdf",
  "doc_id": "3f2a1b4c-...",
  "chunks": 24,
  "message": "Successfully indexed 'my_doc.pdf' into 24 chunks."
}
```

---

### `POST /ask`

Ask a question against indexed documents.

**Request:**
```json
{
  "question": "What are the data retention policies?",
  "mode": "trust"
}
```

| Field      | Type   | Default | Description                              |
|------------|--------|---------|------------------------------------------|
| `question` | string | —       | Natural-language question (3–2000 chars) |
| `mode`     | string | `trust` | `"trust"` = full pipeline, `"fast"` = skip verification |
| `top_k`    | int    | `null`  | Override default retrieval count (1–10)  |

**Response:**
```json
{
  "answer": "Data must be retained for a minimum of 90 days per §4.2.",
  "confidence": 0.9,
  "verdict": "SUPPORTED",
  "explanation": "The answer directly references §4.2 of compliance_2024.txt.",
  "sources": [
    {
      "chunk_id": "abc123_2",
      "doc_id": "abc123",
      "filename": "compliance_2024.txt",
      "text": "access logs retention minimum 90 days per regulatory requirement §4.2",
      "score": 0.14
    }
  ],
  "cached": false,
  "mode": "trust"
}
```

**Verdict → Confidence mapping:**

| Verdict               | Confidence | Meaning                                  |
|-----------------------|------------|------------------------------------------|
| `SUPPORTED`           | 0.9        | Answer fully grounded in context         |
| `PARTIALLY_SUPPORTED` | 0.6        | Answer partially grounded — use caution  |
| `NOT_SUPPORTED`       | 0.2        | Likely hallucination — answer withheld   |

---

### `GET /health`

```json
{
  "status": "ok",
  "faiss_loaded": true,
  "doc_count": 3
}
```

---

## Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  1. Retrieval (FAISS)           │  embed query → similarity search → top-k chunks
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  2. Answer Generation (LLM #1) │  strict grounding prompt — no prior knowledge
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3. Verification (LLM #2)      │  ← skipped in fast mode
│     SUPPORTED / PARTIAL / NOT  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  4. Trust Score Mapping        │  0.9 / 0.6 / 0.2
└────────────────┬────────────────┘
                 │
                 ▼
          Structured Response
```

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable           | Default                    | Description                       |
|--------------------|----------------------------|-----------------------------------|
| `OPENAI_API_KEY`   | *(required)*               | Your OpenAI API key               |
| `LLM_MODEL`        | `gpt-4o-mini`              | Chat model for generation+verify  |
| `EMBEDDING_MODEL`  | `text-embedding-3-small`   | Embedding model for FAISS         |
| `TOP_K_CHUNKS`     | `4`                        | Chunks retrieved per query        |
| `CHUNK_SIZE`       | `512`                      | Characters per chunk              |
| `CHUNK_OVERLAP`    | `64`                       | Overlap between chunks            |
| `ENABLE_CACHE`     | `true`                     | LRU cache for repeated queries    |
| `RATE_LIMIT_CALLS` | `30`                       | Requests allowed per window       |
| `RATE_LIMIT_PERIOD`| `60`                       | Rate limit window (seconds)       |

---

## Connecting to the TrustLayer Frontend

Update `app.js` in the frontend to point to this backend:

```js
async function classifyQuery(question, mode) {
  const res = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  return res.json();
  // Returns: { answer, confidence, verdict, explanation, sources }
}
```

---

## Production Notes

- Replace `allow_origins=["*"]` in `main.py` with your frontend's exact origin
- Use `gpt-4o` instead of `gpt-4o-mini` for higher accuracy
- Store your FAISS index on persistent disk (e.g. EFS on AWS) if deploying to containers
- Add authentication (API key header or OAuth) before exposing publicly
- Consider `gunicorn -k uvicorn.workers.UvicornWorker` for multi-worker production deploys