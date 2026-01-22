# rag_lab

A lightweight retrieval-augmented generation (RAG) lab that ingests local files into a Chroma vector store and answers questions using Ollama-compatible chat and embedding endpoints.

## What it does

- **Ingests files** from `data/` into a persistent Chroma DB in `db/`.
- **Embeds content** via Ollama-compatible embeddings endpoints.
- **Chats with context** by retrieving top-k relevant chunks and sending them to a chat model.

## Project layout

- `ingest.py` — loads files in `data/`, chunks text, embeds, and stores vectors.
- `query.py` — interactive CLI to ask questions against the vector DB.
- `rag_core.py` — shared utilities (chunking, embeddings, chat, context formatting).
- `data/` — place your documents here.
- `db/` — auto-created persistent Chroma DB directory.

## Requirements

- Python 3.10+ (recommended)
- Ollama (or any server exposing compatible endpoints)
- See `requirements.txt` for Python dependencies

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Add documents

Put your files in `data/`. Supported extensions include:

- `.txt`, `.md`, `.json`, `.py`, `.php`, `.js`, `.html`, `.css`
- `.pdf`, `.docx`

### 2) Ingest

```bash
python ingest.py
```

This will:

- Scan `data/` for supported files
- Chunk text (default 900 chars with 150 overlap)
- Embed and persist vectors in `db/`
- Write a run log to `ingest_log.txt`

### 3) Query

```bash
python query.py
```

Ask questions in the CLI. Type `exit`, `quit`, or `q` to leave.

## Configuration

All configuration is via environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for Ollama-compatible server |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `RAG_CHAT_MODEL` | `gemma3:latest` | Chat model name |
| `RAG_TOP_K` | `6` | Number of chunks retrieved per query |
| `PYPDF_FAIL_FRACTION` | `0.10` | Fail threshold before forcing pdfminer |
| `PYPDF_FAIL_ABS` | `5` | Absolute fail threshold before forcing pdfminer |
| `PDF_MIN_TEXT_LEN` | `1200` | Minimum extracted text length before fallback |

Example:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export RAG_EMBED_MODEL="nomic-embed-text"
export RAG_CHAT_MODEL="gemma3:latest"
```

## Notes & troubleshooting

- If embeddings work but chat fails, your server may only expose embeddings endpoints or restrict chat routes.
- PDFs are parsed with `pypdf` first, then `pdfminer.six` if extraction looks weak.
- If you re-run ingest, previously ingested files are skipped based on `source` metadata.

## License

See `LICENSE`.
