# ingest.py
import os
import glob
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from pypdf import PdfReader
from docx import Document

from rag_core import chunk_text, ollama_embed, clean_text

DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "mark_rag"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
LOG_PATH = "ingest_log.txt"

# If pypdf fails on >= this fraction of pages, we force pdfminer
PYPDF_FAIL_FRACTION = float(os.environ.get("PYPDF_FAIL_FRACTION", "0.10"))
# Or if it fails on >= this many pages, force pdfminer
PYPDF_FAIL_ABS = int(os.environ.get("PYPDF_FAIL_ABS", "5"))
# If extracted text is shorter than this, try pdfminer
MIN_TEXT_LEN = int(os.environ.get("PDF_MIN_TEXT_LEN", "1200"))


def log(msg: str) -> None:
    line = msg.strip()
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def read_docx(path: str) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n".join(parts))


def read_pdf_pypdf_safe(path: str) -> Tuple[str, int, int]:
    """
    Returns: (text, pages_total, pages_failed)
    Extract per page, skipping any broken pages.
    """
    reader = PdfReader(path)
    parts = []
    failed = 0
    total = len(reader.pages)

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = clean_text(text)
            if text:
                parts.append(f"\n\n[Page {i+1}]\n{text}")
        except Exception as e:
            failed += 1
            log(f"[WARN] pypdf failed on {path} page {i+1}: {type(e).__name__}: {e}")

    return "\n".join(parts).strip(), total, failed


def read_pdf_pdfminer(path: str) -> str:
    """
    Fallback PDF extraction using pdfminer.six.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path) or ""
        return clean_text(text)
    except Exception as e:
        log(f"[WARN] pdfminer failed on {path}: {type(e).__name__}: {e}")
        return ""


def read_pdf(path: str) -> str:
    """
    Strategy:
    - Try pypdf safely.
    - If pypdf fails often or yields too little, force pdfminer fallback.
    - Choose whichever yields more usable text.
    """
    text_pypdf, total_pages, failed_pages = read_pdf_pypdf_safe(path)
    fail_fraction = (failed_pages / total_pages) if total_pages else 0.0

    log(f"[INFO] PDF summary for {path}: pages={total_pages}, pypdf_failed={failed_pages} ({fail_fraction:.1%}), pypdf_text_len={len(text_pypdf)}")

    force_fallback = (
        failed_pages >= PYPDF_FAIL_ABS
        or fail_fraction >= PYPDF_FAIL_FRACTION
        or len(text_pypdf) < MIN_TEXT_LEN
    )

    if force_fallback:
        log(f"[INFO] Forcing pdfminer fallback for {path} (pypdf too broken/short).")
        text_pdfminer = read_pdf_pdfminer(path)

        # pick best
        if len(text_pdfminer) > len(text_pypdf):
            log(f"[INFO] Using pdfminer text for {path} (len={len(text_pdfminer)}).")
            return text_pdfminer
        else:
            log(f"[INFO] Keeping pypdf text for {path} (len={len(text_pypdf)}).")
            return text_pypdf

    log(f"[INFO] Using pypdf text for {path} (len={len(text_pypdf)}).")
    return text_pypdf


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".json", ".py", ".php", ".js", ".css", ".html"]:
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return ""


def main():
    # reset log each run
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    files: List[str] = []
    for ext in ["*.txt", "*.md", "*.pdf", "*.docx", "*.json", "*.py", "*.php", "*.js", "*.html", "*.css"]:
        files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    if not files:
        log(f"[INFO] No files found in {DATA_DIR}/. Add docs and rerun.")
        return

    existing = set()
    try:
        peek = collection.get(include=["metadatas"])
        for m in peek.get("metadatas", []):
            if m and "source" in m:
                existing.add(m["source"])
    except Exception:
        pass

    new_files = [f for f in files if f not in existing]

    if not new_files:
        log("[INFO] No new files to ingest (everything already in DB).")
        return

    log(f"[INFO] Ingesting {len(new_files)} new files using embed model: {EMBED_MODEL}")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for path in tqdm(new_files):
        try:
            text = load_file(path)
        except Exception as e:
            log(f"[ERROR] load_file crashed for {path}: {type(e).__name__}: {e}")
            continue

        if not text:
            log(f"[WARN] No extractable text from {path} (skipping).")
            continue

        chunks = chunk_text(text, chunk_size=900, overlap=150)
        if not chunks:
            log(f"[WARN] Chunking produced 0 chunks for {path} (skipping).")
            continue

        for idx, ch in enumerate(chunks):
            doc_id = f"{path}::chunk::{idx}"
            ids.append(doc_id)
            docs.append(ch)
            metas.append({"source": path, "loc": f"(chunk {idx})"})

    if not docs:
        log("[INFO] Nothing to embed. Done.")
        return

    BATCH = 32
    log(f"[INFO] Embedding {len(docs)} chunks...")
    for i in tqdm(range(0, len(docs), BATCH), desc="Embedding"):
        batch_docs = docs[i:i + BATCH]
        batch_ids = ids[i:i + BATCH]
        batch_metas = metas[i:i + BATCH]

        try:
            embeddings = ollama_embed(batch_docs, EMBED_MODEL)
        except Exception as e:
            log(f"[ERROR] Embedding batch failed at i={i}: {type(e).__name__}: {e}")
            continue

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    log("[INFO] Done. Vector DB updated.")
    log(f"[INFO] See {LOG_PATH} for warnings/errors.")


if __name__ == "__main__":
    main()
