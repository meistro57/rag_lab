# query.py
import os
import re
import chromadb
from chromadb.config import Settings

from rag_core import ollama_embed, ollama_chat, format_context

DB_DIR = "db"
COLLECTION_NAME = "mark_rag"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gemma3:latest")
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))

SYSTEM_PROMPT = """You are Eli: clear-minded, grounded, curious, and constructive.
If CONTEXT is provided, use it first.
If the answer is not in the context, say so and suggest what to add.
Be precise and practical. Avoid fluff.
When you use the context, cite chunk numbers like [1], [2], etc.
"""

# Questions that should NOT use RAG (avoid "Seth answers everything" syndrome)
NO_RAG_PATTERNS = [
    r"^\s*who are you\??\s*$",
    r"^\s*what are you\??\s*$",
    r"^\s*what can you do\??\s*$",
    r"^\s*help\??\s*$",
    r"^\s*how do you work\??\s*$",
    r"^\s*what model\??\s*$",
    r"^\s*what is rag\??\s*$",
]

def should_skip_rag(q: str) -> bool:
    q2 = q.strip().lower()
    return any(re.match(p, q2) for p in NO_RAG_PATTERNS)


def main():
    print(f"Using embed model: {EMBED_MODEL}")
    print(f"Using chat model:  {CHAT_MODEL}")
    print(f"Retrieving top_k:  {TOP_K}")

    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    while True:
        q = input("\nAsk> ").strip()
        if not q or q.lower() in ["exit", "quit", "q"]:
            break

        if should_skip_rag(q):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            answer = ollama_chat(CHAT_MODEL, messages, temperature=0.2)
            print("\n" + answer)
            continue

        q_vec = ollama_embed([q], EMBED_MODEL)[0]

        results = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        if not docs:
            print("No results retrieved from vector DB.")
            continue

        print("\nRetrieved:")
        for i, (m, dist) in enumerate(zip(metas, dists), 1):
            print(f"  [{i}] {m.get('source','?')} {m.get('loc','')}  (distance={dist:.4f})")

        retrieved = list(zip(docs, metas))
        context = format_context(retrieved)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"},
        ]

        answer = ollama_chat(CHAT_MODEL, messages, temperature=0.2)
        print("\n" + answer)


if __name__ == "__main__":
    main()
