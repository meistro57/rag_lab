# rag_core.py
import os
import re
import requests
from typing import List, Dict, Any, Tuple, Optional

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def clean_text(s: str) -> str:
    s = s.replace("\x00", "")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ollama_embed(texts: List[str], embed_model: str) -> List[List[float]]:
    """
    Prefer Ollama-native embeddings endpoint: POST /api/embeddings
    Fallback to OpenAI-compatible embeddings endpoint: POST /v1/embeddings
    """
    vectors = []
    for t in texts:
        # 1) Ollama native
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": embed_model, "prompt": t},
            timeout=120,
        )
        if r.status_code == 404:
            # 2) OpenAI compatible
            r = requests.post(
                f"{OLLAMA_BASE_URL}/v1/embeddings",
                json={"model": embed_model, "input": t},
                timeout=120,
            )
        r.raise_for_status()
        data = r.json()

        # Ollama-native shape: {"embedding":[...]}
        if "embedding" in data:
            vectors.append(data["embedding"])
            continue

        # OpenAI-compatible: {"data":[{"embedding":[...]}], ...}
        if "data" in data and data["data"] and "embedding" in data["data"][0]:
            vectors.append(data["data"][0]["embedding"])
            continue

        raise RuntimeError(f"Unexpected embeddings response shape: {data}")
    return vectors


def _try_api_chat(model: str, messages: List[Dict[str, str]], temperature: float) -> Optional[str]:
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "options": {"temperature": temperature}, "stream": False},
            timeout=300,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
        return None
    except requests.RequestException:
        return None


def _try_api_generate(model: str, prompt: str, temperature: float) -> Optional[str]:
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "options": {"temperature": temperature}, "stream": False},
            timeout=300,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if "response" in data:
            return data["response"]
        return None
    except requests.RequestException:
        return None


def _try_v1_chat_completions(model: str, messages: List[Dict[str, str]], temperature: float) -> Optional[str]:
    """
    OpenAI-compatible endpoint: POST /v1/chat/completions
    Ollama exposes this on many builds.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature, "stream": False},
            timeout=300,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        # Expected: choices[0].message.content
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            if "content" in msg:
                return msg["content"]
        return None
    except requests.RequestException:
        return None


def _try_v1_completions(model: str, prompt: str, temperature: float) -> Optional[str]:
    """
    OpenAI-compatible endpoint: POST /v1/completions
    Some servers expose this.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/v1/completions",
            json={"model": model, "prompt": prompt, "temperature": temperature, "stream": False},
            timeout=300,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        # Expected: choices[0].text
        if "choices" in data and data["choices"]:
            if "text" in data["choices"][0]:
                return data["choices"][0]["text"]
        return None
    except requests.RequestException:
        return None


def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    Try, in order:
    1) /api/chat
    2) /v1/chat/completions
    3) /api/generate (flatten messages)
    4) /v1/completions (flatten messages)
    """
    out = _try_api_chat(model, messages, temperature)
    if out is not None:
        return out

    out = _try_v1_chat_completions(model, messages, temperature)
    if out is not None:
        return out

    # Flatten for prompt-style endpoints
    prompt_lines = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        prompt_lines.append(f"{role}:\n{content}\n")
    prompt_lines.append("ASSISTANT:\n")
    prompt = "\n".join(prompt_lines)

    out = _try_api_generate(model, prompt, temperature)
    if out is not None:
        return out

    out = _try_v1_completions(model, prompt, temperature)
    if out is not None:
        return out

    raise RuntimeError(
        "Text generation endpoints not reachable on the server at OLLAMA_BASE_URL.\n"
        f"Base URL: {OLLAMA_BASE_URL}\n\n"
        "Tried:\n"
        " - POST /api/chat\n"
        " - POST /v1/chat/completions\n"
        " - POST /api/generate\n"
        " - POST /v1/completions\n\n"
        "But embeddings worked, so something is definitely listening.\n"
        "Most likely causes:\n"
        "1) You are not actually hitting the Ollama server (a proxy/service only exposing embeddings).\n"
        "2) Ollama is running with restricted routes.\n"
        "3) The server is OpenAI-compatible but not exposing completions for some reason.\n"
    )


def format_context(chunks: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = []
    for i, (txt, meta) in enumerate(chunks, 1):
        src = meta.get("source", "unknown")
        loc = meta.get("loc", "")
        lines.append(f"[{i}] SOURCE: {src} {loc}\n{txt}\n")
    return "\n".join(lines).strip()
