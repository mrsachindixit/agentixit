from typing import List
import os
from utils.ollama_client import chat as ollama_chat


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Use Ollama chat to produce a short summary; fallback to a heuristic if unavailable."""
    system = "You are a concise summarizer. Produce a short summary (3-5 sentences)."
    user = f"Summarize the following text:\n\n{text[:4000]}"
    try:
        return ollama_chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.2)
    except Exception:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return '. '.join(sentences[:max_sentences]) + ('.' if sentences else '')


def summarize_all(pdf_folder: str) -> dict:
    out = []
    if not os.path.exists(pdf_folder):
        return {"error": "folder not found", "path": pdf_folder}
    for fn in os.listdir(pdf_folder):
        if fn.lower().endswith(('.pdf', '.txt')):
            path = os.path.join(pdf_folder, fn)
            try:
                with open(path, 'rb') as f:
                    # for binary PDFs, we don't attempt to parse here; placeholder
                    text = ''
            except Exception:
                text = ''
            # read text for txt files
            if fn.lower().endswith('.txt'):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            summary = summarize_text(text or fn)
            out.append({"file": fn, "summary": summary})
    return {"summaries": out}
