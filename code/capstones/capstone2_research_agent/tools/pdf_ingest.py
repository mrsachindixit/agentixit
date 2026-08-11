import os
from typing import List
from pypdf import PdfReader

def extract_text(path: str) -> str:
    if path.lower().endswith('.pdf'):
        reader = PdfReader(path)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n".join(texts)
    else:
        # support simple text files for example datasets
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


def ingest_folder(folder: str) -> dict:
    """Ingest all PDFs and TXT files in `folder` and return metadata list including extracted text."""
    out = []
    if not os.path.exists(folder):
        return {"error": "folder not found", "path": folder}
    for fn in os.listdir(folder):
        if fn.lower().endswith(('.pdf', '.txt')):
            path = os.path.join(folder, fn)
            txt = extract_text(path)
            out.append({"file": fn, "path": path, "length": len(txt), 'text': txt})
    return {"ingested": len(out), "files": out}
