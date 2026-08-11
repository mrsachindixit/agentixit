from typing import List
from pypdf import PdfReader
from langchain.schema import Document
import os


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            # some PDFs may raise on specific pages
            texts.append("")
    return "\n".join(texts)


def load_pdfs_as_documents(data_dir: str) -> List[Document]:
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                full = os.path.join(root, f)
                text = extract_text_from_pdf(full)
                if not text.strip():
                    continue
                metadata = {"source": full, "title": f}
                docs.append(Document(page_content=text, metadata=metadata))
    return docs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="Folder with PDFs")
    args = p.parse_args()
    docs = load_pdfs_as_documents(args.data_dir)
    print(f"Loaded {len(docs)} PDF documents from {args.data_dir}")
