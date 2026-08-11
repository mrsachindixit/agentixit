import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from ingest_pdfs import load_pdfs_as_documents
from utils import ensure_dir
from utils.ollama_client import embed as ollama_embed
import os


def build_index(data_dir: str, persist_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    print("Loading PDFs...")
    docs = load_pdfs_as_documents(data_dir)
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for i, p in enumerate(pieces):
            split_docs.append({"page_content": p, "metadata": {**d.metadata, "chunk": i}})

    print(f"Total chunks: {len(split_docs)}")

    ensure_dir(persist_dir)

    print("Creating/updating Chroma DB using Ollama embeddings...")
    # Chroma accepts an embedding function; use Ollama's embed wrapper
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in split_docs]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=ollama_embed,
        persist_directory=persist_dir,
    )
    vectordb.persist()
    print("Index built and persisted to", persist_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma index from PDFs")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    args = parser.parse_args()
    build_index(args.data_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)
