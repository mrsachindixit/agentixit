from typing import List, Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from utils.ollama_client import embed as ollama_embed
from tools.pdf_ingest import ingest_folder
from utils import ensure_dir
import os


def build_index_from_folder(pdf_folder: str, persist_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200,
                            embedding_fn: Callable = None):
    """Ingest files from folder, chunk, create embeddings (via embedding_fn) and persist Chroma index.

    embedding_fn: function that accepts list[str] and returns List[List[float]].
    If omitted, uses `ollama_embed`.
    """
    docs_meta = ingest_folder(pdf_folder)
    # collect texts
    raw_texts = []
    metadatas = []
    # ingest_folder returns dict with 'files'
    for item in docs_meta.get('files', []):
        path = item.get('path')
        text = item.get('text', '')
        raw_texts.append({'text': text, 'metadata': {'file': item.get('file'), 'path': path}})

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Document] = []
    for r in raw_texts:
        pieces = splitter.split_text(r['text'])
        for i, p in enumerate(pieces):
            docs.append(Document(page_content=p, metadata={**r['metadata'], 'chunk': i}))

    ensure_dir(persist_dir)

    emb = embedding_fn or ollama_embed
    vectordb = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


def load_retriever(persist_dir: str, embedding_fn: Callable = None, k: int = 4):
    emb = embedding_fn or ollama_embed
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=emb)
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k': k})
    return retriever


def answer_question(retriever, question: str, chat_fn: Callable, k: int = 4) -> str:
    """Retrieve top-k documents and ask `chat_fn` (Ollama chat wrapper) to answer using context."""
    docs = retriever.get_relevant_documents(question)
    context = '\n---\n'.join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
    system = "You are an assistant that answers questions using the provided document context. Cite sources when possible."
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely using the context and include citations."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat_fn(messages)
