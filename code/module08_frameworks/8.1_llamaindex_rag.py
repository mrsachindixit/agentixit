import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

Settings.llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE, request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED, base_url=OLLAMA_BASE)
Settings.chunk_size = 512
Settings.chunk_overlap = 64

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "module01_raw", "1.10_rag_basic", "data"
)

def build_index():
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".txt"],
        recursive=False,
    ).load_data()
    print(f"Loaded {len(documents)} chunks")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index

def query_index(index, question: str) -> str:
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question)

    print("\nSources:")
    for node in response.source_nodes:
        score = f"{node.score:.3f}" if node.score else "n/a"
        snippet = node.text[:120].replace("\n", " ")
        print(f"  [{score}] {snippet}...")
    print()

    return str(response)

if __name__ == "__main__":
    print("=== LlamaIndex RAG ===\n")

    idx = build_index()

    questions = [
        "How do agents use tools and memory?",
        "What is Sachin's full name?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = query_index(idx, q)
        print(f"A: {answer}\n{'=' * 60}\n")
