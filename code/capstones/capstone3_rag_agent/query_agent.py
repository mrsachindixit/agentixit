import argparse
from langchain.vectorstores import Chroma
from utils import get_env
from utils.ollama_client import chat as ollama_chat, embed as ollama_embed
import os
from typing import Any


def _build_prompt(query: str, retrieved_texts: list, chat_history: list, citations: list[dict[str, Any]], abstain: bool = False):
    system = (
        "You are an assistant that answers questions using the provided document context. "
        "When helpful, cite the source path. Keep answers concise and factual."
    )
    if abstain:
        system += " If context is weak or missing, explicitly say you are unsure and ask for more documents."
    context = "\n---\n".join(retrieved_texts)
    history_block = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history]) if chat_history else ""
    user = (
        f"Context:\n{context}\n\n"
        f"Citations metadata:\n{citations}\n\n"
        f"History:\n{history_block}\n\n"
        f"User question: {query}\n\n"
        "Answer using the context above and include source paths when available."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages


def load_chain(persist_dir: str, model: str | None = None):
    # Chroma with Ollama embedding function
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=ollama_embed)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Simple memory as list of (user, assistant) tuples
    memory = []

    def ask(question: str):
        docs_with_score = []
        if hasattr(vectordb, "similarity_search_with_relevance_scores"):
            docs_with_score = vectordb.similarity_search_with_relevance_scores(question, k=4)
        else:
            docs = retriever.get_relevant_documents(question)
            docs_with_score = [(d, 0.5) for d in docs]

        texts = []
        citations = []
        top_score = 0.0
        for d, score in docs_with_score:
            text = d.page_content if hasattr(d, 'page_content') else str(d)
            meta = getattr(d, 'metadata', {}) or {}
            source = meta.get('source') or meta.get('path') or meta.get('file') or 'unknown_source'
            texts.append(text)
            citations.append({"source": source, "score": round(float(score), 4)})
            top_score = max(top_score, float(score))

        abstain = top_score < 0.2 or not texts
        messages = _build_prompt(question, texts, memory, citations, abstain=abstain)
        resp = ollama_chat(messages, model=model) if model else ollama_chat(messages)
        if abstain:
            resp = (
                "I am not confident I have enough grounded evidence from the retrieved documents. "
                "Please add more relevant PDFs or ask a narrower question.\n\n"
                f"Top retrieval score: {top_score:.3f}; citations: {citations}"
            )

        memory.append((question, resp))
        return resp

    return ask


def chat_loop(ask_fn):
    print("RAG conversational agent (Ollama) ready — ask questions, or type 'exit' to quit")
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if not q:
            continue
        if q.strip().lower() in ("exit", "quit"):
            break
        resp = ask_fn(q)
        print("Agent:\n", resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    args = parser.parse_args()

    chain = load_chain(args.persist_dir, args.model)
    chat_loop(chain)
