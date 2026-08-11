import os
import glob
from typing import TypedDict, Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

llm = ChatOllama(model="llama3.1:latest", temperature=0.2, base_url='http://localhost:11434')

data_path = os.path.join(os.path.dirname(__file__), "..", "module01_raw", "1.10_rag_basic", "data", "*.txt")
texts = []
for fp in glob.glob(data_path):
    with open(fp, "r", encoding="utf-8") as f:
        texts.append(f.read())

splits = [s for t in texts for s in RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text(t)]
retriever = FAISS.from_texts(splits, OllamaEmbeddings(model="nomic-embed-text")).as_retriever(search_kwargs={"k": 4})

answer_prompt = PromptTemplate.from_template(
    "Use the context to answer the question. If unknown, say so.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
grade_prompt = PromptTemplate.from_template(
    "Is this document relevant to the question?\nDocument: {doc}\nQuestion: {question}\nAnswer with only 'yes' or 'no':"
)
need_retrieval_prompt = PromptTemplate.from_template(
    "Does answering this question require looking up specific facts?\nQuestion: {question}\nAnswer with only 'yes' or 'no':"
)
route_prompt = PromptTemplate.from_template(
    "Classify the question into one of: 'factual', 'general', 'websearch'.\n"
    "factual = answerable from a local knowledge base about AI agents.\n"
    "general = common knowledge, no lookup needed.\n"
    "websearch = requires current or live data.\n"
    "Question: {question}\nAnswer with only one word:"
)

parser = StrOutputParser()


print("=" * 60)
print("PATTERN 1 : Self-RAG — agent decides whether to retrieve")
print("=" * 60)

class SelfRAGState(TypedDict):
    question: str
    context: str
    answer: str

def decide_retrieval(state: SelfRAGState) -> Literal["retrieve", "direct_answer"]:
    decision = parser.invoke(llm.invoke(need_retrieval_prompt.format(question=state["question"]))).strip().lower()
    print(f"  [Self-RAG] Retrieve needed: {decision}")
    return "retrieve" if decision == "yes" else "direct_answer"

def retrieve_node(state: SelfRAGState) -> SelfRAGState:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}

def answer_node(state: SelfRAGState) -> SelfRAGState:
    ctx = state.get("context", "No context.")
    prompt = answer_prompt.format(context=ctx, question=state["question"])
    return {"answer": parser.invoke(llm.invoke(prompt))}

self_rag = StateGraph(SelfRAGState)
self_rag.add_node("retrieve", retrieve_node)
self_rag.add_node("answer", answer_node)
self_rag.add_conditional_edges(START, decide_retrieval, {"retrieve": "retrieve", "direct_answer": "answer"})
self_rag.add_edge("retrieve", "answer")
self_rag.add_edge("answer", END)
self_rag_graph = self_rag.compile()

q1 = "How do tools and memory interplay in agents?"
q2 = "What is 2 + 2?"
print(f"Q (factual): {q1}")
print("A:", self_rag_graph.invoke({"question": q1})["answer"])
print(f"\nQ (general): {q2}")
print("A:", self_rag_graph.invoke({"question": q2})["answer"])


print("\n" + "=" * 60)
print("PATTERN 2 : Corrective RAG — grades retrieved docs, reformulates if poor")
print("=" * 60)

class CRAGState(TypedDict):
    question: str
    docs: list
    context: str
    answer: str
    attempts: int

def crag_retrieve(state: CRAGState) -> CRAGState:
    docs = retriever.invoke(state["question"])
    return {"docs": docs, "attempts": state.get("attempts", 0) + 1}

def grade_docs(state: CRAGState) -> Literal["use_docs", "reformulate"]:
    relevant = []
    for doc in state["docs"]:
        grade = parser.invoke(llm.invoke(grade_prompt.format(doc=doc.page_content[:300], question=state["question"]))).strip().lower()
        if grade == "yes":
            relevant.append(doc)
    print(f"  [CRAG] Relevant docs: {len(relevant)}/{len(state['docs'])}")
    if relevant or state.get("attempts", 0) >= 2:
        return "use_docs"
    return "reformulate"

def reformulate(state: CRAGState) -> CRAGState:
    new_q = parser.invoke(llm.invoke(f"Rewrite this question to be more specific for retrieval: {state['question']}"))
    print(f"  [CRAG] Reformulated question: {new_q.strip()}")
    return {"question": new_q.strip()}

def crag_answer(state: CRAGState) -> CRAGState:
    ctx = "\n\n".join(d.page_content for d in state["docs"])
    prompt = answer_prompt.format(context=ctx, question=state["question"])
    return {"answer": parser.invoke(llm.invoke(prompt))}

crag = StateGraph(CRAGState)
crag.add_node("retrieve", crag_retrieve)
crag.add_node("reformulate", reformulate)
crag.add_node("answer", crag_answer)
crag.add_edge(START, "retrieve")
crag.add_conditional_edges("retrieve", grade_docs, {"use_docs": "answer", "reformulate": "reformulate"})
crag.add_edge("reformulate", "retrieve")
crag.add_edge("answer", END)
crag_graph = crag.compile()

q3 = "What persistence strategies exist for long-running agents?"
print(f"Q: {q3}")
print("A:", crag_graph.invoke({"question": q3, "docs": [], "attempts": 0})["answer"])


print("\n" + "=" * 60)
print("PATTERN 3 : Adaptive RAG — routes between no-retrieval, RAG, and web search")
print("=" * 60)

class AdaptiveState(TypedDict):
    question: str
    route: str
    context: str
    answer: str

def route_question(state: AdaptiveState) -> Literal["factual", "general", "websearch"]:
    r = parser.invoke(llm.invoke(route_prompt.format(question=state["question"]))).strip().lower()
    print(f"  [Adaptive] Route: {r}")
    if r not in ("factual", "general", "websearch"):
        r = "general"
    return r

def adaptive_retrieve(state: AdaptiveState) -> AdaptiveState:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}

def direct_answer(state: AdaptiveState) -> AdaptiveState:
    return {"answer": parser.invoke(llm.invoke(state["question"]))}

def web_search_stub(state: AdaptiveState) -> AdaptiveState:
    print("  [Adaptive] Web search would run here (Tavily/SerpAPI). Returning stub.")
    return {"context": f"[Live web results for: {state['question']}]"}

def adaptive_answer(state: AdaptiveState) -> AdaptiveState:
    ctx = state.get("context", "")
    prompt = answer_prompt.format(context=ctx, question=state["question"])
    return {"answer": parser.invoke(llm.invoke(prompt))}

adaptive = StateGraph(AdaptiveState)
adaptive.add_node("retrieve", adaptive_retrieve)
adaptive.add_node("direct_answer", direct_answer)
adaptive.add_node("web_search", web_search_stub)
adaptive.add_node("rag_answer", adaptive_answer)
adaptive.add_conditional_edges(
    START,
    route_question,
    {"factual": "retrieve", "general": "direct_answer", "websearch": "web_search"}
)
adaptive.add_edge("retrieve", "rag_answer")
adaptive.add_edge("web_search", "rag_answer")
adaptive.add_edge("direct_answer", END)
adaptive.add_edge("rag_answer", END)
adaptive_graph = adaptive.compile()

for question in [
    "How do tools and memory interplay in agents?",
    "What is the capital of France?",
    "What are the latest AI model releases this week?",
]:
    print(f"\nQ: {question}")
    result = adaptive_graph.invoke({"question": question})
    print("A:", result["answer"])
