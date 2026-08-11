
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import glob
import os

def load_docs():
    docs = []
    for fp in glob.glob(os.path.join(os.path.dirname(__file__), "..","module01_raw", "1.10_rag_basic" ,"data", "*.txt")):
        with open(fp, "r", encoding="utf-8") as f:
            docs.append(f.read())
    print(docs)
    return docs

texts = load_docs()
# Updated: Use new langchain_text_splitters package
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
splits = [s for t in texts for s in splitter.split_text(t)]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
print(f"Generated embeddings for {len(splits)} text chunks.")
vectordb = FAISS.from_texts(splits, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

def rag_chain(query):
    template = """Use the context to answer the question. If unknown, say so.
Context:
{context}

Question: {question}
Answer:"""
    # Updated: Use PromptTemplate from langchain_core.prompts
    prompt = PromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.1:latest", temperature=0.2)

    def format_docs(docs):
        return "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
    
    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)
    
    # Step 2: Format the retrieved documents
    context = format_docs(retrieved_docs)
    
    # Step 3: Create the prompt with context and question
    prompt_input = prompt.format(context=context, question=query)
    
    # Step 4: Get response from LLM
    response = llm.invoke(prompt_input)
    
    # Step 5: Parse and print output
    output_parser = StrOutputParser()
    result = output_parser.invoke(response)

    return result
    

if __name__ == "__main__":
    query_text = "How do tools and memory interplay in agents?"
    print(f"Query: {query_text}")
    print("\n" + "="*60 + "\n")
    answer = rag_chain(query_text)
    print("Answer:")
    print(answer)
    print("\n" + "="*60 + "\n")

    query_text = "What is Sachin's full name?"
    print(f"Query: {query_text}")
    print("\n" + "="*60 + "\n")
    answer = rag_chain(query_text)
    print("Answer:")
    print(answer)
    print("\n" + "="*60 + "\n")

    query_text = "Tell me about history of Pune"
    print(f"Query: {query_text}")
    print("\n" + "="*60 + "\n")
    answer = rag_chain(query_text)
    print("Answer:")
    print(answer)
    print("\n" + "="*60 + "\n")

    query_text = "If I go on heritage walk in Pune with Sachin, which places he would show me?"
    print(f"Query: {query_text}")
    print("\n" + "="*60 + "\n")
    answer = rag_chain(query_text)
    print("Answer:")
    print(answer)
    print("\n" + "="*60 + "\n")