import os
import json
import glob
import sys

import requests
from utils.ollama_client import embed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Output index file path
OUT_FILE = "index.json"

def load_docs(path=os.path.join(os.path.dirname(__file__), "data", "*.txt")):
    docs = []
    for fp in glob.glob(path):
        with open(fp, "r", encoding="utf-8") as f:
            docs.append({"id": os.path.basename(fp), "text": f.read()})
    return docs

if __name__ == "__main__":
    # Load all .txt files from data/
    docs = load_docs()
    if not docs:
        print("No documents found in data/ directory.")
        sys.exit(1)
    
    # Embed all documents using the local model
    print(f"Embedding {len(docs)} documents...")
    vectors = []
    for d in docs:
        texts = d["text"]
        if isinstance(texts, str):
            texts = [texts]
        url = f"http://localhost:11434/api/embed"
        payload = {"model": "nomic-embed-text", "input": texts}
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        if "embeddings" in data:
            vectors.append(data["embeddings"])
        elif "data" in data:
            vectors.append([item["embedding"] for item in data["data"]])
        else:
            raise ValueError("Unexpected embeddings response format")


    # Build index with documents and their embeddings
    index = [{"id": d["id"], "text": d["text"], "vec": v} for d, v in zip(docs, vectors)]
    
    # Save index to JSON file
    out_file_path=os.path.join(os.path.dirname(__file__), "data", OUT_FILE)
    with open(out_file_path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    
    print(f"✓ Built index with {len(index)} documents -> {out_file_path}")
