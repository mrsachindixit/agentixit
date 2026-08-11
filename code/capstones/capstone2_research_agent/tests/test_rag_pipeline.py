import os
import tempfile
import json
from capstones.capstone2_research_agent import rag_pipeline


def test_build_index_and_answer(monkeypatch):
    # Create temporary data folder with a sample text
    with tempfile.TemporaryDirectory() as tmp:
        pdfs = os.path.join(tmp, 'pdfs')
        os.makedirs(pdfs, exist_ok=True)
        sample_path = os.path.join(pdfs, 's1.txt')
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write('This is a test document about NER methods. Transfer learning and data augmentation are discussed.')

        # Patch embedding fn to simple deterministic vectors
        def fake_embed(texts):
            return [[float(len(t))] for t in texts]

        # Patch chat function to return a canned answer
        def fake_chat(messages, **kwargs):
            return 'CANNED_ANSWER'

        persist = os.path.join(tmp, 'chroma')
        # Build index
        vectordb = rag_pipeline.build_index_from_folder(pdfs, persist, embedding_fn=fake_embed)
        assert os.path.exists(persist)

        retriever = rag_pipeline.load_retriever(persist, embedding_fn=fake_embed)
        ans = rag_pipeline.answer_question(retriever, 'What does the document talk about?', chat_fn=fake_chat)
        assert 'CANNED_ANSWER' in ans
