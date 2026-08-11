import os
import tempfile
from capstones.capstone2_research_agent.tools import summarize


def test_summarize_all(monkeypatch):
    # create tmp pdf folder with a txt file
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'pdfs')
        os.makedirs(p, exist_ok=True)
        sample_path = os.path.join(p, 'doc.txt')
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write('Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.')

        # monkeypatch ollama chat to return a predictable summary
        def fake_chat(messages, **kwargs):
            return 'Short summary produced by Ollama.'

        monkeypatch.setattr('capstones.capstone2_research_agent.tools.summarize.ollama_chat', fake_chat)

        out = summarize.summarize_all(p)
        assert 'summaries' in out
        assert out['summaries'][0]['summary'] == 'Short summary produced by Ollama.'
