import os, json, sys
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ollama_client import chat, embed

st.set_page_config(page_title='Agentic AI Playground', page_icon='robot', layout='wide')
mode = st.sidebar.radio('Mode', ['Chat', 'RAG Explorer'])

if mode=='Chat':
    st.title('Chat')
    sys_prompt = st.text_area('System prompt', 'You are a helpful, concise assistant.', height=120)
    user = st.text_input('Your question', 'Explain what an AI agent is in one sentence.')
    if st.button('Ask'):
        msgs=[{"role":"system","content":sys_prompt},{"role":"user","content":user}]
        out = chat(msgs, temperature=0.2)
        st.success(out)
else:
    st.title('RAG Explorer (Minimal)')
    index_file = os.path.join(os.path.dirname(__file__), '..', 'module01_raw', '1.10_rag_basic', 'data', 'index.json')
    st.markdown("Run `python module01_raw/1.10_rag_basic/build_index.py` to (re)build the small demo index.")
    st.caption(f"Index path: {index_file}")
    q = st.text_input('Question', 'How do agents use tools and memory?')
    k = st.slider('Top K', 1, 5, 3)
    if st.button('Show index'):
        if not os.path.exists(index_file):
            st.error('Index not found. Run build_index.py first.')
        else:
            try:
                idx = json.load(open(index_file,'r',encoding='utf-8'))
                st.write(f"Index contains {len(idx)} entries")
                st.json(idx[:5])
            except Exception as e:
                st.error(f"Failed to load index: {e}")
    if st.button('Search'):
        if not os.path.exists(index_file):
            st.error('Index not found. Run build_index.py first.')
        else:
            idx = json.load(open(index_file,'r',encoding='utf-8'))
            qv = embed(q)[0]
            import numpy as np
            def cos(a,b):
                a=np.array(a); b=np.array(b)
                return float(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
            scored = sorted(idx, key=lambda r: cos(qv, r['vec']), reverse=True)[:k]
            st.subheader('Top Matches')
            for r in scored:
                st.markdown(f"**{r['id']}**\n\n{r['text'][:500]}...")