import streamlit as st
import pickle
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="HR Chatbot", page_icon="🤖")
st.title("🤖 HR Policy Chatbot")

# --- Load saved FAISS index ---
try:
    with open("faiss_store.pkl", "rb") as f:
        index, chunks = pickle.load(f)
except Exception as e:
    st.error(f"Error loading faiss_store.pkl: {e}")
    st.stop()

# --- Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load small LLM ---
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# --- Question Input ---
question = st.text_input("Ask your question about HR Policy:")

if question:
    q_emb = model.encode([question])
    _, top_indices = index.search(np.array(q_emb), k=3)
    context = " ".join([chunks[i] for i in top_indices[0]])

    prompt = f"Answer this question using only this HR policy info:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    with st.spinner("Thinking..."):
        answer = qa_model(prompt, max_new_tokens=200)[0]["generated_text"]
    st.success(answer)

