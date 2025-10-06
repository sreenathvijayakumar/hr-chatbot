import streamlit as st
import pickle
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

st.title("🤖 HR Policy Chatbot")

# Load the learned data
with open("faiss_store.pkl", "rb") as f:
    index, chunks = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Ask questions
user_question = st.text_input("Ask your HR question:")

if user_question:
    q_emb = model.encode([user_question])
    _, top_indices = index.search(np.array(q_emb), k=3)
    context = " ".join([chunks[i] for i in top_indices[0]])

    prompt = f"Answer the question using only this context:\n\n{context}\n\nQuestion: {user_question}\nAnswer:"
    answer = qa_model(prompt)[0]["generated_text"]

    st.success(answer)
