import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# -----------------------------
# Load & preprocess the PDF once
# -----------------------------
@st.cache_resource
def load_model():
    embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

@st.cache_data
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

@st.cache_data
def create_embeddings(text, model):
    sentences = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    embeddings = model.encode(sentences)
    return sentences, embeddings

# -----------------------------
# Core logic for answering
# -----------------------------
def generate_answer(question, model, generator, sentences, embeddings):
    query_emb = model.encode([question])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(scores)
    context = sentences[top_idx]

    prompt = f"Answer the HR question clearly based only on this context:\n\n{context}\n\nQuestion: {question}"
    result = generator(prompt, max_new_tokens=200)
    return result[0]['generated_text']

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬", layout="centered")

st.title("💬 HR Policy Chatbot")
st.write("Ask me anything about your HR policies (like leave, benefits, working hours, etc.)")

pdf_path = "HR_Policy.pdf"

if not os.path.exists(pdf_path):
    st.error("⚠️ HR_Policy.pdf not found. Please upload it to the app folder.")
else:
    embedder, generator = load_model()
    text = extract_text_from_pdf(pdf_path)
    sentences, embeddings = create_embeddings(text, embedder)

    question = st.text_input("Enter your HR-related question:")
    if question:
        with st.spinner("Finding the best answer..."):
            answer = generate_answer(question, embedder, generator, sentences, embeddings)
        st.markdown("### 🟢 Answer:")
        st.write(answer)
