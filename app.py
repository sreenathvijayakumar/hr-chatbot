import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# Load models (only once)
# -----------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    return embedder, generator


# -----------------------------
# Extract text from PDF
# -----------------------------
@st.cache_data
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -----------------------------
# Create embeddings (no caching model object)
# -----------------------------
@st.cache_data
def create_embeddings(text):
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")
    sentences = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    embeddings = model.encode(sentences)
    return sentences, embeddings


# -----------------------------
# Answer generation
# -----------------------------
def generate_answer(question, generator, sentences, embeddings):
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")
    query_emb = model.encode([question])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(scores)
    context = sentences[top_idx]

    prompt = f"Answer this HR-related question clearly based only on the context:\n\nContext: {context}\n\nQuestion: {question}"
    result = generator(prompt, max_new_tokens=200)
    return result[0]['generated_text']


# -----------------------------
# Streamlit UI
# ---------------------
