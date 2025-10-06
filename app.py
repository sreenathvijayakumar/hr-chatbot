import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
import torch

# Use CPU only
torch.set_default_device("cpu")

st.title("💼 HR Policy Chatbot")

# Load model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Load embeddings once
@st.cache_resource
def load_embeddings():
    pdf = PdfReader("HR_Policy.pdf")
    text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    sentences = text.split(". ")
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return sentences, index, embeddings

sentences, index, embeddings = load_embeddings()

# Chat input
query = st.text_input("Ask your HR question:")
if query:
    q_emb = model.encode([query])
    D, I = index.search(q_emb, 1)
    st.write("📄", sentences[I[0][0]])

