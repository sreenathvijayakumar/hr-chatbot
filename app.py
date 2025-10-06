import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Extract text
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

st.title("💬 HR Policy Chatbot")

pdf_path = "HR_Policy.pdf"
if not pdf_path:
    st.error("PDF not found!")
else:
    model = load_model()
    text = extract_text(pdf_path)
    chunks = [chunk for chunk in text.split("\n") if chunk.strip()]
    embeddings = model.encode(chunks)

    q = st.text_input("Ask a question about HR policy:")
    if q:
        q_emb = model.encode([q])
        scores = cosine_similarity(q_emb, embeddings)[0]
        best = chunks[np.argmax(scores)]
        st.markdown("### 🟢 Answer:")
        st.write(best)

