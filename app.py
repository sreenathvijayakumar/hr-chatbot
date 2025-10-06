import streamlit as st
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Streamlit app configuration
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬")
st.title("💬 HR Policy Chatbot (CPU Version)")

# Load PDF
PDF_PATH = "HR_Policy.pdf"

def extract_text(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

@st.cache_resource
def load_model():
    """Load embedding model (CPU friendly)"""
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_data
def create_embeddings(text, model):
    """Split text into chunks and embed"""
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    embeddings = model.encode(chunks)
    return chunks, embeddings

if not PDF_PATH:
    st.error("❌ HR_Policy.pdf not found! Please upload your PDF.")
else:
    text = extract_text(PDF_PATH)
    if not text:
        st.error("❌ Could not extract text from PDF.")
    else:
        model = load_model()
        sentences, embeddings = create_embeddings(text, model)

        query = st.text_input("Ask a question about HR Policy:")
        if query:
            q_emb = model.encode([query])
            sims = cosine_similarity(q_emb, embeddings)[0]
            answer = sentences[int(np.argmax(sims))]
            st.markdown("### 🟢 Answer:")
            st.write(answer)
