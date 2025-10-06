import streamlit as st
import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Page setup
st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬")
st.title("💬 HR Policy Chatbot (Fixed Version)")

PDF_PATH = "HR_Policy.pdf"

# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Extract PDF text
def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Create embeddings (do NOT pass model as parameter)
@st.cache_data
def create_embeddings(text):
    model = load_model()  # load inside the function
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    embeddings = model.encode(chunks)
    return chunks, embeddings

# Main logic
if not PDF_PATH:
    st.error("❌ HR_Policy.pdf not found!")
else:
    text = extract_text(PDF_PATH)
    if not text:
        st.error("❌ Could not extract text from PDF.")
    else:
        sentences, embeddings = create_embeddings(text)
        model = load_model()

        query = st.text_input("Ask a question about HR Policy:")
        if query:
            q_emb = model.encode([query])
            sims = cosine_similarity(q_emb, embeddings)[0]
            best_answer = sentences[int(np.argmax(sims))]
            st.markdown("### 🟢 Answer:")
            st.write(best_answer)
