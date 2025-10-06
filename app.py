import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="HR Policy Chatbot", page_icon="💬")

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Extract and clean text from PDF
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "  # add space instead of newline
    # Merge small lines into paragraphs
    paragraphs = [p.strip() for p in text.split(". ") if len(p.strip()) > 30]
    return paragraphs

st.title("💬 HR Policy Chatbot")
st.caption("Ask any question from the HR Policy document.")

pdf_path = "HR_Policy.pdf"

try:
    model = load_model()
    paragraphs = extract_text(pdf_path)
    embeddings = model.encode(paragraphs, convert_to_numpy=True)

    query = st.text_input("🔍 Ask your HR policy question:")
    if query:
        q_emb = model.encode([query])
        scores = cosine_similarity(q_emb, embeddings)[0]
        best_idx = np.argmax(scores)
        answer = paragraphs[best_idx]

        st.markdown("### 🟢 Answer:")
        st.write(answer)

except FileNotFoundError:
    st.error("❌ PDF file not found. Please upload HR_Policy.pdf.")
except Exception as e:
    st.error(f"⚠️ Error: {str(e)}")
