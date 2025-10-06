import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
import faiss

# --- Safe Model Loading (no torch.to()) ---
@st.cache_resource
def load_model():
    import torch
    torch.set_default_tensor_type(torch.FloatTensor)  # Prevents conversion issues
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=None)
    return model

model = load_model()

st.title("💼 HR Policy Chatbot")

# --- Load and Embed PDF ---
@st.cache_resource
def load_data():
    pdf = PdfReader("HR_Policy.pdf")
    text = " ".join([page.extract_text() or "" for page in pdf.pages])
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    embeddings = model.encode(sentences, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return sentences, index

sentences, index = load_data()

# --- Chat Interface ---
query = st.text_input("Ask your HR question:")
if query:
    q_emb = model.encode([query])
    D, I = index.search(q_emb, 3)
    st.subheader("Top Answers:")
    for i in I[0]:
        st.write("-", sentences[i])
