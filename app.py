import os
from flask import Flask, request, render_template, jsonify
import pdfplumber
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store chunks and embeddings in memory for demo purposes
pdf_chunks = []
pdf_embeddings = []

def chunk_text(text, chunk_size=500):
    """Split text into chunks of chunk_size words."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def load_pdf(file_path):
    global pdf_chunks, pdf_embeddings
    pdf_chunks = []
    pdf_embeddings = []
    with pdfplumber.open(file_path) as pdf:
        text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
    pdf_chunks = chunk_text(text)
    pdf_embeddings = model.encode(pdf_chunks, convert_to_tensor=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    load_pdf(file_path)
    return 'PDF uploaded and processed!'

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    question_embedding = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, pdf_embeddings, top_k=3)[0]
    results = [pdf_chunks[hit['corpus_id']] for hit in hits]
    return jsonify({'answers': results})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
