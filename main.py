# main.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# 1️⃣ INITIAL SETUP (RUNS ONCE)
# ---------------------------

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM (Ollama must be running)
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

# ---------------------------
# 2️⃣ LOAD & CHUNK ALL PDFs
# ---------------------------

raw_documents = []

# Load every PDF in the project folder
pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]

if not pdf_files:
    raise ValueError("❌ No PDF files found in the project folder.")

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    raw_documents.extend(loader.load())

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = text_splitter.split_documents(raw_documents)

# ---------------------------
# 3️⃣ CREATE EMBEDDINGS & FAISS INDEX
# ---------------------------

texts = [doc.page_content for doc in documents]
embeddings = embedding_model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ---------------------------
# 4️⃣ RAG QUERY FUNCTION (USED BY STREAMLIT)
# ---------------------------

def answer_query(question, top_k=3):
    """
    Takes a user question
    Retrieves relevant chunks from all PDFs using FAISS
    Generates an answer strictly from document context
    Returns answer and sources
    """

    # Embed the question
    query_embedding = embedding_model.encode([question])

    # Search FAISS
    distances, indices = index.search(np.array(query_embedding), top_k)

    if len(indices[0]) == 0:
        return "I don't know.", []

    # Collect retrieved chunks
    retrieved_docs = [documents[i] for i in indices[0]]

    # Build context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt
    prompt = f"""
You are a document-based assistant.
Answer strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    # LLM call
    response = llm.invoke(prompt)

    # Build sources list
    sources = [
        f"{doc.metadata.get('source')} - page {doc.metadata.get('page')}"
        for doc in retrieved_docs
    ]

    return response.content, sources