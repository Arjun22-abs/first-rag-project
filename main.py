import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

print("🚀 Starting Local RAG System")


pdf_path = "sample.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages from PDF")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"✅ Split into {len(chunks)} chunks")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Embeddings model loaded")


vectorstore = FAISS.from_documents(chunks, embeddings)

print("✅ Vector store created")


llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

print("✅ Local LLM (gemma3:4b) connected")


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

print("\n🧠 RAG is ready! Ask questions (type 'exit' to quit)\n")


while True:
    query = input("❓ Question: ")
    if query.lower() == "exit":
        print("👋 Exiting RAG")
        break

    answer = qa_chain.invoke({"query": query})["result"]
    print("\n✅ Answer:\n", answer)
    print("-" * 60)