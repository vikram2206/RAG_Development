import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- SETUP ---
PDF_FOLDER = ""  # Absolute path is safer here
QUERY = "deprexis used for more than 10 weeks"
MODEL_NAME = "mistral"

# --- LOAD PDFs ---
documents = []
print(f" Looking for PDFs in: {PDF_FOLDER}")
for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, file)
        try:
            loader = PyPDFLoader(path)
            doc = loader.load()
            documents.extend(doc)
            print(f"✓ Loaded {file} with {len(doc)} pages")
        except Exception as e:
            print(f" Failed to load {file}: {e}")

print(f"📄 Total documents loaded: {len(documents)}")

# --- SPLIT TEXT ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f" Total chunks after splitting: {len(docs)}")

# --- Check before FAISS ---
if not docs:
    print(" No text chunks were generated. Cannot proceed to vector indexing.")
    exit()

# --- EMBEDDINGS + VECTOR DB ---
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embedding)
print("✓ Vector DB created")

# --- RAG with OLLAMA ---
llm = Ollama(model=MODEL_NAME)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

# --- QUERY ---
result = qa.invoke({"query": QUERY})

# --- OUTPUT ---
print("\n Answer:\n", result["result"])
print("\n Sources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source"))
