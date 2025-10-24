import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Ensure corpus directory exists
corpus_dir = "data/corpus"
os.makedirs(corpus_dir, exist_ok=True)

# Collect all text files inside data/corpus
docs = []
for filename in os.listdir(corpus_dir):
    if filename.endswith(".txt"):
        path = os.path.join(corpus_dir, filename)
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

if not docs:
    raise ValueError("❌ No text files found in data/corpus. Please add .txt documents first.")

# Split text into chunks for indexing
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Create FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("data/vector_index")
print("✅ Successfully built FAISS index at data/vector_index")
