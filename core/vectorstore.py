from langchain_community.vectorstores import FAISS
from config import FAISS_DIR

def build_or_load_faiss(texts, embeddings):
    if texts:
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_DIR)
        return vector_store
    else:
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
