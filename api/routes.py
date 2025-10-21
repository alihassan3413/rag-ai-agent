from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os, shutil

from core.pdf_loader import load_pdfs, split_docs
from core.embeddings import get_embeddings
from core.vectorstore import build_or_load_faiss
from core.llm import get_llm
from core.qa_chain import get_retriever, get_qa_chain
from config import PDF_DIR

router = APIRouter()

embeddings = get_embeddings()
llm = get_llm()
vector_store = None
retriever = None
qa_chain = None

class QueryRequest(BaseModel):
    question: str
    
def initialize_index():
    global vector_store, retriever, qa_chain
    try:
        # Load FAISS index if exists
        if os.path.exists("data/faiss_index"):
            vector_store = build_or_load_faiss(None, embeddings)  # loads saved index
            retriever = get_retriever(vector_store)
            qa_chain = get_qa_chain(llm, retriever)
            print("✅ Loaded existing FAISS index")
        else:
            print("⚠️ No FAISS index found. Upload PDFs first.")
    except Exception as e:
        print("❌ Error initializing index:", e)

# call at import time
initialize_index()

@router.post("/ask")
def ask(req: QueryRequest):
    global qa_chain
    if not qa_chain:
        return JSONResponse({"error": "No index available yet. Upload PDFs first."}, status_code=400)

    result = qa_chain({"query": req.question})
    return {"answer": result["result"]}  # no sources shown

@router.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vector_store, retriever, qa_chain

    saved_files = []
    for file in files:
        path = os.path.join(PDF_DIR, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    # Load only new PDFs
    docs = load_pdfs()
    texts = split_docs(docs)

    if vector_store:
        # Incremental add
        vector_store.add_documents(texts)
    else:
        vector_store = build_or_load_faiss(texts, embeddings)

    # Save updated index
    vector_store.save_local("data/faiss_index")

    retriever = get_retriever(vector_store)
    qa_chain = get_qa_chain(llm, retriever)

    return {"uploaded": saved_files, "chunks": len(texts)}
