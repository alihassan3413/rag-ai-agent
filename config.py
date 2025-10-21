import os
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "data/pdfs"
FAISS_DIR = "data/faiss_index"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
