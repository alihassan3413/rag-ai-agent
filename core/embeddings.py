from langchain_huggingface import HuggingFaceEmbeddings
import os

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
