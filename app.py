from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Insurance Agent RAG Bot")

app.include_router(router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Insurance Agent API running"}
