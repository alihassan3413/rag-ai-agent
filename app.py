from fastapi import FastAPI
from api.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Insurance Agent RAG Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or put your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Insurance Agent API running"}
