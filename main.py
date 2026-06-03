from fastapi import FastAPI
from routes.rag import router as rag_router
app = FastAPI(title="RAG API", version="1.0")
app.include_router(rag_router)
