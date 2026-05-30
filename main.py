from fastapi import FastAPI
from routes.rag import router as rag_router

app = FastAPI(title="RAG API", version="1.0")

app.include_router(rag_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)