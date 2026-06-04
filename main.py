import asyncio
from fastapi import FastAPI
from cache.conversation import cache
from routes.rag import router as rag_router
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0")
app.include_router(rag_router)

async def cleanup_expired_conversations():
    while True:
        await asyncio.sleep(3600)
        removed = await cache.clear_expired()
        if removed > 0:
            logger.info(f"Cleanup job removed {removed} expired conversations")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_expired_conversations())
    logger.info("Started cleanup task for expired conversations")