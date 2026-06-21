import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database.database import db_manager
from cache.conversation import cache
from routes.rag import router as rag_router
from logger.logger import get_logger

logger = get_logger(__name__)


async def cleanup_expired_conversations():
    try:
        while True:
            await asyncio.sleep(3600)

            removed = await cache.clear_expired()

            if removed > 0:
                logger.info(f"Cleanup job removed {removed} expired conversations")

    except asyncio.CancelledError:
        logger.info("Cleanup task received shutdown signal")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_expired_conversations())
    await db_manager.create_tables()
    logger.info("Started cleanup task for expired conversations")
    try:
        yield
        await db_manager.close_all()
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Cleanup task stopped cleanly")


app = FastAPI(
    title="RAG API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(rag_router)