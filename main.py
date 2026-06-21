import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from database.database import db_manager
from cache.conversation import cache
from ingestion.embedding_model import embedding_service
from database.vector_database import pinecone_service

from routes.rag import router as rag_router
from logger.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")

    await db_manager.initialize()
    logger.info("Database initialized")
    await embedding_service.initialize()
    logger.info("Embedding service initialized")
    pinecone_service.initialize()
    logger.info("Pinecone service ready")
    cleanup_task = asyncio.create_task(
        cache.cleanup_expired_conversations()
    )
    logger.info("Started cleanup task for expired conversations")

    try:
        yield
    finally:
        logger.info("Shutting down application...")

        await db_manager.close_all()
        logger.info("Database connections closed")

        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            logger.info("Cleanup task stopped cleanly")


app = FastAPI(
    title="RAG API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(rag_router)