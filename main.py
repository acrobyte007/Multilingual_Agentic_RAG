import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database.database import db_manager
from cache.conversation import cache
from services.embedding_model import embedding_service
from database.vector_database import pinecone_service
from cache.redis_client import redis_client
from cache.conversation import initialize_cache
from routes.rag import router as rag_router
from routes.auth import router as auth_router
from routes.docs import router as docs_router
from logger.logger import get_logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")
    redis_client.connect()
    await db_manager.initialize()
    await embedding_service.initialize()
    pinecone_service.initialize()
    await cache.start_background_tasks()
    logger.info("Cache background tasks started")
    
    try:
        yield
    finally:
        logger.info("Shutting down application...")

        try:
            logger.info("Flushing queued messages to database...")
            queue_size = cache._save_queue.qsize() if hasattr(cache, '_save_queue') else 0
            if queue_size > 0:
                await cache.flush_all_to_db()
                logger.info(f"Flushed {queue_size} queued messages")
            else:
                logger.info("No queued messages to flush")
            
            try:
                logger.info("Saving all conversations to database...")
                saved_count = await cache.save_all_conversations_to_db()
                logger.info(f"Successfully saved {saved_count} conversations")
            except Exception as e:
                logger.error(f"Error saving conversations: {e}", exc_info=True)
            
            try:
                logger.info("Stopping background tasks...")
                await cache.stop_background_tasks()
                logger.info("Background tasks stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping background tasks: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
        
        finally:
            try:
                await db_manager.close_all()
                logger.info("Database connections closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}", exc_info=True)
            
            try:
                if hasattr(pinecone_service, 'close'):
                    pinecone_service.close()
                    logger.info("Pinecone service closed")
            except Exception as e:
                logger.error(f"Error closing pinecone: {e}", exc_info=True)
            
            logger.info("Shutdown complete")


app = FastAPI(
    title="RAG API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(auth_router)
app.include_router(rag_router)
app.include_router(docs_router)