from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
import os
import dotenv
from logger.logger import get_logger, log_info, log_error, log_exception

dotenv.load_dotenv()
logger = get_logger(__name__)


class DatabaseManager:
    
    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self._initialized = False
    
    async def initialize(self):
        if self._initialized:
            return
        
        database_url = os.getenv("DB_URL")
        pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        pool_pre_ping = os.getenv("DB_POOL_PRE_PING", "True").lower() == "true"
        
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            pool_use_lifo=True,
        )
        
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        self._initialized = True
        log_info(logger, "Database manager initialized with connection pooling")
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                log_exception(logger, e)
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def connect(self):
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_maker() as session:
            try:
                yield session
            finally:
                await session.close()
    
    async def get_pool_stats(self) -> dict:
        if not self._initialized or not self.engine:
            return {"error": "Database not initialized"}
        
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
            "checked_out": pool.checkedout(),
            "initialized": self._initialized,
        }
    
    async def health_check(self) -> bool:
        try:
            async with self.connect() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            log_error(logger, f"Health check failed: {str(e)}")
            return False
    
    async def close_all(self):
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            log_info(logger, "All database connections closed")


db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in db_manager.get_session():
        yield session

