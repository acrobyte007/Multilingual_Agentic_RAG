import asyncio
from sqlalchemy import text
from database.database import db_manager
from logger.logger import get_logger

logger = get_logger(__name__)


async def clean_all_data():
    """
    Clean all data from tables without dropping schema
    """
    await db_manager.initialize()

    async with db_manager.engine.begin() as conn:
        logger.info("Cleaning all tables...")
        
        # Truncate tables in correct order to handle foreign key constraints
        # Using CASCADE will automatically handle dependent tables
        await conn.execute(text("TRUNCATE TABLE messages CASCADE;"))
        await conn.execute(text("TRUNCATE TABLE conversations CASCADE;"))
        await conn.execute(text("TRUNCATE TABLE documents CASCADE;"))
        await conn.execute(text("TRUNCATE TABLE users CASCADE;"))
        
        # Reset sequences
        await conn.execute(text("""
            SELECT setval(pg_get_serial_sequence('users', 'id'), 
                COALESCE((SELECT MAX(id) FROM users), 1), false);
        """))
        await conn.execute(text("""
            SELECT setval(pg_get_serial_sequence('documents', 'id'), 
                COALESCE((SELECT MAX(id) FROM documents), 1), false);
        """))
        
        logger.info("All tables cleaned and sequences reset!")


if __name__ == "__main__":
    asyncio.run(clean_all_data())