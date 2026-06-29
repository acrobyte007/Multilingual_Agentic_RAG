import asyncio
from sqlalchemy import text

from database.database import db_manager
from logger.logger import get_logger

logger = get_logger(__name__)


async def migrate_messages_to_uuid_fk():
    """
    One-time migration:
    Convert messages.conversation_id from INTEGER → UUID FK
    """

    await db_manager.initialize()

    async with db_manager.engine.begin() as conn:

        # 1. Add new UUID column
        await conn.execute(
            text("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS conversation_uuid VARCHAR;
            """)
        )

        # 2. Backfill UUID from conversations table
        await conn.execute(
            text("""
            UPDATE messages m
            SET conversation_uuid = c.conversation_uuid
            FROM conversations c
            WHERE m.conversation_id = c.id;
            """)
        )

        # 3. Drop old FK constraint (safe attempt)
        await conn.execute(
            text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints
                    WHERE constraint_name LIKE '%messages_conversation_id_fkey%'
                ) THEN
                    ALTER TABLE messages
                    DROP CONSTRAINT messages_conversation_id_fkey;
                END IF;
            END
            $$;
            """)
        )

        # 4. Drop old integer column
        await conn.execute(
            text("""
            ALTER TABLE messages
            DROP COLUMN conversation_id;
            """)
        )

        # 5. Rename UUID column → conversation_id
        await conn.execute(
            text("""
            ALTER TABLE messages
            RENAME COLUMN conversation_uuid TO conversation_id;
            """)
        )

        # 6. Add new FK constraint (UUID → conversations.uuid)
        await conn.execute(
            text("""
            ALTER TABLE messages
            ADD CONSTRAINT fk_messages_conversation_uuid
            FOREIGN KEY (conversation_id)
            REFERENCES conversations(conversation_uuid)
            ON DELETE CASCADE;
            """)
        )

        # 7. Add index
        await conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS
            ix_messages_conversation_id
            ON messages(conversation_id);
            """)
        )

    logger.info("Messages table migration to UUID FK completed successfully")


if __name__ == "__main__":
    asyncio.run(migrate_messages_to_uuid_fk())