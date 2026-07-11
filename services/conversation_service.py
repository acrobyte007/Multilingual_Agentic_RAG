from logger.logger import get_logger
logger = get_logger(__name__)
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import select
from database.database import db_manager
from services.utils import normalize_datetime
from database.database_models import conversations, messages

class ConversationDB:
    @staticmethod
    async def save_conversation(conversation_data: Dict[str, Any]) -> bool:
        try:
            async with db_manager.connect() as session:
                stmt = select(conversations).where(
                    conversations.conversation_uuid == conversation_data["conversation_id"]
                )
                result = await session.execute(stmt)
                existing_conv = result.scalar_one_or_none()
                
                if existing_conv:
                    existing_conv.meta_data = conversation_data.get("metadata", {})
                    session.add(existing_conv)
                    await session.flush()
                    
                    existing_msg_stmt = select(messages).where(
                        messages.conversation_id == conversation_data["conversation_id"]
                    )
                    existing_msg_result = await session.execute(existing_msg_stmt)
                    existing_messages = existing_msg_result.scalars().all()
                    existing_contents = {(msg.role, msg.content, msg.created_at) for msg in existing_messages}
                    
                    for msg in conversation_data.get("messages", []):
                        timestamp = normalize_datetime(msg.get("timestamp", datetime.now()))
                        msg_key = (msg.get("role"), msg.get("content"), timestamp)
                        if msg_key not in existing_contents:
                            new_message = messages(
                                conversation_id=conversation_data["conversation_id"],
                                role=msg.get("role"),
                                content=msg.get("content"),
                                meta_data=msg.get("metadata", {}),
                                created_at=timestamp
                            )
                            session.add(new_message)
                    
                    await session.commit()
                    logger.debug(f"Updated conversation {conversation_data['conversation_id']} in database")
                    return True
                else:
                    new_conversation = conversations(
                        conversation_uuid=conversation_data["conversation_id"],
                        user_id=int(conversation_data["user_id"]),
                        meta_data=conversation_data.get("metadata", {})
                    )
                    session.add(new_conversation)
                    await session.flush()
                    
                    for msg in conversation_data.get("messages", []):
                        timestamp = normalize_datetime(msg.get("timestamp", datetime.now()))
                        new_message = messages(
                            conversation_id=conversation_data["conversation_id"],
                            role=msg.get("role"),
                            content=msg.get("content"),
                            meta_data=msg.get("metadata", {}),
                            created_at=timestamp
                        )
                        session.add(new_message)
                    
                    await session.commit()
                    logger.debug(f"Created conversation {conversation_data['conversation_id']} in database")
                    return True
        except Exception as e:
            logger.error(f"Error saving conversation to DB: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def save_message(conversation_id: str, message_data: Dict[str, Any]) -> bool:
        try:
            async with db_manager.connect() as session:
                stmt = select(conversations).where(conversations.conversation_uuid == conversation_id)
                result = await session.execute(stmt)
                existing_conv = result.scalar_one_or_none()
                
                if not existing_conv:
                    logger.error(f"Conversation {conversation_id} not found in DB")
                    return False
                
                existing_msg_stmt = select(messages).where(
                    messages.conversation_id == conversation_id,
                    messages.role == message_data.get("role"),
                    messages.content == message_data.get("content")
                )
                existing_msg_result = await session.execute(existing_msg_stmt)
                if existing_msg_result.scalar_one_or_none():
                    return True
                
                new_message = messages(
                    conversation_id=conversation_id,
                    role=message_data.get("role"),
                    content=message_data.get("content"),
                    meta_data=message_data.get("metadata", {}),
                    created_at=normalize_datetime(message_data.get("timestamp", datetime.now()))
                )
                session.add(new_message)
                await session.commit()
                logger.debug(f"Saved message to conversation {conversation_id} in database")
                return True
        except Exception as e:
            logger.error(f"Error saving message to DB: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def load_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            async with db_manager.connect() as session:
                stmt = select(conversations).where(conversations.conversation_uuid == conversation_id)
                result = await session.execute(stmt)
                conv = result.scalar_one_or_none()
                
                if not conv:
                    return None
                
                msg_stmt = select(messages).where(messages.conversation_id == conversation_id).order_by(messages.created_at)
                msg_result = await session.execute(msg_stmt)
                messages_list = msg_result.scalars().all()
                
                return {
                    "conversation_id": conv.conversation_uuid,
                    "user_id": str(conv.user_id),
                    "metadata": conv.meta_data or {},
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "metadata": msg.meta_data or {},
                            "timestamp": msg.created_at
                        }
                        for msg in messages_list
                    ]
                }
        except Exception as e:
            logger.error(f"Error loading conversation from DB: {e}")
            return None
    
    @staticmethod
    async def load_user_conversations(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            async with db_manager.connect() as session:
                stmt = select(conversations).where(
                    conversations.user_id == int(user_id)
                ).limit(limit).order_by(conversations.updated_at.desc())
                result = await session.execute(stmt)
                convs_data = result.scalars().all()
                
                convs = []
                for conv in convs_data:
                    conv_data = await ConversationDB.load_conversation(conv.conversation_uuid)
                    if conv_data:
                        convs.append(conv_data)
                return convs
        except Exception as e:
            logger.error(f"Error loading user conversations from DB: {e}")
            return []

conversation_db = ConversationDB()