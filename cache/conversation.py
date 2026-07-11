from logger.logger import get_logger
logger = get_logger(__name__)
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import uuid
import json
import time
from sqlalchemy import select, func
from database.database import db_manager
from cache.utils import normalize_datetime
from database.database_models import conversations, messages
from cache.redis_client import redis_client

class ConversationCache:
    def __init__(self, ttl_hours: int = 2, batch_size: int = 50):
        self.ttl_seconds = ttl_hours * 3600
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._user_conversations: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
        
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._batch_size = batch_size
        self._worker_task = None
        self._is_running = True
        
        self._cache_ttl = 60
        self._cache_timestamps: Dict[str, float] = {}
        self._redis = redis_client
    
    def _ensure_redis_connected(self):
        if self._redis.client is None:
            self._redis.connect()
    
    def _get_conv_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}"
    
    def _get_user_convs_key(self, user_id: str) -> str:
        return f"user:{user_id}:convs"
    
    def _get_msgs_key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}:msgs"
    
    def _remove_conversation(self, conv_id: str) -> bool:
        if conv_id in self._conversations:
            user_id = self._conversations[conv_id]["user_id"]
            if user_id in self._user_conversations and conv_id in self._user_conversations[user_id]:
                self._user_conversations[user_id].remove(conv_id)
            del self._conversations[conv_id]
            
            try:
                self._ensure_redis_connected()
                self._redis.delete(
                    self._get_conv_key(conv_id),
                    self._get_msgs_key(conv_id)
                )
                self._redis.lrem(self._get_user_convs_key(user_id), 0, conv_id)
            except Exception as e:
                logger.error(f"Error removing from Redis: {e}")
            
            return True
        return False
    
    def _clean_expired(self) -> None:
        current_time = time.time()
        expired_keys = []
        for conv_id, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self._cache_ttl:
                expired_keys.append(conv_id)
        for conv_id in expired_keys:
            self._conversations.pop(conv_id, None)
            self._cache_timestamps.pop(conv_id, None)
    
    async def _save_conversation_to_db(self, conversation_data: Dict[str, Any]) -> bool:
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
                        msg_key = (
                            msg.get("role"), 
                            msg.get("content"), 
                            timestamp
                        )
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
    
    async def _save_message_to_db(self, conversation_id: str, message_data: Dict[str, Any]) -> bool:
        try:
            async with db_manager.connect() as session:
                stmt = select(conversations).where(conversations.conversation_uuid == conversation_id)
                result = await session.execute(stmt)
                existing_conv = result.scalar_one_or_none()
                
                if not existing_conv:
                    if conversation_id in self._conversations:
                        return await self._save_conversation_to_db(self._conversations[conversation_id])
                    else:
                        conv_data = await self._load_conversation_from_redis(conversation_id)
                        if conv_data:
                            return await self._save_conversation_to_db(conv_data)
                        else:
                            logger.error(f"Conversation {conversation_id} not found")
                            return False
                
                existing_msg_stmt = select(messages).where(
                    messages.conversation_id == conversation_id,
                    messages.role == message_data.get("role"),
                    messages.content == message_data.get("content")
                )
                existing_msg_result = await session.execute(existing_msg_stmt)
                if existing_msg_result.scalar_one_or_none():
                    logger.debug(f"Message already exists in conversation {conversation_id}")
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
    
    async def _load_conversation_from_redis(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_redis_connected()
            conv_data = self._redis.hgetall(self._get_conv_key(conversation_id))
            if not conv_data:
                return None
            
            messages_key = self._get_msgs_key(conversation_id)
            raw_messages = self._redis.zrange(messages_key, 0, -1)
            
            messages_list = []
            for raw_msg in raw_messages:
                msg = json.loads(raw_msg)
                msg["timestamp"] = datetime.fromisoformat(msg["timestamp"])
                msg["metadata"] = json.loads(msg["metadata"])
                messages_list.append(msg)
            
            return {
                "conversation_id": conv_data["conversation_id"],
                "user_id": conv_data["user_id"],
                "metadata": json.loads(conv_data["metadata"]),
                "created_at": datetime.fromisoformat(conv_data["created_at"]),
                "updated_at": datetime.fromisoformat(conv_data["updated_at"]),
                "expires_at": datetime.fromisoformat(conv_data["expires_at"]),
                "messages": messages_list
            }
        except Exception as e:
            logger.error(f"Error loading conversation from Redis: {e}")
            return None
    
    async def _db_worker(self):
        batch = []
        last_flush = datetime.now()
        
        while self._is_running or not self._save_queue.empty():
            try:
                try:
                    item = await asyncio.wait_for(
                        self._save_queue.get(), 
                        timeout=1.0
                    )
                    batch.append(item)
                    self._save_queue.task_done()
                except asyncio.TimeoutError:
                    pass
                
                current_time = datetime.now()
                if (len(batch) >= self._batch_size or 
                    (batch and (current_time - last_flush).seconds >= 5)):
                    
                    tasks = []
                    for conv_id, message_data in batch:
                        task = asyncio.create_task(
                            self._save_message_to_db(conv_id, message_data)
                        )
                        tasks.append(task)
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        successful = sum(1 for r in results if r is True)
                        logger.info(f"Batch saved {successful}/{len(batch)} messages to DB")
                    
                    batch.clear()
                    last_flush = current_time
                
            except Exception as e:
                logger.error(f"Error in DB worker: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def _format_conversation_response(self, conv: Dict[str, Any], include_messages: bool, max_messages: Optional[int]) -> Dict[str, Any]:
        conv_copy = conv.copy()
        conv_copy["expires_at"] = conv_copy["expires_at"].isoformat() if isinstance(conv_copy["expires_at"], datetime) else conv_copy["expires_at"]
        conv_copy["created_at"] = conv_copy["created_at"].isoformat() if isinstance(conv_copy["created_at"], datetime) else conv_copy["created_at"]
        conv_copy["updated_at"] = conv_copy["updated_at"].isoformat() if isinstance(conv_copy["updated_at"], datetime) else conv_copy["updated_at"]
        
        if not include_messages:
            conv_copy.pop("messages", None)
        elif max_messages and conv_copy.get("messages"):
            conv_copy["messages"] = conv_copy["messages"][-max_messages:]
            for msg in conv_copy["messages"]:
                if isinstance(msg["timestamp"], datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
        return conv_copy
    
    async def create_conversation(self, user_id: str, conversation_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        async with self._lock:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            
            conversation_data = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "messages": [],
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=self.ttl_seconds)
            }
            
            self._conversations[conversation_id] = conversation_data
            self._user_conversations[user_id].append(conversation_id)
            self._cache_timestamps[conversation_id] = time.time()
            
            try:
                self._ensure_redis_connected()
                pipe = self._redis.pipeline()
                
                pipe.hset(self._get_conv_key(conversation_id), mapping={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "metadata": json.dumps(metadata or {}),
                    "created_at": conversation_data["created_at"].isoformat(),
                    "updated_at": conversation_data["updated_at"].isoformat(),
                    "expires_at": conversation_data["expires_at"].isoformat()
                })
                pipe.expire(self._get_conv_key(conversation_id), self.ttl_seconds)
                
                pipe.lpush(self._get_user_convs_key(user_id), conversation_id)
                pipe.expire(self._get_user_convs_key(user_id), self.ttl_seconds)
                
                pipe.execute()
            except Exception as e:
                logger.error(f"Error saving to Redis: {e}")
            
            await self._save_queue.put((conversation_id, {"action": "create_conversation"}))
            
            logger.info(f"Created conversation {conversation_id} for user {user_id}")
            return conversation_id
    
    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._lock:
            if conversation_id not in self._conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            conv = self._conversations[conversation_id]
            conv["expires_at"] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(),
                "metadata": metadata or {}
            }
            conv["messages"].append(message)
            conv["updated_at"] = datetime.now()
            self._cache_timestamps[conversation_id] = time.time()
            
            try:
                self._ensure_redis_connected()
                pipe = self._redis.pipeline()
                
                pipe.expire(self._get_conv_key(conversation_id), self.ttl_seconds)
                
                msg_data = {
                    "role": role,
                    "content": content,
                    "metadata": json.dumps(metadata or {}),
                    "timestamp": message["timestamp"].isoformat()
                }
                pipe.zadd(self._get_msgs_key(conversation_id), {json.dumps(msg_data): message["timestamp"].timestamp()})
                pipe.expire(self._get_msgs_key(conversation_id), self.ttl_seconds)
                
                pipe.execute()
            except Exception as e:
                logger.error(f"Error saving message to Redis: {e}")
            
            await self._save_queue.put((conversation_id, message))
            
            logger.debug(f"Added {role} message to conversation {conversation_id}")
            return message
    
    async def get_conversation(self, conversation_id: str, include_messages: bool = True, max_messages: Optional[int] = None) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if conversation_id in self._conversations:
                conv = self._conversations[conversation_id].copy()
                if time.time() - self._cache_timestamps.get(conversation_id, 0) < self._cache_ttl:
                    return self._format_conversation_response(conv, include_messages, max_messages)
            
            try:
                conv_data = await self._load_conversation_from_redis(conversation_id)
                if not conv_data:
                    return None
                
                self._conversations[conversation_id] = conv_data
                self._cache_timestamps[conversation_id] = time.time()
                
                return self._format_conversation_response(conv_data, include_messages, max_messages)
                
            except Exception as e:
                logger.error(f"Error fetching from Redis: {e}")
                return None
    
    async def get_user_conversations(self, user_id: str, limit: int = 10, include_messages: bool = False) -> List[Dict[str, Any]]:
        async with self._lock:
            if user_id in self._user_conversations:
                conv_ids = self._user_conversations[user_id][-limit:]
                convs = []
                for conv_id in conv_ids:
                    conv = await self.get_conversation(conv_id, include_messages)
                    if conv:
                        convs.append(conv)
                return convs
            
            try:
                self._ensure_redis_connected()
                user_key = self._get_user_convs_key(user_id)
                conv_ids = self._redis.lrange(user_key, 0, limit - 1)
                
                if not conv_ids:
                    return []
                
                self._user_conversations[user_id] = conv_ids
                
                convs = []
                for conv_id in conv_ids:
                    conv = await self.get_conversation(conv_id, include_messages)
                    if conv:
                        convs.append(conv)
                
                return convs
                
            except Exception as e:
                logger.error(f"Error fetching user conversations from Redis: {e}")
                return []
    
    async def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        async with self._lock:
            if conversation_id not in self._conversations:
                return False
            
            conv = self._conversations[conversation_id]
            conv["metadata"].update(metadata)
            conv["updated_at"] = datetime.now()
            conv["expires_at"] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            self._cache_timestamps[conversation_id] = time.time()
            
            try:
                self._ensure_redis_connected()
                self._redis.hset(self._get_conv_key(conversation_id), {
                    "metadata": json.dumps(conv["metadata"]),
                    "updated_at": conv["updated_at"].isoformat(),
                    "expires_at": conv["expires_at"].isoformat()
                })
                self._redis.expire(self._get_conv_key(conversation_id), self.ttl_seconds)
            except Exception as e:
                logger.error(f"Error updating metadata in Redis: {e}")
            
            await self._save_queue.put((conversation_id, {"action": "update_metadata", "metadata": metadata}))
            
            logger.debug(f"Updated metadata for conversation {conversation_id}")
            return True
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self._lock:
            if conversation_id not in self._conversations:
                return False
            
            user_id = self._conversations[conversation_id]["user_id"]
            
            self._remove_conversation(conversation_id)
            self._cache_timestamps.pop(conversation_id, None)
            
            try:
                self._ensure_redis_connected()
                self._redis.delete(
                    self._get_conv_key(conversation_id),
                    self._get_msgs_key(conversation_id)
                )
                self._redis.lrem(self._get_user_convs_key(user_id), 0, conversation_id)
            except Exception as e:
                logger.error(f"Error deleting from Redis: {e}")
            
            logger.info(f"Deleted conversation {conversation_id} from cache")
            return True
    
    async def delete_user_conversations(self, user_id: str) -> int:
        async with self._lock:
            conversations_list = self._user_conversations.get(user_id, []).copy()
            count = 0
            for conv_id in conversations_list:
                if self._remove_conversation(conv_id):
                    self._cache_timestamps.pop(conv_id, None)
                    count += 1
            if user_id in self._user_conversations:
                del self._user_conversations[user_id]
            
            try:
                self._ensure_redis_connected()
                conv_ids = self._redis.lrange(self._get_user_convs_key(user_id), 0, -1)
                for conv_id in conv_ids:
                    self._redis.delete(
                        self._get_conv_key(conv_id),
                        self._get_msgs_key(conv_id)
                    )
                self._redis.delete(self._get_user_convs_key(user_id))
            except Exception as e:
                logger.error(f"Error deleting user conversations from Redis: {e}")
            
            logger.info(f"Deleted {count} conversations for user {user_id}")
            return count
    
    async def get_conversation_history(self, conversation_id: str, max_messages: int = 20, format_type: str = "list") -> List:
        conv = await self.get_conversation(conversation_id, include_messages=True, max_messages=max_messages)
        if not conv:
            return []
        
        messages_list = conv.get("messages", [])
        
        if format_type == "string":
            formatted = []
            for msg in messages_list:
                role = msg["role"]
                content = msg["content"]
                formatted.append(f"{role}: {content}")
            return formatted
        
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages_list]
    
    async def clear_expired(self) -> int:
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            for conv_id, timestamp in self._cache_timestamps.items():
                if current_time - timestamp > self._cache_ttl:
                    expired_keys.append(conv_id)
            
            for conv_id in expired_keys:
                self._conversations.pop(conv_id, None)
                self._cache_timestamps.pop(conv_id, None)
            
            logger.info(f"Cleared {len(expired_keys)} expired conversations from cache")
            return len(expired_keys)
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            total_messages = sum(len(conv.get("messages", [])) for conv in self._conversations.values())
            return {
                "total_conversations": len(self._conversations),
                "total_users": len(self._user_conversations),
                "total_messages": total_messages,
                "ttl_hours": self.ttl_seconds // 3600,
                "queue_size": self._save_queue.qsize(),
                "redis_connected": self._redis.client is not None and self._redis.ping()
            }
    
    async def start_background_tasks(self):
        await self.start_cleanup_task()
        
        if self._worker_task is None or self._worker_task.done():
            self._is_running = True
            self._worker_task = asyncio.create_task(self._db_worker())
            logger.info("Started DB worker task")
    
    async def stop_background_tasks(self):
        await self.stop_cleanup_task()
        
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped DB worker task")
    
    async def start_cleanup_task(self):
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started background cleanup task")
    
    async def stop_cleanup_task(self):
        if self._cleanup_task and not self._cleanup_task.done():
            self._shutdown_event.set()
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background cleanup task")
    
    async def _cleanup_loop(self):
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=3600
                    )
                    break
                except asyncio.TimeoutError:
                    removed = await self.clear_expired()
                    if removed > 0:
                        logger.info(f"Cleanup job removed {removed} expired conversations from cache")
        except asyncio.CancelledError:
            logger.info("Cleanup task received cancellation signal")
            raise
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}", exc_info=True)
    
    async def flush_all_to_db(self) -> int:
        remaining = self._save_queue.qsize()
        if remaining > 0:
            logger.info(f"Flushing {remaining} items to database...")
            await self._save_queue.join()
            logger.info(f"Flushed all {remaining} items to database")
        return remaining
    
    async def save_all_conversations_to_db(self) -> int:
        await self.flush_all_to_db()
        
        async with self._lock:
            conv_list = list(self._conversations.values())
            if not conv_list:
                return 0
            
            save_tasks = []
            for conv_data in conv_list:
                task = asyncio.create_task(self._save_conversation_to_db(conv_data))
                save_tasks.append(task)
            
            results = await asyncio.gather(*save_tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            logger.info(f"Saved {successful}/{len(conv_list)} conversations to DB during shutdown")
            return successful
    
    async def close(self):
        self._redis.close()

cache = ConversationCache(ttl_hours=2)

async def initialize_cache():
    await cache.start_background_tasks()
    cache._ensure_redis_connected()