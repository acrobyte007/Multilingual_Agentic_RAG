from logger.logger import get_logger
logger = get_logger(__name__)
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import uuid
import json
from services.conversation_service import conversation_db
from cache.redis_client import redis_client

class ConversationCache:
    def __init__(self, ttl_hours: int = 2, batch_size: int = 50):
        self.ttl_seconds = ttl_hours * 3600
        self.delete_after_seconds = 3600
        self.batch_size = batch_size
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
        self._save_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task = None
        self._is_running = True
        self._redis = redis_client
        self._redis_initialized = False
    
    def _ensure_redis(self):
        if not self._redis_initialized:
            self._redis.connect()
            self._redis_initialized = True
    
    def _get_conv_key(self, conv_id: str) -> str:
        return f"conv:{conv_id}"
    
    def _get_user_key(self, user_id: str) -> str:
        return f"user:{user_id}:convs"
    
    def _get_msgs_key(self, conv_id: str) -> str:
        return f"conv:{conv_id}:msgs"
    
    def _get_saved_key(self, conv_id: str) -> str:
        return f"conv:{conv_id}:saved"
    
    def _get_status_key(self, conv_id: str) -> str:
        return f"conv:{conv_id}:status"
    
    def _get_msg_id(self, message: Dict[str, Any]) -> str:
        ts = message['timestamp'].isoformat() if isinstance(message['timestamp'], datetime) else message['timestamp']
        return f"{message['role']}:{message['content'][:50]}:{ts}"
    
    def _format_response(self, conv: Dict[str, Any], include_messages: bool, max_messages: Optional[int]) -> Dict[str, Any]:
        conv_copy = conv.copy()
        for field in ["expires_at", "created_at", "updated_at"]:
            if field in conv_copy and isinstance(conv_copy[field], datetime):
                conv_copy[field] = conv_copy[field].isoformat()
        
        if not include_messages:
            conv_copy.pop("messages", None)
        elif max_messages and conv_copy.get("messages"):
            conv_copy["messages"] = conv_copy["messages"][-max_messages:]
            for msg in conv_copy["messages"]:
                if isinstance(msg["timestamp"], datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()
        return conv_copy
    
    async def _load_from_redis(self, conv_id: str) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_redis()
            conv_data = self._redis.hgetall(self._get_conv_key(conv_id))
            if not conv_data:
                return None
            
            raw_msgs = self._redis.zrange(self._get_msgs_key(conv_id), 0, -1)
            messages = []
            for raw in raw_msgs:
                msg = json.loads(raw)
                msg["timestamp"] = datetime.fromisoformat(msg["timestamp"])
                msg["metadata"] = json.loads(msg["metadata"])
                messages.append(msg)
            
            return {
                "conversation_id": conv_data["conversation_id"],
                "user_id": conv_data["user_id"],
                "metadata": json.loads(conv_data["metadata"]),
                "created_at": datetime.fromisoformat(conv_data["created_at"]),
                "updated_at": datetime.fromisoformat(conv_data["updated_at"]),
                "expires_at": datetime.fromisoformat(conv_data["expires_at"]),
                "messages": messages
            }
        except Exception as e:
            logger.error(f"Error loading from Redis: {e}")
            return None
    
    async def _save_to_redis(self, conv_data: Dict[str, Any]) -> bool:
        try:
            self._ensure_redis()
            now = datetime.now()
            pipe = self._redis.pipeline()
            
            pipe.hset(self._get_conv_key(conv_data["conversation_id"]), mapping={
                "conversation_id": conv_data["conversation_id"],
                "user_id": conv_data["user_id"],
                "metadata": json.dumps(conv_data["metadata"]),
                "created_at": conv_data["created_at"].isoformat(),
                "updated_at": conv_data["updated_at"].isoformat(),
                "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat()
            })
            pipe.expire(self._get_conv_key(conv_data["conversation_id"]), self.ttl_seconds)
            
            pipe.lpush(self._get_user_key(conv_data["user_id"]), conv_data["conversation_id"])
            pipe.expire(self._get_user_key(conv_data["user_id"]), self.ttl_seconds)
            
            for msg in conv_data.get("messages", []):
                msg_data = {
                    "role": msg["role"],
                    "content": msg["content"],
                    "metadata": json.dumps(msg.get("metadata", {})),
                    "timestamp": msg["timestamp"].isoformat() if isinstance(msg["timestamp"], datetime) else msg["timestamp"]
                }
                ts = msg["timestamp"].timestamp() if isinstance(msg["timestamp"], datetime) else datetime.fromisoformat(msg["timestamp"]).timestamp()
                pipe.zadd(self._get_msgs_key(conv_data["conversation_id"]), {json.dumps(msg_data): ts})
            
            pipe.expire(self._get_msgs_key(conv_data["conversation_id"]), self.ttl_seconds)
            pipe.set(self._get_status_key(conv_data["conversation_id"]), "restored", ex=self.ttl_seconds)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Error saving to Redis: {e}")
            return False
    
    async def _db_worker(self):
        batch = []
        last_flush = datetime.now()
        while self._is_running or not self._save_queue.empty():
            try:
                try:
                    item = await asyncio.wait_for(self._save_queue.get(), timeout=1.0)
                    batch.append(item)
                    self._save_queue.task_done()
                except asyncio.TimeoutError:
                    pass
                
                if len(batch) >= self.batch_size or (batch and (datetime.now() - last_flush).seconds >= 5):
                    tasks = []
                    for conv_id, data in batch:
                        if isinstance(data, dict) and data.get("action") == "create_conversation":
                            conv_data = await self._load_from_redis(conv_id)
                            if conv_data:
                                tasks.append(asyncio.create_task(conversation_db.save_conversation(conv_data)))
                        else:
                            tasks.append(asyncio.create_task(conversation_db.save_message(conv_id, data)))
                    
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        successful = sum(1 for r in results if r is True)
                        logger.info(f"Batch saved {successful}/{len(batch)} items to DB")
                    batch.clear()
                    last_flush = datetime.now()
            except Exception as e:
                logger.error(f"Error in DB worker: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def create_conversation(self, user_id: str, conversation_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        async with self._lock:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            
            now = datetime.now()
            self._ensure_redis()
            
            # Save to Redis immediately
            pipe = self._redis.pipeline()
            pipe.hset(self._get_conv_key(conversation_id), mapping={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "metadata": json.dumps(metadata or {}),
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=self.ttl_seconds)).isoformat()
            })
            pipe.expire(self._get_conv_key(conversation_id), self.ttl_seconds)
            pipe.lpush(self._get_user_key(user_id), conversation_id)
            pipe.expire(self._get_user_key(user_id), self.ttl_seconds)
            pipe.set(self._get_status_key(conversation_id), "pending", ex=self.ttl_seconds)
            pipe.execute()
            
            # Queue for background DB save
            await self._save_queue.put((conversation_id, {"action": "create_conversation"}))
            
            logger.info(f"Created conversation {conversation_id} for user {user_id} in Redis")
            return conversation_id
    
    async def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self._lock:
            # Check Redis first
            self._ensure_redis()
            if not self._redis.exists(self._get_conv_key(conversation_id)):
                # If not in Redis, try to load from DB and restore to Redis
                conv_data = await conversation_db.load_conversation(conversation_id)
                if not conv_data:
                    raise ValueError(f"Conversation {conversation_id} not found")
                await self._save_to_redis(conv_data)
            
            now = datetime.now()
            message = {"role": role, "content": content, "timestamp": now, "metadata": metadata or {}}
            
            # Save to Redis immediately
            pipe = self._redis.pipeline()
            pipe.expire(self._get_conv_key(conversation_id), self.ttl_seconds)
            msg_data = {"role": role, "content": content, "metadata": json.dumps(metadata or {}), "timestamp": now.isoformat()}
            pipe.zadd(self._get_msgs_key(conversation_id), {json.dumps(msg_data): now.timestamp()})
            pipe.expire(self._get_msgs_key(conversation_id), self.ttl_seconds)
            pipe.hset(self._get_conv_key(conversation_id), "updated_at", now.isoformat())
            pipe.hset(self._get_conv_key(conversation_id), "expires_at", (now + timedelta(seconds=self.ttl_seconds)).isoformat())
            pipe.execute()
            
            # Queue for background DB save
            await self._save_queue.put((conversation_id, message))
            
            logger.debug(f"Added {role} message to conversation {conversation_id} in Redis")
            return message
    
    async def get_conversation(self, conversation_id: str, include_messages: bool = True, max_messages: Optional[int] = None) -> Optional[Dict[str, Any]]:
        async with self._lock:
            # Always read from Redis first
            conv_data = await self._load_from_redis(conversation_id)
            if conv_data:
                logger.debug(f"Conversation {conversation_id} found in Redis")
                return self._format_response(conv_data, include_messages, max_messages)
            
            # If not in Redis, check Database
            logger.debug(f"Conversation {conversation_id} not in Redis, checking DB")
            conv_data = await conversation_db.load_conversation(conversation_id)
            if conv_data:
                # Restore to Redis for future requests
                await self._save_to_redis(conv_data)
                return self._format_response(conv_data, include_messages, max_messages)
            
            return None
    
    async def get_user_conversations(self, user_id: str, limit: int = 10, include_messages: bool = False) -> List[Dict[str, Any]]:
        async with self._lock:
            # Always read from Redis first
            self._ensure_redis()
            conv_ids = self._redis.lrange(self._get_user_key(user_id), 0, limit - 1)
            
            if conv_ids:
                logger.debug(f"User {user_id} conversations found in Redis")
                convs = []
                for conv_id in conv_ids:
                    conv = await self.get_conversation(conv_id, include_messages)
                    if conv:
                        convs.append(conv)
                return convs
            
            # If not in Redis, check Database
            logger.debug(f"User {user_id} conversations not in Redis, checking DB")
            convs = await conversation_db.load_user_conversations(user_id, limit)
            # Restore to Redis
            for conv in convs:
                await self._save_to_redis(conv)
            return convs
    
    async def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        async with self._lock:
            # Check Redis first
            conv_data = await self.get_conversation(conversation_id, include_messages=True)
            if not conv_data:
                return False
            
            conv_data["metadata"].update(metadata)
            conv_data["updated_at"] = datetime.now()
            
            # Update in Redis immediately
            self._ensure_redis()
            pipe = self._redis.pipeline()
            pipe.hset(self._get_conv_key(conversation_id), "metadata", json.dumps(conv_data["metadata"]))
            pipe.hset(self._get_conv_key(conversation_id), "updated_at", datetime.now().isoformat())
            pipe.execute()
            
            # Queue for background DB save
            await self._save_queue.put((conversation_id, {"action": "update_metadata", "metadata": metadata}))
            
            logger.debug(f"Updated metadata for conversation {conversation_id}")
            return True
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self._lock:
            # Check Redis first
            conv_data = await self.get_conversation(conversation_id, include_messages=True)
            if not conv_data:
                return False
            
            # Flush pending saves to DB
            await self.flush_all_to_db()
            
            # Save to DB if not already saved
            status = self._redis.get(self._get_status_key(conversation_id))
            if status != "saved":
                await conversation_db.save_conversation(conv_data)
            
            # Delete from Redis
            self._ensure_redis()
            pipe = self._redis.pipeline()
            pipe.delete(self._get_conv_key(conversation_id))
            pipe.delete(self._get_msgs_key(conversation_id))
            pipe.delete(self._get_saved_key(conversation_id))
            pipe.delete(self._get_status_key(conversation_id))
            pipe.lrem(self._get_user_key(conv_data["user_id"]), 0, conversation_id)
            pipe.execute()
            
            logger.info(f"Deleted conversation {conversation_id} from Redis")
            return True
    
    async def delete_user_conversations(self, user_id: str) -> int:
        async with self._lock:
            self._ensure_redis()
            conv_ids = self._redis.lrange(self._get_user_key(user_id), 0, -1)
            count = 0
            
            for conv_id in conv_ids:
                conv_data = await self.get_conversation(conv_id, include_messages=True)
                if conv_data:
                    status = self._redis.get(self._get_status_key(conv_id))
                    if status != "saved":
                        await conversation_db.save_conversation(conv_data)
                    
                    pipe = self._redis.pipeline()
                    pipe.delete(self._get_conv_key(conv_id))
                    pipe.delete(self._get_msgs_key(conv_id))
                    pipe.delete(self._get_saved_key(conv_id))
                    pipe.delete(self._get_status_key(conv_id))
                    pipe.execute()
                    count += 1
            
            self._redis.delete(self._get_user_key(user_id))
            await self.flush_all_to_db()
            logger.info(f"Deleted {count} conversations for user {user_id}")
            return count
    
    async def get_conversation_history(self, conversation_id: str, max_messages: int = 20, format_type: str = "list") -> List:
        conv = await self.get_conversation(conversation_id, include_messages=True, max_messages=max_messages)
        if not conv:
            return []
        
        messages_list = conv.get("messages", [])
        if format_type == "string":
            return [f"{msg['role']}: {msg['content']}" for msg in messages_list]
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages_list]
    
    async def clear_expired(self) -> int:
        self._ensure_redis()
        pattern = "conv:*"
        cursor = 0
        deleted = 0
        
        while True:
            cursor, keys = self._redis.get_client().scan(cursor, match=pattern, count=100)
            for key in keys:
                if ":" in key:
                    conv_id = key.split(":")[1]
                    created_at_str = self._redis.hget(key, "created_at")
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str)
                        age = (datetime.now() - created_at).total_seconds()
                        
                        if age > self.delete_after_seconds:
                            status = self._redis.get(self._get_status_key(conv_id))
                            if status == "saved":
                                user_id = self._redis.hget(key, "user_id")
                                pipe = self._redis.pipeline()
                                pipe.delete(key)
                                pipe.delete(self._get_msgs_key(conv_id))
                                pipe.delete(self._get_saved_key(conv_id))
                                pipe.delete(self._get_status_key(conv_id))
                                if user_id:
                                    pipe.lrem(self._get_user_key(user_id), 0, conv_id)
                                pipe.execute()
                                deleted += 1
            if cursor == 0:
                break
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired conversations from Redis")
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        self._ensure_redis()
        pattern = "conv:*"
        cursor = 0
        conv_count = 0
        msg_count = 0
        
        while True:
            cursor, keys = self._redis.get_client().scan(cursor, match=pattern, count=100)
            for key in keys:
                if ":" in key:
                    conv_id = key.split(":")[1]
                    if conv_id:
                        conv_count += 1
                        msg_count += self._redis.zcard(self._get_msgs_key(conv_id))
            if cursor == 0:
                break
        
        return {
            "total_conversations": conv_count,
            "total_messages": msg_count,
            "ttl_hours": self.ttl_seconds // 3600,
            "delete_after_hours": self.delete_after_seconds // 3600,
            "queue_size": self._save_queue.qsize(),
            "redis_connected": self._redis_initialized and self._redis.ping()
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
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=3600)
                    break
                except asyncio.TimeoutError:
                    removed = await self.clear_expired()
                    if removed > 0:
                        logger.info(f"Cleanup job removed {removed} expired conversations")
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
        self._ensure_redis()
        pattern = "conv:*"
        cursor = 0
        saved = 0
        
        while True:
            cursor, keys = self._redis.get_client().scan(cursor, match=pattern, count=100)
            for key in keys:
                if ":" in key:
                    conv_id = key.split(":")[1]
                    if conv_id:
                        conv_data = await self._load_from_redis(conv_id)
                        if conv_data:
                            if await conversation_db.save_conversation(conv_data):
                                saved += 1
            if cursor == 0:
                break
        
        logger.info(f"Saved {saved} conversations to DB during shutdown")
        return saved
    
    async def close(self):
        self._redis.close()

cache = ConversationCache(ttl_hours=2)

async def initialize_cache():
    await cache.start_background_tasks()
    cache._ensure_redis()