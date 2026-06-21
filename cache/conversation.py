from logger.logger import get_logger
logger = get_logger(__name__)
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import uuid

class ConversationCache:
    def __init__(self, ttl_hours: int = 2):
        self.ttl_seconds = ttl_hours * 3600
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._user_conversations: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
    def _clean_expired(self) -> None:
        current_time = datetime.now()
        expired_keys = []
        for conv_id, conv_data in self._conversations.items():
            if conv_data["expires_at"] < current_time:
                expired_keys.append(conv_id)
        for conv_id in expired_keys:
            self._remove_conversation(conv_id)
    
    def _remove_conversation(self, conv_id: str) -> None:
        if conv_id in self._conversations:
            user_id = self._conversations[conv_id]["user_id"]
            if user_id in self._user_conversations and conv_id in self._user_conversations[user_id]:
                self._user_conversations[user_id].remove(conv_id)
            del self._conversations[conv_id]
    
    async def create_conversation(self, user_id: str, conversation_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        async with self._lock:
            self._clean_expired()
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())
            if conversation_id in self._conversations:
                raise ValueError(f"Conversation {conversation_id} already exists")
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
            logger.info(f"Added {role} message to conversation {conversation_id}")
            return message
    
    async def get_conversation(self, conversation_id: str, include_messages: bool = True, max_messages: Optional[int] = None) -> Optional[Dict[str, Any]]:
        async with self._lock:
            self._clean_expired()
            if conversation_id not in self._conversations:
                return None
            conv = self._conversations[conversation_id].copy()
            conv["expires_at"] = conv["expires_at"].isoformat() if isinstance(conv["expires_at"], datetime) else conv["expires_at"]
            conv["created_at"] = conv["created_at"].isoformat() if isinstance(conv["created_at"], datetime) else conv["created_at"]
            conv["updated_at"] = conv["updated_at"].isoformat() if isinstance(conv["updated_at"], datetime) else conv["updated_at"]
            if not include_messages:
                conv.pop("messages", None)
            elif max_messages and conv.get("messages"):
                conv["messages"] = conv["messages"][-max_messages:]
                for msg in conv["messages"]:
                    if isinstance(msg["timestamp"], datetime):
                        msg["timestamp"] = msg["timestamp"].isoformat()
            return conv
    
    async def get_user_conversations(self, user_id: str, limit: int = 10, include_messages: bool = False) -> List[Dict[str, Any]]:
        async with self._lock:
            self._clean_expired()
            conversations = []
            for conv_id in self._user_conversations.get(user_id, [])[-limit:]:
                conv = await self.get_conversation(conv_id, include_messages)
                if conv:
                    conversations.append(conv)
            return conversations
    
    async def update_conversation_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        async with self._lock:
            if conversation_id not in self._conversations:
                return False
            conv = self._conversations[conversation_id]
            conv["metadata"].update(metadata)
            conv["updated_at"] = datetime.now()
            conv["expires_at"] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            logger.info(f"Updated metadata for conversation {conversation_id}")
            return True
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self._lock:
            if conversation_id not in self._conversations:
                return False
            self._remove_conversation(conversation_id)
            logger.info(f"Deleted conversation {conversation_id}")
            return True
    
    async def delete_user_conversations(self, user_id: str) -> int:
        async with self._lock:
            conversations = self._user_conversations.get(user_id, []).copy()
            count = 0
            for conv_id in conversations:
                if self._remove_conversation(conv_id):
                    count += 1
            if user_id in self._user_conversations:
                del self._user_conversations[user_id]
            logger.info(f"Deleted {count} conversations for user {user_id}")
            return count
    
    async def get_conversation_history(self, conversation_id: str, max_messages: int = 20, format_type: str = "list") -> List:
        conv = await self.get_conversation(conversation_id, include_messages=True, max_messages=max_messages)
        if not conv:
            return []
        messages = conv.get("messages", [])
        if format_type == "string":
            formatted = []
            for msg in messages:
                content = msg["content"] if isinstance(msg, dict) else msg.content
                role = msg["role"] if isinstance(msg, dict) else msg.role
                formatted.append(f"{role}: {content}")
            return formatted
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    
    async def clear_expired(self) -> int:
        async with self._lock:
            before_count = len(self._conversations)
            self._clean_expired()
            after_count = len(self._conversations)
            removed = before_count - after_count
            logger.info(f"Cleared {removed} expired conversations")
            return removed
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            self._clean_expired()
            total_messages = sum(len(conv.get("messages", [])) for conv in self._conversations.values())
            return {
                "total_conversations": len(self._conversations),
                "total_users": len(self._user_conversations),
                "total_messages": total_messages,
                "ttl_hours": self.ttl_seconds // 3600
            }

cache = ConversationCache()