from logger.logger import get_logger
from dotenv import load_dotenv
import os
import redis
import json
from typing import Optional, Any, Dict, List

logger = get_logger(__name__)
load_dotenv()

class RedisClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.client = None
    
    def connect(self):
        if self.client is None:
            try:
                self.client = redis.Redis(
                    host=os.getenv("REDIS_HOST"),
                    port=int(os.getenv("REDIS_PORT")),
                    decode_responses=True,
                    username="default",
                    password=os.getenv("REDIS_PASSWORD"),
                )
                self.client.ping()
                logger.info("Redis connection established successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self.client
    
    def get_client(self):
        if self.client is None:
            self.connect()
        return self.client
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self.get_client().set(key, value)
            if ttl:
                self.get_client().expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.get_client().get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    def delete(self, *keys) -> int:
        try:
            return self.get_client().delete(*keys)
        except Exception as e:
            logger.error(f"Error deleting keys: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        try:
            return self.get_client().exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        try:
            return self.get_client().expire(key, ttl)
        except Exception as e:
            logger.error(f"Error setting expire for {key}: {e}")
            return False
    
    def hset(self, key: str, mapping: Dict[str, Any]) -> int:
        try:
            return self.get_client().hset(key, mapping=mapping)
        except Exception as e:
            logger.error(f"Error in hset for {key}: {e}")
            return 0
    
    def hgetall(self, key: str) -> Dict[str, str]:
        try:
            return self.get_client().hgetall(key)
        except Exception as e:
            logger.error(f"Error in hgetall for {key}: {e}")
            return {}
    
    def hget(self, key: str, field: str) -> Optional[str]:
        try:
            return self.get_client().hget(key, field)
        except Exception as e:
            logger.error(f"Error in hget for {key}: {e}")
            return None
    
    def hdel(self, key: str, *fields) -> int:
        try:
            return self.get_client().hdel(key, *fields)
        except Exception as e:
            logger.error(f"Error in hdel for {key}: {e}")
            return 0
    
    def lpush(self, key: str, *values) -> int:
        try:
            return self.get_client().lpush(key, *values)
        except Exception as e:
            logger.error(f"Error in lpush for {key}: {e}")
            return 0
    
    def lrem(self, key: str, count: int, value: str) -> int:
        try:
            return self.get_client().lrem(key, count, value)
        except Exception as e:
            logger.error(f"Error in lrem for {key}: {e}")
            return 0
    
    def lrange(self, key: str, start: int, end: int) -> List[str]:
        try:
            return self.get_client().lrange(key, start, end)
        except Exception as e:
            logger.error(f"Error in lrange for {key}: {e}")
            return []
    
    def ltrim(self, key: str, start: int, end: int) -> bool:
        try:
            return self.get_client().ltrim(key, start, end)
        except Exception as e:
            logger.error(f"Error in ltrim for {key}: {e}")
            return False
    
    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        try:
            return self.get_client().zadd(key, mapping)
        except Exception as e:
            logger.error(f"Error in zadd for {key}: {e}")
            return 0
    
    def zrange(self, key: str, start: int, end: int, withscores: bool = False) -> List:
        try:
            return self.get_client().zrange(key, start, end, withscores=withscores)
        except Exception as e:
            logger.error(f"Error in zrange for {key}: {e}")
            return []
    
    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        try:
            return self.get_client().zremrangebyscore(key, min_score, max_score)
        except Exception as e:
            logger.error(f"Error in zremrangebyscore for {key}: {e}")
            return 0
    
    def zcard(self, key: str) -> int:
        try:
            return self.get_client().zcard(key)
        except Exception as e:
            logger.error(f"Error in zcard for {key}: {e}")
            return 0
    
    def pipeline(self):
        try:
            return self.get_client().pipeline()
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return None
    
    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Redis connection closed")
    
    def ping(self) -> bool:
        try:
            if self.client is None:
                return False
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

redis_client = RedisClient()