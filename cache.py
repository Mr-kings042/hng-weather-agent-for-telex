import os
import json
import logging
from typing import Optional, Any
import socket
import time
import subprocess
from urllib.parse import urlparse

try:
    import redis.asyncio as redis  # type: ignore
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

# safe alias for redis connection exception when redis is not installed
RedisConnectionError = getattr(redis, "ConnectionError", ConnectionError)

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass


class Cache:
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("REDIS_URL", "")
        # use Any for runtime compatibility even if redis is not installed
        self._redis: Optional[Any] = None
        self._local: dict[str, Any] = {}
        self._connected: bool = False

        if self.url and not REDIS_AVAILABLE:
            logger.warning(
                "Redis URL provided but redis.asyncio not available. "
                "Install with: pip install redis[hiredis]. Falling back to in-memory cache."
            )

    async def connect(self) -> None:
        if not self.url:
            logger.info("No REDIS_URL configured. Using in-memory cache.")
            return

        if not REDIS_AVAILABLE:
            logger.warning("redis.asyncio not installed. Using in-memory cache.")
            return

        # Probe host:port to avoid noisy stack traces when host is unreachable
        try:
            parsed = urlparse(self.url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            with socket.create_connection((host, port), timeout=1):
                logger.debug(f"Redis reachable at {host}:{port} (probe)")
        except Exception:
            autostart = os.environ.get("REDIS_AUTOSTART", "").lower() in ("1", "true", "yes")
            if autostart:
                logger.info(f"Redis not reachable at {self.url}. Attempting to start local Redis via Docker...")
                container_name = os.environ.get("REDIS_DOCKER_NAME", "stage_three_redis")
                try:
                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "--name",
                            container_name,
                            "-p",
                            f"{port}:6379",
                            "-d",
                            "--restart",
                            "unless-stopped",
                            "redis:7",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # wait for container to become reachable
                    start = time.time()
                    timeout = int(os.environ.get("REDIS_AUTOSTART_WAIT", "10"))
                    while time.time() - start < timeout:
                        try:
                            with socket.create_connection((host, port), timeout=1):
                                logger.info(f"Redis started and reachable at {host}:{port}")
                                break
                        except Exception:
                            time.sleep(0.5)
                    else:
                        logger.warning("Redis Docker container started but port did not become reachable in time.")
                        logger.info("Falling back to in-memory cache.")
                        return
                except Exception as e:
                    logger.warning(f"Failed to auto-start Redis via Docker: {e}. Falling back to in-memory cache.")
                    return
            else:
                logger.info(f"Redis host unreachable at {self.url}. Skipping Redis and using in-memory cache.")
                return

        try:
            self._redis = redis.from_url(  # type: ignore[arg-type]
                self.url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )

            # Verify connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Successfully connected to Redis: {self._mask_url(self.url)}")

        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}. Using in-memory cache.")
            self._redis = None
            self._connected = False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}. Using in-memory cache.")
            self._redis = None
            self._connected = False

    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        """
        if not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._connected = False
            return False

    @staticmethod
    def _mask_url(url: str) -> str:
        """
        Mask sensitive information in Redis URL for logging.
        """
        if "@" in url:
            parts = url.split("@")
            return f"redis://***@{parts[-1]}"
        return url

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache (Redis first, fallback to local).
        """
        if self._redis and self._connected:
            try:
                val = await self._redis.get(key)
                if not val:
                    logger.debug(f"Cache MISS (Redis): {key}")
                    return None

                logger.debug(f"Cache HIT (Redis): {key}")
                try:
                    return json.loads(val)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to deserialize cached value for key '{key}': {e}")
                    # Delete corrupted cache entry
                    await self.delete(key)
                    return None

            except RedisConnectionError as e:
                logger.warning(f"Redis connection error on GET: {e}. Falling back to local cache.")
                self._connected = False
                # fall through to local cache
            except Exception as e:
                logger.error(f"Error getting key '{key}' from Redis: {e}")
                return None

        # local cache fallback
        result = self._local.get(key)
        if result is not None:
            logger.debug(f"Cache HIT (local): {key}")
        else:
            logger.debug(f"Cache MISS (local): {key}")
        return result

    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """
        Store value in cache with optional expiration.
        """
        if self._redis and self._connected:
            try:
                payload = json.dumps(value)

                if nx:
                    result = await self._redis.set(key, payload, ex=ex, nx=True)
                    success = result is not None
                elif ex:
                    await self._redis.set(key, payload, ex=ex)
                    success = True
                else:
                    await self._redis.set(key, payload)
                    success = True

                if success:
                    logger.debug(f"Cache SET (Redis): {key} (TTL: {ex}s)" if ex else f"Cache SET (Redis): {key}")
                return success

            except RedisConnectionError as e:
                logger.warning(f"Redis connection error on SET: {e}. Falling back to local cache.")
                self._connected = False
                # fall through to local cache
            except Exception as e:
                logger.error(f"Error setting key '{key}' in Redis: {e}")
                return False

        # local cache fallback (no TTL support)
        self._local[key] = value
        logger.debug(f"Cache SET (local): {key}")
        return True

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        """
        if self._redis and self._connected:
            try:
                result = await self._redis.delete(key)
                logger.debug(f"Cache DELETE (Redis): {key}")
                return result > 0
            except Exception as e:
                logger.error(f"Error deleting key '{key}' from Redis: {e}")
                return False

        if key in self._local:
            del self._local[key]
            logger.debug(f"Cache DELETE (local): {key}")
            return True
        return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        """
        if self._redis and self._connected:
            try:
                result = await self._redis.exists(key)
                return result > 0
            except Exception as e:
                logger.error(f"Error checking existence of key '{key}': {e}")
                return False

        return key in self._local

    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching pattern (or all if no pattern).
        """
        if self._redis and self._connected:
            try:
                if pattern:
                    keys = []
                    async for key in self._redis.scan_iter(match=pattern):
                        keys.append(key)

                    if keys:
                        deleted = await self._redis.delete(*keys)
                        logger.info(f"Deleted {deleted} keys matching pattern '{pattern}'")
                        return deleted
                    return 0
                else:
                    await self._redis.flushdb()
                    logger.warning("Cleared entire Redis database")
                    return -1
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return 0

        # local cache clear
        if pattern:
            import fnmatch
            keys_to_delete = [k for k in self._local.keys() if fnmatch.fnmatch(k, pattern)]
            for key in keys_to_delete:
                del self._local[key]
            logger.info(f"Deleted {len(keys_to_delete)} keys from local cache")
            return len(keys_to_delete)
        else:
            count = len(self._local)
            self._local.clear()
            logger.info(f"Cleared {count} keys from local cache")
            return count

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        """
        stats = {
            "backend": "redis" if (self._redis and self._connected) else "local",
            "connected": self._connected,
            "redis_available": REDIS_AVAILABLE,
        }

        if self._redis and self._connected:
            try:
                info = await self._redis.info()
                stats.update(
                    {
                        "redis_version": info.get("redis_version"),
                        "used_memory": info.get("used_memory_human"),
                        "total_keys": await self._redis.dbsize(),
                    }
                )
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        else:
            stats["total_keys"] = len(self._local)

        return stats

    async def close(self) -> None:
        """
        Close Redis connection and cleanup resources.
        """
        if self._redis:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._redis = None
                self._connected = False

        # Clear local cache
        self._local.clear()

    @property
    def is_redis(self) -> bool:
        """Check if using Redis backend"""
        return self._redis is not None and self._connected

    @property
    def is_local(self) -> bool:
        """Check if using local/in-memory backend"""
        return not self.is_redis


cache = Cache()
