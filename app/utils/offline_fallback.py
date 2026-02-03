"""
Offline Fallback Utilities
Provides resilience for low-bandwidth/offline scenarios
"""
from typing import Dict, Optional, Callable, Any
import logging
import time

logger = logging.getLogger(__name__)

async def with_offline_fallback(
    primary_func: Callable,
    fallback_func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Execute primary function with fallback to secondary function on failure
    
    Args:
        primary_func: Primary function to try first
        fallback_func: Fallback function if primary fails
        *args, **kwargs: Arguments to pass to functions
        
    Returns:
        Result from primary or fallback function
    """
    try:
        return await primary_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Primary function failed: {e}. Using fallback.")
        try:
            return await fallback_func(*args, **kwargs)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            raise

def create_offline_response(message: str, language: str = "hi") -> Dict:
    """
    Create a standardized offline response message
    
    Args:
        message: Message text
        language: Language code
        
    Returns:
        Standardized response dictionary
    """
    return {
        "message": message,
        "language": language,
        "offline_mode": True,
        "timestamp": None  # Can be added if needed
    }

class OfflineCache:
    """
    Enhanced in-memory cache for offline resilience with TTL
    Stores recent queries and responses for offline access
    """
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        # Check if entry has expired
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp"""
        # Clean expired entries
        self._clean_expired()
        
        # Remove oldest entries if cache is full
        while len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def _clean_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry["timestamp"] > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        self._clean_expired()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

# Global offline cache instance with increased size for better performance
_offline_cache = OfflineCache(max_size=500, ttl_seconds=3600)  # 1 hour TTL

def get_offline_cache() -> OfflineCache:
    """Get global offline cache instance"""
    return _offline_cache
