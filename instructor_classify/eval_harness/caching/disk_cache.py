"""
Disk-based cache implementation.

This module provides a persistent cache that stores data on disk,
allowing for resilience against failures and resuming interrupted evaluations.
"""

import os
import json
import hashlib
import pickle
from typing import Any, Dict, Optional
from pathlib import Path
from instructor_classify.eval_harness.base import Cache
from rich.console import Console


class DiskCache(Cache):
    """
    Disk-based cache that stores data in files.
    
    This cache implementation stores cached data on disk, providing persistence
    across runs and allowing for recovery from failures.
    """
    
    def __init__(self, cache_dir: str = ".cache", use_pickle: bool = False):
        """
        Initialize the disk cache.
        
        Parameters
        ----------
        cache_dir : str
            Directory to store cache files
        use_pickle : bool
            Whether to use pickle for serialization (more flexible but less safe)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_pickle = use_pickle
        self.console = Console()
        
        # Track cache stats
        self.hits = 0
        self.misses = 0
    
    def _hash_key(self, key: str) -> str:
        """
        Hash a key to create a filename.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        str
            Hashed key suitable for a filename
        """
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get the path for a cache file.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Path
            Path to the cache file
        """
        hashed_key = self._hash_key(key)
        return self.cache_dir / f"{hashed_key}.{'pkl' if self.use_pickle else 'json'}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Optional[Any]
            Cached value, or None if not found
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.misses += 1
            return None
        
        try:
            if self.use_pickle:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
            else:
                with open(cache_path, 'r') as f:
                    value = json.load(f)
            
            self.hits += 1
            return value
        except Exception as e:
            self.console.print(f"[yellow]Cache read error for {key}: {str(e)}[/yellow]")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if self.use_pickle:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(value, f, default=lambda o: str(o) if isinstance(o, object) else o)
        except Exception as e:
            self.console.print(f"[yellow]Cache write error for {key}: {str(e)}[/yellow]")
    
    def clear(self) -> None:
        """Clear all cached values."""
        for cache_file in self.cache_dir.glob("*"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.console.print(f"[yellow]Error deleting cache file {cache_file}: {str(e)}[/yellow]")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(list(self.cache_dir.glob("*"))),
            "cache_dir": str(self.cache_dir),
        }
        
        return stats
    
    def generate_key(self, model: str, text: str, is_multi: bool = False) -> str:
        """
        Generate a cache key for a prediction.
        
        Parameters
        ----------
        model : str
            Model name
        text : str
            Text to predict
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        str
            Cache key
        """
        # Include enough information to uniquely identify the prediction
        key_parts = [
            model,
            text,
            "multi" if is_multi else "single"
        ]
        
        return hashlib.md5("||".join(key_parts).encode()).hexdigest()


class CachedClassifier:
    """
    Decorator for classifiers that adds caching.
    
    This wraps a classifier to add caching of prediction results.
    """
    
    def __init__(self, classifier: Any, cache: Cache):
        """
        Initialize the cached classifier.
        
        Parameters
        ----------
        classifier : Any
            The classifier to wrap
        cache : Cache
            The cache to use
        """
        self.classifier = classifier
        self.cache = cache
        self.model_name = getattr(classifier, "model_name", "unknown")
    
    def predict(self, text: str):
        """
        Predict with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, False)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = self.classifier.predict(text)
        self.cache.set(key, result)
        return result
    
    def predict_multi(self, text: str):
        """
        Predict multi-label with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, True)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = self.classifier.predict_multi(text)
        self.cache.set(key, result)
        return result
    
    def batch_predict(self, texts, n_jobs=None):
        """
        Batch predict with caching.
        
        Parameters
        ----------
        texts : List[str]
            Texts to predict
        n_jobs : Optional[int]
            Number of jobs (ignored, for compatibility)
            
        Returns
        -------
        List[Any]
            Prediction results
        """
        # Use batch method from underlying classifier if available and no cached results
        if hasattr(self.classifier, "batch_predict"):
            return self.classifier.batch_predict(texts, n_jobs)
        
        # Otherwise, fall back to individual predictions with caching
        return [self.predict(text) for text in texts]
    
    def batch_predict_multi(self, texts, n_jobs=None):
        """
        Batch predict multi-label with caching.
        
        Parameters
        ----------
        texts : List[str]
            Texts to predict
        n_jobs : Optional[int]
            Number of jobs (ignored, for compatibility)
            
        Returns
        -------
        List[Any]
            Prediction results
        """
        # Use batch method from underlying classifier if available and no cached results
        if hasattr(self.classifier, "batch_predict_multi"):
            return self.classifier.batch_predict_multi(texts, n_jobs)
        
        # Otherwise, fall back to individual predictions with caching
        return [self.predict_multi(text) for text in texts]


class AsyncCachedClassifier:
    """
    Decorator for async classifiers that adds caching.
    
    This wraps an async classifier to add caching of prediction results.
    """
    
    def __init__(self, classifier: Any, cache: Cache):
        """
        Initialize the cached async classifier.
        
        Parameters
        ----------
        classifier : Any
            The async classifier to wrap
        cache : Cache
            The cache to use
        """
        self.classifier = classifier
        self.cache = cache
        self.model_name = getattr(classifier, "model_name", "unknown")
    
    async def predict(self, text: str):
        """
        Predict with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, False)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = await self.classifier.predict(text)
        self.cache.set(key, result)
        return result
    
    async def predict_multi(self, text: str):
        """
        Predict multi-label with caching.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        key = self.cache.generate_key(self.model_name, text, True)
        cached = self.cache.get(key)
        
        if cached is not None:
            return cached
        
        result = await self.classifier.predict_multi(text)
        self.cache.set(key, result)
        return result