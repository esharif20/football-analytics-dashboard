"""
Video Hash Caching Utility

Computes SHA256 hash of video files and manages cache lookups
to skip re-processing identical videos with the same model config.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def compute_video_hash(video_path: str | Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a video file.
    
    Uses chunked reading to handle large files efficiently.
    
    Args:
        video_path: Path to the video file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        SHA256 hash as hex string
    """
    sha256 = hashlib.sha256()
    
    with open(video_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def get_cache_key(video_hash: str, mode: str, model_config: Dict[str, str]) -> str:
    """
    Generate a unique cache key from video hash + mode + model config.
    
    Args:
        video_hash: SHA256 hash of the video
        mode: Pipeline mode (all, radar, team, etc.)
        model_config: Dictionary of model settings
        
    Returns:
        16-character cache key
    """
    # Sort model config for consistent hashing
    config_str = json.dumps(model_config, sort_keys=True)
    combined = f"{video_hash}:{mode}:{config_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache entry."""
    return CACHE_DIR / f"{cache_key}.json"


def check_cache(video_hash: str, mode: str, model_config: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Check if cached results exist for this video + mode + config.
    
    Args:
        video_hash: SHA256 hash of the video
        mode: Pipeline mode
        model_config: Model configuration
        
    Returns:
        Cached results dict if found, None otherwise
    """
    cache_key = get_cache_key(video_hash, mode, model_config)
    cache_path = get_cache_path(cache_key)
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                
            # Verify the cache entry matches
            if (data.get("video_hash") == video_hash and 
                data.get("mode") == mode and
                data.get("model_config") == model_config):
                return data.get("results")
        except (json.JSONDecodeError, KeyError):
            # Invalid cache entry, remove it
            cache_path.unlink(missing_ok=True)
    
    return None


def save_to_cache(
    video_hash: str, 
    mode: str, 
    model_config: Dict[str, str],
    results: Dict[str, Any]
) -> str:
    """
    Save processing results to cache.
    
    Args:
        video_hash: SHA256 hash of the video
        mode: Pipeline mode
        model_config: Model configuration
        results: Processing results to cache
        
    Returns:
        Cache key
    """
    cache_key = get_cache_key(video_hash, mode, model_config)
    cache_path = get_cache_path(cache_key)
    
    cache_entry = {
        "video_hash": video_hash,
        "mode": mode,
        "model_config": model_config,
        "results": results,
        "cached_at": __import__("datetime").datetime.now().isoformat(),
    }
    
    with open(cache_path, "w") as f:
        json.dump(cache_entry, f, indent=2)
    
    return cache_key


def clear_cache(video_hash: Optional[str] = None) -> int:
    """
    Clear cache entries.
    
    Args:
        video_hash: If provided, only clear entries for this video.
                   If None, clear all cache entries.
                   
    Returns:
        Number of entries cleared
    """
    count = 0
    
    for cache_file in CACHE_DIR.glob("*.json"):
        if video_hash:
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                if data.get("video_hash") == video_hash:
                    cache_file.unlink()
                    count += 1
            except:
                pass
        else:
            cache_file.unlink()
            count += 1
    
    return count


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache stats
    """
    cache_files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    return {
        "entries": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR),
    }


# Integration with main pipeline
def process_with_cache(
    video_path: str | Path,
    mode: str,
    model_config: Dict[str, str],
    process_fn: callable
) -> Dict[str, Any]:
    """
    Process a video with caching support.
    
    Args:
        video_path: Path to the video file
        mode: Pipeline mode
        model_config: Model configuration
        process_fn: Function to call if cache miss
        
    Returns:
        Processing results (from cache or fresh)
    """
    video_hash = compute_video_hash(video_path)
    
    # Check cache first
    cached = check_cache(video_hash, mode, model_config)
    if cached:
        print(f"[Cache] HIT - Using cached results for {video_hash[:8]}...")
        return cached
    
    # Cache miss - run processing
    print(f"[Cache] MISS - Processing video {video_hash[:8]}...")
    results = process_fn()
    
    # Save to cache
    cache_key = save_to_cache(video_hash, mode, model_config, results)
    print(f"[Cache] Saved results to cache: {cache_key}")
    
    return results
