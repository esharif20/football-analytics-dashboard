"""Stub/cache management utilities.

This module provides caching functionality for the pipeline to avoid
re-processing the same video multiple times. Stubs are mode-dependent:
- Player detection/tracking/team classification share the same stub
- Ball detection has its own stub
- Each mode can be cached independently

Stub files are stored in the STUB_DIR and named as:
    {video_stem}_{stub_key}.pkl
"""

import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import STUB_DIR


def get_video_hash(video_path: str) -> str:
    """Generate a content-based hash of the video file.

    Reads the first 8 MB + file size so identical clips get the same
    hash even when uploaded under different filenames.

    Args:
        video_path: Path to the video file

    Returns:
        Hash string for the video
    """
    path = Path(video_path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        # First 8 MB is enough to identify the clip
        h.update(f.read(8 * 1024 * 1024))
    return h.hexdigest()[:16]


def stub_path(source_video_path: str, mode: "Mode") -> Path:
    """Generate stub file path for caching.

    Uses a content hash so the same video re-uploaded under a different
    filename still hits the cache.

    Args:
        source_video_path: Path to source video
        mode: Pipeline mode enum

    Returns:
        Path to stub file
    """
    # Import here to avoid circular imports
    from pipeline import Mode

    STUB_DIR.mkdir(parents=True, exist_ok=True)
    video_hash = get_video_hash(source_video_path)
    if mode in {Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        stub_key = "people_tracks"
    elif mode == Mode.BALL_DETECTION:
        stub_key = "ball_tracks"
    else:
        stub_key = mode.value.lower()
    return STUB_DIR / f"{video_hash}_{stub_key}.pkl"


def stub_paths_for_mode(source_video_path: str, mode: "Mode") -> List[Path]:
    """Get all stub paths relevant to a pipeline mode.

    Args:
        source_video_path: Path to source video
        mode: Pipeline mode enum

    Returns:
        List of relevant stub paths
    """
    # Import here to avoid circular imports
    from pipeline import Mode

    ball_stub = stub_path(source_video_path, Mode.BALL_DETECTION)
    ball_full_stub = ball_stub.with_name(f"{ball_stub.stem}_full{ball_stub.suffix}")
    if mode == Mode.ALL:
        return [
            stub_path(source_video_path, Mode.TEAM_CLASSIFICATION),
            ball_stub,
            ball_full_stub,
        ]
    if mode == Mode.BALL_DETECTION:
        return [ball_stub, ball_full_stub]
    if mode in {Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        return [stub_path(source_video_path, Mode.TEAM_CLASSIFICATION)]
    return []


def clear_stubs(paths: List[Path]) -> None:
    """Delete specified stub files.

    Args:
        paths: List of stub file paths to delete
    """
    for stub in paths:
        if stub.exists():
            stub.unlink()
            print(f"Deleted stub: {stub}")


def load_stub(stub_file: Path) -> Optional[Dict[str, Any]]:
    """Load a stub file if it exists.
    
    Args:
        stub_file: Path to the stub file
        
    Returns:
        Loaded data or None if file doesn't exist
    """
    if stub_file.exists():
        try:
            with open(stub_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load stub {stub_file}: {e}")
            return None
    return None


def save_stub(stub_file: Path, data: Dict[str, Any], video_path: str = None) -> None:
    """Save data to a stub file with metadata.
    
    Args:
        stub_file: Path to save the stub
        data: Data to cache
        video_path: Optional video path for hash validation
    """
    stub_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'video_hash': get_video_hash(video_path) if video_path else None,
    }
    
    save_data = {
        '_metadata': metadata,
        'data': data,
    }
    
    with open(stub_file, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved stub: {stub_file}")


def check_stub_status(source_video_path: str) -> Dict[str, Any]:
    """Check the status of all stubs for a video.
    
    Args:
        source_video_path: Path to the video file
        
    Returns:
        Dictionary with stub status information
    """
    from pipeline import Mode
    
    stem = Path(source_video_path).stem
    video_hash = get_video_hash(source_video_path)
    
    status = {
        'video': stem,
        'video_hash': video_hash,
        'stubs': {},
    }
    
    # Check each possible stub
    stub_types = [
        ('people_tracks', [Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION]),
        ('ball_tracks', [Mode.BALL_DETECTION]),
        ('all', [Mode.ALL]),
        ('radar', [Mode.RADAR]),
    ]
    
    for stub_key, modes in stub_types:
        stub_file = STUB_DIR / f"{stem}_{stub_key}.pkl"
        if stub_file.exists():
            try:
                data = load_stub(stub_file)
                if data and '_metadata' in data:
                    metadata = data['_metadata']
                    status['stubs'][stub_key] = {
                        'exists': True,
                        'created_at': metadata.get('created_at'),
                        'valid': metadata.get('video_hash') == video_hash,
                        'size_mb': stub_file.stat().st_size / (1024 * 1024),
                        'modes': [m.value for m in modes],
                    }
                else:
                    # Legacy stub without metadata
                    status['stubs'][stub_key] = {
                        'exists': True,
                        'created_at': None,
                        'valid': True,  # Assume valid for legacy
                        'size_mb': stub_file.stat().st_size / (1024 * 1024),
                        'modes': [m.value for m in modes],
                    }
            except Exception:
                status['stubs'][stub_key] = {
                    'exists': True,
                    'valid': False,
                    'error': 'Failed to read stub',
                }
        else:
            status['stubs'][stub_key] = {
                'exists': False,
                'modes': [m.value for m in modes],
            }
    
    return status


def estimate_processing_time(
    source_video_path: str,
    mode: "Mode",
    device: str = "cpu"
) -> Tuple[float, Dict[str, float]]:
    """Estimate processing time based on video length and cached stubs.
    
    Args:
        source_video_path: Path to the video file
        mode: Pipeline mode
        device: Processing device (cpu, cuda, mps)
        
    Returns:
        Tuple of (total_seconds, breakdown_by_stage)
    """
    import cv2
    from pipeline import Mode
    
    # Get video info
    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration_sec = frame_count / fps
    
    # Base times per second of video (on CPU)
    base_times = {
        'detection': 2.0,      # 2 sec per video sec
        'tracking': 0.1,       # Fast
        'team_classification': 1.0,
        'ball_detection': 1.5,
        'pitch_detection': 0.5,
        'homography': 0.1,
        'analytics': 0.05,
        'rendering': 1.0,
    }
    
    # Device multipliers
    device_multipliers = {
        'cpu': 1.0,
        'mps': 0.4,   # Apple Silicon ~2.5x faster
        'cuda': 0.2,  # CUDA ~5x faster
    }
    multiplier = device_multipliers.get(device.lower(), 1.0)
    
    # Check which stubs exist
    stub_status = check_stub_status(source_video_path)
    
    breakdown = {}
    total = 0.0
    
    # Calculate time for each stage based on mode and cached stubs
    if mode in {Mode.ALL, Mode.PLAYER_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION}:
        if stub_status['stubs'].get('people_tracks', {}).get('exists'):
            breakdown['detection'] = 0.5  # Just loading
            breakdown['tracking'] = 0.0
            breakdown['team_classification'] = 0.0
        else:
            breakdown['detection'] = base_times['detection'] * duration_sec * multiplier
            breakdown['tracking'] = base_times['tracking'] * duration_sec * multiplier
            breakdown['team_classification'] = base_times['team_classification'] * duration_sec * multiplier
    
    if mode in {Mode.ALL, Mode.BALL_DETECTION}:
        if stub_status['stubs'].get('ball_tracks', {}).get('exists'):
            breakdown['ball_detection'] = 0.5
        else:
            breakdown['ball_detection'] = base_times['ball_detection'] * duration_sec * multiplier
    
    if mode in {Mode.ALL, Mode.PITCH_DETECTION}:
        breakdown['pitch_detection'] = base_times['pitch_detection'] * duration_sec * multiplier
        breakdown['homography'] = base_times['homography'] * duration_sec * multiplier
    
    if mode == Mode.ALL:
        breakdown['analytics'] = base_times['analytics'] * duration_sec
        breakdown['rendering'] = base_times['rendering'] * duration_sec * multiplier
    
    if mode == Mode.RADAR:
        # Radar mode needs people + ball + pitch but no annotated video
        for key in ['detection', 'tracking', 'team_classification', 'ball_detection', 'pitch_detection', 'homography']:
            if key not in breakdown:
                breakdown[key] = base_times.get(key, 0) * duration_sec * multiplier
        breakdown['rendering'] = base_times['rendering'] * duration_sec * multiplier * 0.5  # Radar is simpler
    
    total = sum(breakdown.values())
    
    return total, breakdown


def get_cache_summary(source_video_path: str) -> str:
    """Get a human-readable summary of cache status.
    
    Args:
        source_video_path: Path to the video file
        
    Returns:
        Formatted string summary
    """
    status = check_stub_status(source_video_path)
    
    lines = [f"Cache status for: {status['video']}"]
    lines.append("-" * 40)
    
    for stub_key, info in status['stubs'].items():
        if info.get('exists'):
            valid = "✓" if info.get('valid') else "✗ (outdated)"
            size = f"{info.get('size_mb', 0):.1f}MB"
            lines.append(f"  {stub_key}: {valid} {size}")
        else:
            lines.append(f"  {stub_key}: not cached")
    
    return "\n".join(lines)
