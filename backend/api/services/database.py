"""
SQLite Database Service - Simplified for Local Development
No user authentication, just videos and analyses
"""
import sqlite3
import os
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "football.db")

def init_db():
    """Initialize SQLite database with schema"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Videos table - simplified
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            duration_ms INTEGER,
            fps REAL,
            width INTEGER,
            height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Analyses table - simplified
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            current_stage TEXT,
            error_message TEXT,
            annotated_video_path TEXT,
            radar_video_path TEXT,
            tracking_data_path TEXT,
            analytics_data_path TEXT,
            processing_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)
    
    # Stubs table - for caching intermediate results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stubs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_hash TEXT NOT NULL,
            mode TEXT NOT NULL,
            stub_type TEXT NOT NULL,
            stub_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_hash, mode, stub_type)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

@contextmanager
def get_db():
    """Get database connection context manager"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def dict_from_row(row) -> Optional[Dict[str, Any]]:
    """Convert sqlite Row to dictionary"""
    if row is None:
        return None
    return dict(row)

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# Video operations
def create_video(title: str, file_path: str, file_size: int = None) -> int:
    """Create a new video record"""
    file_hash = compute_file_hash(file_path) if os.path.exists(file_path) else None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (title, file_path, file_hash, file_size)
            VALUES (?, ?, ?, ?)
        """, (title, file_path, file_hash, file_size))
        return cursor.lastrowid

def get_video_by_id(video_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        return dict_from_row(cursor.fetchone())

def get_video_by_hash(file_hash: str) -> Optional[Dict]:
    """Find video by file hash - for detecting duplicate uploads"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE file_hash = ?", (file_hash,))
        return dict_from_row(cursor.fetchone())

def get_all_videos() -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY created_at DESC")
        return [dict_from_row(row) for row in cursor.fetchall()]

def delete_video(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        # Delete associated analyses first
        cursor.execute("DELETE FROM analyses WHERE video_id = ?", (video_id,))
        cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))

def update_video_metadata(video_id: int, duration_ms: int = None, fps: float = None, 
                          width: int = None, height: int = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE videos 
            SET duration_ms = COALESCE(?, duration_ms),
                fps = COALESCE(?, fps),
                width = COALESCE(?, width),
                height = COALESCE(?, height)
            WHERE id = ?
        """, (duration_ms, fps, width, height, video_id))

# Analysis operations
def create_analysis(video_id: int, mode: str) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (video_id, mode, status, progress)
            VALUES (?, ?, 'pending', 0)
        """, (video_id, mode))
        return cursor.lastrowid

def get_analysis_by_id(analysis_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        return dict_from_row(cursor.fetchone())

def get_all_analyses() -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses ORDER BY created_at DESC")
        return [dict_from_row(row) for row in cursor.fetchall()]

def get_analyses_by_video(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE video_id = ? ORDER BY created_at DESC", (video_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

def get_analysis_by_video_and_mode(video_id: int, mode: str) -> Optional[Dict]:
    """Find existing analysis for a video+mode combination"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM analyses 
            WHERE video_id = ? AND mode = ? AND status = 'completed'
            ORDER BY created_at DESC LIMIT 1
        """, (video_id, mode))
        return dict_from_row(cursor.fetchone())

def update_analysis_status(analysis_id: int, status: str, progress: int, 
                           current_stage: str = None, error_message: str = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses 
            SET status = ?, progress = ?, current_stage = ?, error_message = ?,
                completed_at = CASE WHEN ? IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END
            WHERE id = ?
        """, (status, progress, current_stage, error_message, status, analysis_id))

def update_analysis_results(analysis_id: int, annotated_video_path: str = None,
                            radar_video_path: str = None, tracking_data_path: str = None,
                            analytics_data_path: str = None, processing_time_ms: int = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses 
            SET annotated_video_path = COALESCE(?, annotated_video_path),
                radar_video_path = COALESCE(?, radar_video_path),
                tracking_data_path = COALESCE(?, tracking_data_path),
                analytics_data_path = COALESCE(?, analytics_data_path),
                processing_time_ms = COALESCE(?, processing_time_ms)
            WHERE id = ?
        """, (annotated_video_path, radar_video_path, tracking_data_path, 
              analytics_data_path, processing_time_ms, analysis_id))

def delete_analysis(analysis_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))

# Stub operations (for caching)
def save_stub(video_hash: str, mode: str, stub_type: str, stub_path: str):
    """Save a stub file reference for caching"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO stubs (video_hash, mode, stub_type, stub_path)
            VALUES (?, ?, ?, ?)
        """, (video_hash, mode, stub_type, stub_path))

def get_stub(video_hash: str, mode: str, stub_type: str) -> Optional[str]:
    """Get cached stub path if exists"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stub_path FROM stubs 
            WHERE video_hash = ? AND mode = ? AND stub_type = ?
        """, (video_hash, mode, stub_type))
        row = cursor.fetchone()
        if row and os.path.exists(row[0]):
            return row[0]
        return None

def get_all_stubs_for_video(video_hash: str, mode: str) -> Dict[str, str]:
    """Get all cached stubs for a video+mode"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stub_type, stub_path FROM stubs 
            WHERE video_hash = ? AND mode = ?
        """, (video_hash, mode))
        stubs = {}
        for row in cursor.fetchall():
            if os.path.exists(row[1]):
                stubs[row[0]] = row[1]
        return stubs

def clear_stubs(video_hash: str = None, mode: str = None):
    """Clear stubs - optionally filtered by video_hash and/or mode"""
    with get_db() as conn:
        cursor = conn.cursor()
        if video_hash and mode:
            cursor.execute("DELETE FROM stubs WHERE video_hash = ? AND mode = ?", (video_hash, mode))
        elif video_hash:
            cursor.execute("DELETE FROM stubs WHERE video_hash = ?", (video_hash,))
        elif mode:
            cursor.execute("DELETE FROM stubs WHERE mode = ?", (mode,))
        else:
            cursor.execute("DELETE FROM stubs")
