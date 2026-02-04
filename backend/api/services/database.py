"""
SQLite Database Service for Local Development
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "football.db")

def init_db():
    """Initialize SQLite database with schema"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            original_url TEXT,
            file_key TEXT,
            file_size INTEGER,
            mime_type TEXT,
            duration_ms INTEGER,
            fps REAL,
            width INTEGER,
            height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Analyses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            progress INTEGER DEFAULT 0,
            current_stage TEXT,
            error_message TEXT,
            annotated_video_url TEXT,
            radar_video_url TEXT,
            tracking_data_url TEXT,
            analytics_data_url TEXT,
            processing_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            frame_number INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            player_id INTEGER,
            team_id INTEGER,
            target_player_id INTEGER,
            start_x REAL,
            start_y REAL,
            end_x REAL,
            end_y REAL,
            success INTEGER,
            confidence REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)
    
    # Tracks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            frame_number INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            player_positions TEXT,
            ball_position TEXT,
            team_formations TEXT,
            voronoi_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)
    
    # Statistics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL UNIQUE,
            possession_team1 REAL,
            possession_team2 REAL,
            passes_team1 INTEGER,
            passes_team2 INTEGER,
            pass_accuracy_team1 REAL,
            pass_accuracy_team2 REAL,
            shots_team1 INTEGER,
            shots_team2 INTEGER,
            distance_covered_team1 REAL,
            distance_covered_team2 REAL,
            avg_speed_team1 REAL,
            avg_speed_team2 REAL,
            heatmap_data_team1 TEXT,
            heatmap_data_team2 TEXT,
            pass_network_team1 TEXT,
            pass_network_team2 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)
    
    # Commentary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commentary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            grounding_data TEXT,
            frame_start INTEGER,
            frame_end INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)
    
    # Create default user for local development
    cursor.execute("""
        INSERT OR IGNORE INTO users (id, email, name) VALUES (1, 'local@localhost', 'Local User')
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

# Video operations
def create_video(user_id: int, title: str, description: str = None, original_url: str = None,
                 file_key: str = None, file_size: int = None, mime_type: str = None) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (user_id, title, description, original_url, file_key, file_size, mime_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, title, description, original_url, file_key, file_size, mime_type))
        return cursor.lastrowid

def get_video_by_id(video_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        return dict_from_row(cursor.fetchone())

def get_videos_by_user(user_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

def delete_video(video_id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))

# Analysis operations
def create_analysis(video_id: int, user_id: int, mode: str) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analyses (video_id, user_id, mode, status, progress)
            VALUES (?, ?, ?, 'pending', 0)
        """, (video_id, user_id, mode))
        return cursor.lastrowid

def get_analysis_by_id(analysis_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        return dict_from_row(cursor.fetchone())

def get_analyses_by_user(user_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

def get_analyses_by_video(video_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE video_id = ? ORDER BY created_at DESC", (video_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

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

def update_analysis_results(analysis_id: int, annotated_video_url: str = None,
                            radar_video_url: str = None, tracking_data_url: str = None,
                            analytics_data_url: str = None, processing_time_ms: int = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses 
            SET annotated_video_url = COALESCE(?, annotated_video_url),
                radar_video_url = COALESCE(?, radar_video_url),
                tracking_data_url = COALESCE(?, tracking_data_url),
                analytics_data_url = COALESCE(?, analytics_data_url),
                processing_time_ms = COALESCE(?, processing_time_ms)
            WHERE id = ?
        """, (annotated_video_url, radar_video_url, tracking_data_url, 
              analytics_data_url, processing_time_ms, analysis_id))

# Events operations
def create_events(events: List[Dict]):
    with get_db() as conn:
        cursor = conn.cursor()
        for event in events:
            cursor.execute("""
                INSERT INTO events (analysis_id, type, frame_number, timestamp, player_id,
                    team_id, target_player_id, start_x, start_y, end_x, end_y, success, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (event['analysis_id'], event['type'], event['frame_number'], event['timestamp'],
                  event.get('player_id'), event.get('team_id'), event.get('target_player_id'),
                  event.get('start_x'), event.get('start_y'), event.get('end_x'), event.get('end_y'),
                  event.get('success'), event.get('confidence'), 
                  str(event.get('metadata')) if event.get('metadata') else None))

def get_events_by_analysis(analysis_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE analysis_id = ? ORDER BY frame_number", (analysis_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

def get_events_by_type(analysis_id: int, event_type: str) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE analysis_id = ? AND type = ? ORDER BY frame_number", 
                       (analysis_id, event_type))
        return [dict_from_row(row) for row in cursor.fetchall()]

# Tracks operations
def create_tracks(tracks: List[Dict]):
    with get_db() as conn:
        cursor = conn.cursor()
        for track in tracks:
            cursor.execute("""
                INSERT INTO tracks (analysis_id, frame_number, timestamp, player_positions,
                    ball_position, team_formations, voronoi_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (track['analysis_id'], track['frame_number'], track['timestamp'],
                  str(track.get('player_positions')), str(track.get('ball_position')),
                  str(track.get('team_formations')), str(track.get('voronoi_data'))))

def get_tracks_by_analysis(analysis_id: int) -> List[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE analysis_id = ? ORDER BY frame_number", (analysis_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]

def get_track_at_frame(analysis_id: int, frame_number: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks WHERE analysis_id = ? AND frame_number = ?", 
                       (analysis_id, frame_number))
        return dict_from_row(cursor.fetchone())

# Statistics operations
def create_statistics(stats: Dict) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO statistics (analysis_id, possession_team1, possession_team2,
                passes_team1, passes_team2, pass_accuracy_team1, pass_accuracy_team2,
                shots_team1, shots_team2, distance_covered_team1, distance_covered_team2,
                avg_speed_team1, avg_speed_team2, heatmap_data_team1, heatmap_data_team2,
                pass_network_team1, pass_network_team2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stats['analysis_id'], stats.get('possession_team1'), stats.get('possession_team2'),
              stats.get('passes_team1'), stats.get('passes_team2'),
              stats.get('pass_accuracy_team1'), stats.get('pass_accuracy_team2'),
              stats.get('shots_team1'), stats.get('shots_team2'),
              stats.get('distance_covered_team1'), stats.get('distance_covered_team2'),
              stats.get('avg_speed_team1'), stats.get('avg_speed_team2'),
              str(stats.get('heatmap_data_team1')), str(stats.get('heatmap_data_team2')),
              str(stats.get('pass_network_team1')), str(stats.get('pass_network_team2'))))
        return cursor.lastrowid

def get_statistics_by_analysis(analysis_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM statistics WHERE analysis_id = ?", (analysis_id,))
        return dict_from_row(cursor.fetchone())

# User operations
def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return dict_from_row(cursor.fetchone())

def get_user_by_email(email: str) -> Optional[Dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        return dict_from_row(cursor.fetchone())

def create_user(email: str, name: str = None, password_hash: str = None) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, name, password_hash)
            VALUES (?, ?, ?)
        """, (email, name, password_hash))
        return cursor.lastrowid
