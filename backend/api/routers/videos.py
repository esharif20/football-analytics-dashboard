"""
Videos Router - Video upload and management
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from datetime import datetime

from api.services.database import (
    create_video, get_video_by_id, get_all_videos, delete_video as db_delete_video,
    compute_file_hash, get_video_by_hash
)

router = APIRouter()

# Local storage path
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class VideoCreate(BaseModel):
    title: str
    description: Optional[str] = None

class VideoResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: Optional[str]
    original_url: Optional[str]
    file_size: Optional[int]
    created_at: str

@router.get("")
async def list_videos():
    """List all videos"""
    videos = get_all_videos()
    return videos

@router.get("/{video_id}")
async def get_video(video_id: int):
    """Get video by ID"""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.post("")
async def upload_video(
    title: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...)
):
    """Upload a new video"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Check if video already exists (by hash) - for caching
    file_hash = compute_file_hash(file_path)
    existing = get_video_by_hash(file_hash)
    if existing:
        # Remove duplicate file, return existing video
        os.remove(file_path)
        return {
            "id": existing["id"],
            "url": f"/api/videos/file/{os.path.basename(existing['file_path'])}",
            "duplicate": True,
            "message": "Video already exists, using cached version"
        }
    
    # Create database record
    video_id = create_video(
        title=title,
        file_path=file_path,
        file_size=file_size
    )
    
    return {
        "id": video_id,
        "url": f"/api/videos/file/{safe_filename}"
    }

@router.post("/upload-base64")
async def upload_video_base64(request: Request):
    """Upload video via base64 (for compatibility with existing frontend)"""
    data = await request.json()
    
    title = data.get("title")
    file_name = data.get("fileName")
    file_base64 = data.get("fileBase64")
    file_size = data.get("fileSize")
    
    if not title or not file_base64:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    import base64
    
    # Decode base64
    file_data = base64.b64decode(file_base64)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file_name}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    # Check if video already exists (by hash)
    file_hash = compute_file_hash(file_path)
    existing = get_video_by_hash(file_hash)
    if existing:
        os.remove(file_path)
        return {
            "id": existing["id"],
            "url": f"/api/videos/file/{os.path.basename(existing['file_path'])}",
            "duplicate": True
        }
    
    # Create database record
    video_id = create_video(
        title=title,
        file_path=file_path,
        file_size=file_size or len(file_data)
    )
    
    return {
        "id": video_id,
        "url": f"/api/videos/file/{safe_filename}"
    }

@router.delete("/{video_id}")
async def delete_video(video_id: int):
    """Delete a video"""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file if exists
    if video.get("file_path") and os.path.exists(video["file_path"]):
        os.remove(video["file_path"])
    
    db_delete_video(video_id)
    return {"success": True}

@router.get("/file/{filename}")
async def serve_video_file(filename: str):
    """Serve video file"""
    from fastapi.responses import FileResponse
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="video/mp4")
