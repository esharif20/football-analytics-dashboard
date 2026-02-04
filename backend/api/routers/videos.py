"""
Videos Router - Video upload and management
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Request
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from datetime import datetime

from api.routers.auth import require_user
from api.services.database import (
    create_video, get_video_by_id, get_videos_by_user, delete_video as db_delete_video
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
async def list_videos(request: Request, user: dict = Depends(require_user)):
    """List all videos for current user"""
    videos = get_videos_by_user(user["id"])
    return videos

@router.get("/{video_id}")
async def get_video(video_id: int, request: Request, user: dict = Depends(require_user)):
    """Get video by ID"""
    video = get_video_by_id(video_id)
    if not video or video["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.post("")
async def upload_video(
    request: Request,
    title: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    user: dict = Depends(require_user)
):
    """Upload a new video"""
    # Validate file type
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{user['id']}_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Create database record
    video_id = create_video(
        user_id=user["id"],
        title=title,
        description=description,
        original_url=f"/api/videos/file/{safe_filename}",
        file_key=safe_filename,
        file_size=file_size,
        mime_type=file.content_type
    )
    
    return {
        "id": video_id,
        "url": f"/api/videos/file/{safe_filename}"
    }

@router.post("/upload-base64")
async def upload_video_base64(
    request: Request,
    user: dict = Depends(require_user)
):
    """Upload video via base64 (for compatibility with existing frontend)"""
    data = await request.json()
    
    title = data.get("title")
    description = data.get("description")
    file_name = data.get("fileName")
    file_base64 = data.get("fileBase64")
    file_size = data.get("fileSize")
    mime_type = data.get("mimeType")
    
    if not title or not file_base64:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    import base64
    
    # Decode base64
    file_data = base64.b64decode(file_base64)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{user['id']}_{timestamp}_{file_name}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    # Create database record
    video_id = create_video(
        user_id=user["id"],
        title=title,
        description=description,
        original_url=f"/api/videos/file/{safe_filename}",
        file_key=safe_filename,
        file_size=file_size or len(file_data),
        mime_type=mime_type
    )
    
    return {
        "id": video_id,
        "url": f"/api/videos/file/{safe_filename}"
    }

@router.delete("/{video_id}")
async def delete_video(video_id: int, request: Request, user: dict = Depends(require_user)):
    """Delete a video"""
    video = get_video_by_id(video_id)
    if not video or video["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file if exists
    if video.get("file_key"):
        file_path = os.path.join(UPLOAD_DIR, video["file_key"])
        if os.path.exists(file_path):
            os.remove(file_path)
    
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
