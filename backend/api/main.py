import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .auth import AutoLoginMiddleware
from .ws import websocket_endpoint

from .routers import system, videos, analyses, events, tracks, stats, commentary, worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure uploads directory exists
    uploads_dir = Path(settings.LOCAL_STORAGE_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Football Analytics Dashboard API", lifespan=lifespan)

# CORS — allow the Vite dev server and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-login middleware (dev mode: every request gets a user)
app.add_middleware(AutoLoginMiddleware)

# Static file serving for /uploads
uploads_path = Path(settings.LOCAL_STORAGE_DIR)
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

# WebSocket endpoint for real-time progress
app.websocket("/ws/{analysis_id}")(websocket_endpoint)

# Auth endpoints (outside /api prefix for simplicity)
from fastapi import APIRouter, Depends, Request
from .deps import get_current_user, get_db
from .models import User
from .schemas import _row_to_dict

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


@auth_router.get("/me")
async def auth_me(request: Request):
    user = getattr(request.state, "user", None)
    if user is None:
        return None
    return _row_to_dict(user)


@auth_router.post("/auto-login")
async def auto_login(request: Request):
    user = getattr(request.state, "user", None)
    if user is None:
        return None
    return _row_to_dict(user)


@auth_router.post("/logout")
async def logout():
    return {"success": True}


app.include_router(auth_router)

# Multipart video upload endpoint (matches /api/upload/video used by Upload.tsx)
from fastapi import UploadFile, File, Form
import time
from .storage import storage_put
from .models import Video as VideoModel
from sqlalchemy.ext.asyncio import AsyncSession


@app.post("/api/upload/video")
async def upload_video_multipart(
    request: Request,
    video: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    user = getattr(request.state, "user", None)
    if user is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Authentication required")

    file_bytes = await video.read()
    file_name = video.filename or "video.mp4"
    mime_type = video.content_type or "video/mp4"

    if not title:
        title = file_name

    file_key = f"videos/{user.id}/{int(time.time() * 1000)}-{file_name}"
    result = await storage_put(file_key, file_bytes, mime_type)
    url = result["url"]

    video_record = VideoModel(
        userId=user.id,
        title=title,
        description=description or None,
        originalUrl=url,
        fileKey=file_key,
        fileSize=len(file_bytes),
        mimeType=mime_type,
    )
    db.add(video_record)
    await db.commit()
    await db.refresh(video_record)

    return {"id": video_record.id, "url": url}


# Mount all API routers under /api prefix
app.include_router(system.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(analyses.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(tracks.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(commentary.router, prefix="/api")
app.include_router(worker.router, prefix="/api")
