import base64
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Video
from ..schemas import VideoUploadBase64, _row_to_dict
from ..storage import storage_put

router = APIRouter(prefix="/videos", tags=["videos"])


@router.get("")
async def list_videos(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Video).where(Video.userId == user.id).order_by(desc(Video.createdAt))
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.get("/{video_id}")
async def get_video(video_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Video).where(Video.id == video_id).limit(1))
    video = result.scalar_one_or_none()
    if not video or video.userId != user.id:
        raise HTTPException(status_code=404, detail="Video not found")
    return _row_to_dict(video)


@router.post("/upload-base64")
async def upload_base64(body: VideoUploadBase64, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    file_bytes = base64.b64decode(body.fileBase64)
    file_key = f"videos/{user.id}/{int(time.time() * 1000)}-{body.fileName}"
    result = await storage_put(file_key, file_bytes, body.mimeType)
    url = result["url"]

    video = Video(
        userId=user.id,
        title=body.title,
        description=body.description,
        originalUrl=url,
        fileKey=file_key,
        fileSize=body.fileSize,
        mimeType=body.mimeType,
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)
    return {"id": video.id, "url": url}


@router.delete("/{video_id}")
async def delete_video(video_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Video).where(Video.id == video_id).limit(1))
    video = result.scalar_one_or_none()
    if not video or video.userId != user.id:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.delete(video)
    await db.commit()
    return {"success": True}
