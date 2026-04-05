"""Test-support routes for seeding deterministic data.

Enabled only when ENABLE_TEST_SUPPORT=true.
"""

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..deps import get_db
from ..models import Analysis, PipelineMode, ProcessingStatus, User, UserRole, Video

router = APIRouter(prefix="/test-support", tags=["test-support"])


_ANALYTICS_FIXTURE = {
    "fps": 25,
    "homography_available": True,
    "possession": {
        "team_1_percentage": 55.2,
        "team_1_frames": 552,
        "team_2_percentage": 44.8,
        "team_2_frames": 448,
        "contested_frames": 12,
        "possession_changes": 7,
        "longest_team_1_spell": 120,
        "longest_team_2_spell": 95,
    },
    "player_kinematics": {
        "4": {
            "team_id": 1,
            "entity_type": "player",
            "total_distance_m": 1023.4,
            "avg_speed_m_per_sec": 1.9,
            "max_speed_m_per_sec": 7.1,
        },
        "8": {
            "team_id": 1,
            "entity_type": "player",
            "total_distance_m": 980.0,
            "avg_speed_m_per_sec": 2.1,
            "max_speed_m_per_sec": 7.5,
        },
        "10": {
            "team_id": 2,
            "entity_type": "player",
            "total_distance_m": 1100.2,
            "avg_speed_m_per_sec": 2.0,
            "max_speed_m_per_sec": 7.0,
        },
    },
    "ball_kinematics": {
        "total_distance_m": 345.6,
        "avg_speed_m_per_sec": 3.4,
        "max_speed_m_per_sec": 9.2,
    },
    "ball_path": {"direction_changes": 12},
    "events": [
        {
            "timestamp_sec": 90,
            "event_type": "pass",
            "team_id": 1,
            "player_track_id": 4,
            "target_player_track_id": 8,
            "success": True,
            "confidence": 0.92,
        },
        {
            "timestamp_sec": 150,
            "event_type": "shot",
            "team_id": 2,
            "player_track_id": 10,
            "confidence": 0.74,
        },
    ],
    "interaction_graph_team1": {"edges": [{"source": 4, "target": 8, "weight": 13}]},
    "interaction_graph_team2": {"edges": [{"source": 10, "target": 7, "weight": 9}]},
}


async def _get_or_create_user(db: AsyncSession) -> User:
    target_open_id = settings.OWNER_OPEN_ID or "test-user"

    result = await db.execute(select(User).where(User.openId == target_open_id))
    user = result.scalar_one_or_none()
    if user:
        return user

    user = User(openId=target_open_id, name="Test User", role=UserRole.user)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _create_video(db: AsyncSession, user_id: int) -> Video:
    video = Video(
        userId=user_id,
        title="Test Video",
        description="Test fixture video",
        originalUrl="/uploads/test-video.mp4",
        fileKey="videos/test/test-video.mp4",
        fileSize=1024,
        mimeType="video/mp4",
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)
    return video


@router.post("/seed-analysis")
async def seed_analysis(db: AsyncSession = Depends(get_db)):
    """Create a completed analysis with baked analytics data for tests."""

    user = await _get_or_create_user(db)

    video = await _create_video(db, user.id)

    analysis = Analysis(
        videoId=video.id,
        userId=user.id,
        mode=PipelineMode.all,
        status=ProcessingStatus.completed,
        progress=100,
        currentStage="done",
        analyticsDataUrl=json.dumps(_ANALYTICS_FIXTURE),
        startedAt=datetime.now(UTC).replace(tzinfo=None),
        completedAt=datetime.now(UTC).replace(tzinfo=None),
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    return {
        "analysisId": analysis.id,
        "videoId": video.id,
        "userId": user.id,
    }
