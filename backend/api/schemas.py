from __future__ import annotations
import enum
import json
from datetime import datetime
from typing import Any
from pydantic import BaseModel


# ==================== Shared helpers ====================

def _dt(v: datetime | None) -> str | None:
    return v.isoformat() if v else None


def _json_safe(val: Any) -> Any:
    """Coerce a single value to a JSON-serializable type."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, enum.Enum):
        return val.value
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, (dict, list)):
        # Round-trip through json to strip any non-serializable nested types
        try:
            return json.loads(json.dumps(val))
        except (TypeError, ValueError, OverflowError):
            return str(val)
    # numpy scalars, bytes, or other unknown types
    try:
        return float(val) if isinstance(val, (int, float)) else str(val)
    except Exception:
        return str(val)


def _row_to_dict(row: Any) -> dict:
    """Convert a SQLAlchemy model instance to a JSON-safe dict."""
    d: dict[str, Any] = {}
    for c in row.__table__.columns:
        col_name = c.name
        val = getattr(row, c.key)
        d[col_name] = _json_safe(val)
    return d


# ==================== Auth ====================

class UserOut(BaseModel):
    id: int
    openId: str
    name: str | None = None
    email: str | None = None
    loginMethod: str | None = None
    role: str
    createdAt: str | None = None
    updatedAt: str | None = None
    lastSignedIn: str | None = None


# ==================== Videos ====================

class VideoUploadBase64(BaseModel):
    title: str
    description: str | None = None
    fileName: str
    fileBase64: str
    fileSize: int
    mimeType: str


# ==================== Analyses ====================

class AnalysisCreate(BaseModel):
    videoId: int
    mode: str
    fresh: bool = False


class AnalysisStatusUpdate(BaseModel):
    status: str
    progress: int
    currentStage: str | None = None
    errorMessage: str | None = None


class AnalysisResultsUpdate(BaseModel):
    annotatedVideoUrl: str | None = None
    radarVideoUrl: str | None = None
    trackingDataUrl: str | None = None
    analyticsDataUrl: str | None = None
    processingTimeMs: int | None = None


# ==================== Events ====================

class EventItem(BaseModel):
    type: str
    frameNumber: int
    timestamp: float
    playerId: int | None = None
    teamId: int | None = None
    targetPlayerId: int | None = None
    startX: float | None = None
    startY: float | None = None
    endX: float | None = None
    endY: float | None = None
    success: bool | None = None
    confidence: float | None = None
    metadata: Any | None = None


class EventsCreate(BaseModel):
    analysisId: int
    events: list[EventItem]


# ==================== Tracks ====================

class TrackItem(BaseModel):
    frameNumber: int
    timestamp: float
    playerPositions: Any | None = None
    ballPosition: Any | None = None
    teamFormations: Any | None = None
    voronoiData: Any | None = None


class TracksCreate(BaseModel):
    analysisId: int
    tracks: list[TrackItem]


# ==================== Statistics ====================

class StatisticsCreate(BaseModel):
    analysisId: int
    possessionTeam1: float | None = None
    possessionTeam2: float | None = None
    passesTeam1: int | None = None
    passesTeam2: int | None = None
    passAccuracyTeam1: float | None = None
    passAccuracyTeam2: float | None = None
    shotsTeam1: int | None = None
    shotsTeam2: int | None = None
    distanceCoveredTeam1: float | None = None
    distanceCoveredTeam2: float | None = None
    avgSpeedTeam1: float | None = None
    avgSpeedTeam2: float | None = None
    heatmapDataTeam1: Any | None = None
    heatmapDataTeam2: Any | None = None
    passNetworkTeam1: Any | None = None
    passNetworkTeam2: Any | None = None


# ==================== Commentary ====================

class CommentaryGenerate(BaseModel):
    type: str
    context: dict[str, Any] = {}


# ==================== Worker ====================

class WorkerStatusUpdate(BaseModel):
    status: str
    currentStage: str | None = None
    progress: int = 0
    error: str | None = None
    eta: int | None = None


class WorkerCompleteRequest(BaseModel):
    annotatedVideo: str | None = None
    radarVideo: str | None = None
    analytics: Any | None = None
    tracks: Any | None = None
    success: bool = True


class WorkerUploadVideo(BaseModel):
    videoData: str
    fileName: str
    contentType: str = "video/mp4"
