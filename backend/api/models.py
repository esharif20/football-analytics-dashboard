import enum
from datetime import datetime
from sqlalchemy import (
    Integer, String, Text, Float, Boolean, Enum, JSON,
    TIMESTAMP, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class UserRole(str, enum.Enum):
    user = "user"
    admin = "admin"


class PipelineMode(str, enum.Enum):
    all = "all"
    radar = "radar"
    team = "team"
    track = "track"
    players = "players"
    ball = "ball"
    pitch = "pitch"


class ProcessingStatus(str, enum.Enum):
    pending = "pending"
    uploading = "uploading"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    openId: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    loginMethod: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(Enum(UserRole), nullable=False, server_default="user")
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())
    lastSignedIn: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    userId: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    originalUrl: Mapped[str] = mapped_column(Text, nullable=False)
    fileKey: Mapped[str] = mapped_column(String(512), nullable=False)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frameCount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fileSize: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mimeType: Mapped[str | None] = mapped_column(String(64), nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    videoId: Mapped[int] = mapped_column(Integer, nullable=False)
    userId: Mapped[int] = mapped_column(Integer, nullable=False)
    mode: Mapped[str] = mapped_column(Enum(PipelineMode), nullable=False)
    status: Mapped[str] = mapped_column(Enum(ProcessingStatus), nullable=False, server_default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    currentStage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    errorMessage: Mapped[str | None] = mapped_column(Text, nullable=True)
    skipCache: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    annotatedVideoUrl: Mapped[str | None] = mapped_column(Text, nullable=True)
    radarVideoUrl: Mapped[str | None] = mapped_column(Text, nullable=True)
    trackingDataUrl: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyticsDataUrl: Mapped[str | None] = mapped_column(Text, nullable=True)
    startedAt: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    completedAt: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    processingTimeMs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysisId: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    frameNumber: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    playerId: Mapped[int | None] = mapped_column(Integer, nullable=True)
    teamId: Mapped[int | None] = mapped_column(Integer, nullable=True)
    targetPlayerId: Mapped[int | None] = mapped_column(Integer, nullable=True)
    startX: Mapped[float | None] = mapped_column(Float, nullable=True)
    startY: Mapped[float | None] = mapped_column(Float, nullable=True)
    endX: Mapped[float | None] = mapped_column(Float, nullable=True)
    endY: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysisId: Mapped[int] = mapped_column(Integer, nullable=False)
    frameNumber: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    playerPositions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ballPosition: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    teamFormations: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    voronoiData: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())


class Statistic(Base):
    __tablename__ = "statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysisId: Mapped[int] = mapped_column(Integer, nullable=False)
    possessionTeam1: Mapped[float | None] = mapped_column(Float, nullable=True)
    possessionTeam2: Mapped[float | None] = mapped_column(Float, nullable=True)
    passesTeam1: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passesTeam2: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passAccuracyTeam1: Mapped[float | None] = mapped_column(Float, nullable=True)
    passAccuracyTeam2: Mapped[float | None] = mapped_column(Float, nullable=True)
    shotsTeam1: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shotsTeam2: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distanceCoveredTeam1: Mapped[float | None] = mapped_column(Float, nullable=True)
    distanceCoveredTeam2: Mapped[float | None] = mapped_column(Float, nullable=True)
    avgSpeedTeam1: Mapped[float | None] = mapped_column(Float, nullable=True)
    avgSpeedTeam2: Mapped[float | None] = mapped_column(Float, nullable=True)
    maxSpeedTeam1: Mapped[float | None] = mapped_column(Float, nullable=True)
    maxSpeedTeam2: Mapped[float | None] = mapped_column(Float, nullable=True)
    possessionChanges: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ballDistance: Mapped[float | None] = mapped_column(Float, nullable=True)
    ballAvgSpeed: Mapped[float | None] = mapped_column(Float, nullable=True)
    ballMaxSpeed: Mapped[float | None] = mapped_column(Float, nullable=True)
    directionChanges: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heatmapDataTeam1: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    heatmapDataTeam2: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    passNetworkTeam1: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    passNetworkTeam2: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    teamColorTeam1: Mapped[str | None] = mapped_column(String(7), nullable=True)
    teamColorTeam2: Mapped[str | None] = mapped_column(String(7), nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class Commentary(Base):
    __tablename__ = "commentary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysisId: Mapped[int] = mapped_column(Integer, nullable=False)
    eventId: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frameStart: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frameEnd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundingData: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
