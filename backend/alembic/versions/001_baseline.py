"""baseline — all 7 tables

Revision ID: 001
Revises:
Create Date: 2026-03-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ---------- Native PostgreSQL enum types ----------

userrole_enum = sa.Enum("user", "admin", name="userrole")
pipelinemode_enum = sa.Enum("all", "radar", "team", "track", "players", "ball", "pitch", name="pipelinemode")
processingstatus_enum = sa.Enum("pending", "uploading", "processing", "completed", "failed", name="processingstatus")


def upgrade() -> None:
    # Create enum types first
    userrole_enum.create(op.get_bind(), checkfirst=True)
    pipelinemode_enum.create(op.get_bind(), checkfirst=True)
    processingstatus_enum.create(op.get_bind(), checkfirst=True)

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("openId", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("loginMethod", sa.String(64), nullable=True),
        sa.Column("role", userrole_enum, nullable=False, server_default=sa.text("'user'")),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("updatedAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("lastSignedIn", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- videos ---
    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("userId", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("originalUrl", sa.Text(), nullable=False),
        sa.Column("fileKey", sa.String(512), nullable=False),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("frameCount", sa.Integer(), nullable=True),
        sa.Column("fileSize", sa.Integer(), nullable=True),
        sa.Column("mimeType", sa.String(64), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("updatedAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- analyses ---
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("videoId", sa.Integer(), nullable=False),
        sa.Column("userId", sa.Integer(), nullable=False),
        sa.Column("mode", pipelinemode_enum, nullable=False),
        sa.Column("status", processingstatus_enum, nullable=False, server_default=sa.text("'pending'")),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("currentStage", sa.String(128), nullable=True),
        sa.Column("errorMessage", sa.Text(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("claimedBy", sa.String(128), nullable=True),
        sa.Column("skipCache", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("annotatedVideoUrl", sa.Text(), nullable=True),
        sa.Column("radarVideoUrl", sa.Text(), nullable=True),
        sa.Column("trackingDataUrl", sa.Text(), nullable=True),
        sa.Column("analyticsDataUrl", sa.Text(), nullable=True),
        sa.Column("startedAt", sa.TIMESTAMP(), nullable=True),
        sa.Column("completedAt", sa.TIMESTAMP(), nullable=True),
        sa.Column("processingTimeMs", sa.Integer(), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("updatedAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- events ---
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysisId", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("frameNumber", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("playerId", sa.Integer(), nullable=True),
        sa.Column("teamId", sa.Integer(), nullable=True),
        sa.Column("targetPlayerId", sa.Integer(), nullable=True),
        sa.Column("startX", sa.Float(), nullable=True),
        sa.Column("startY", sa.Float(), nullable=True),
        sa.Column("endX", sa.Float(), nullable=True),
        sa.Column("endY", sa.Float(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- tracks ---
    op.create_table(
        "tracks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysisId", sa.Integer(), nullable=False),
        sa.Column("frameNumber", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("playerPositions", sa.JSON(), nullable=True),
        sa.Column("ballPosition", sa.JSON(), nullable=True),
        sa.Column("teamFormations", sa.JSON(), nullable=True),
        sa.Column("voronoiData", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- statistics ---
    op.create_table(
        "statistics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysisId", sa.Integer(), nullable=False),
        sa.Column("possessionTeam1", sa.Float(), nullable=True),
        sa.Column("possessionTeam2", sa.Float(), nullable=True),
        sa.Column("passesTeam1", sa.Integer(), nullable=True),
        sa.Column("passesTeam2", sa.Integer(), nullable=True),
        sa.Column("passAccuracyTeam1", sa.Float(), nullable=True),
        sa.Column("passAccuracyTeam2", sa.Float(), nullable=True),
        sa.Column("shotsTeam1", sa.Integer(), nullable=True),
        sa.Column("shotsTeam2", sa.Integer(), nullable=True),
        sa.Column("distanceCoveredTeam1", sa.Float(), nullable=True),
        sa.Column("distanceCoveredTeam2", sa.Float(), nullable=True),
        sa.Column("avgSpeedTeam1", sa.Float(), nullable=True),
        sa.Column("avgSpeedTeam2", sa.Float(), nullable=True),
        sa.Column("maxSpeedTeam1", sa.Float(), nullable=True),
        sa.Column("maxSpeedTeam2", sa.Float(), nullable=True),
        sa.Column("possessionChanges", sa.Integer(), nullable=True),
        sa.Column("ballDistance", sa.Float(), nullable=True),
        sa.Column("ballAvgSpeed", sa.Float(), nullable=True),
        sa.Column("ballMaxSpeed", sa.Float(), nullable=True),
        sa.Column("directionChanges", sa.Integer(), nullable=True),
        sa.Column("heatmapDataTeam1", sa.JSON(), nullable=True),
        sa.Column("heatmapDataTeam2", sa.JSON(), nullable=True),
        sa.Column("passNetworkTeam1", sa.JSON(), nullable=True),
        sa.Column("passNetworkTeam2", sa.JSON(), nullable=True),
        sa.Column("teamColorTeam1", sa.String(7), nullable=True),
        sa.Column("teamColorTeam2", sa.String(7), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column("updatedAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )

    # --- commentary ---
    op.create_table(
        "commentary",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysisId", sa.Integer(), nullable=False),
        sa.Column("eventId", sa.Integer(), nullable=True),
        sa.Column("frameStart", sa.Integer(), nullable=True),
        sa.Column("frameEnd", sa.Integer(), nullable=True),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("groundingData", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    # Drop tables in reverse dependency order
    op.drop_table("commentary")
    op.drop_table("statistics")
    op.drop_table("tracks")
    op.drop_table("events")
    op.drop_table("analyses")
    op.drop_table("videos")
    op.drop_table("users")

    # Drop enum types
    processingstatus_enum.drop(op.get_bind(), checkfirst=True)
    pipelinemode_enum.drop(op.get_bind(), checkfirst=True)
    userrole_enum.drop(op.get_bind(), checkfirst=True)
