"""
Statistics Router - Match statistics management
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional, Any

from api.routers.auth import require_user
from api.services.database import (
    get_analysis_by_id, create_statistics, get_statistics_by_analysis
)

router = APIRouter()

class StatisticsCreate(BaseModel):
    analysisId: int
    possessionTeam1: Optional[float] = None
    possessionTeam2: Optional[float] = None
    passesTeam1: Optional[int] = None
    passesTeam2: Optional[int] = None
    passAccuracyTeam1: Optional[float] = None
    passAccuracyTeam2: Optional[float] = None
    shotsTeam1: Optional[int] = None
    shotsTeam2: Optional[int] = None
    distanceCoveredTeam1: Optional[float] = None
    distanceCoveredTeam2: Optional[float] = None
    avgSpeedTeam1: Optional[float] = None
    avgSpeedTeam2: Optional[float] = None
    heatmapDataTeam1: Optional[Any] = None
    heatmapDataTeam2: Optional[Any] = None
    passNetworkTeam1: Optional[Any] = None
    passNetworkTeam2: Optional[Any] = None

@router.get("/{analysis_id}")
async def get_statistics(analysis_id: int, request: Request, user: dict = Depends(require_user)):
    """Get statistics for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    stats = get_statistics_by_analysis(analysis_id)
    if not stats:
        return None
    return stats

@router.post("")
async def create_new_statistics(
    data: StatisticsCreate,
    request: Request,
    user: dict = Depends(require_user)
):
    """Create statistics for an analysis"""
    analysis = get_analysis_by_id(data.analysisId)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    stats_data = {
        "analysis_id": data.analysisId,
        "possession_team1": data.possessionTeam1,
        "possession_team2": data.possessionTeam2,
        "passes_team1": data.passesTeam1,
        "passes_team2": data.passesTeam2,
        "pass_accuracy_team1": data.passAccuracyTeam1,
        "pass_accuracy_team2": data.passAccuracyTeam2,
        "shots_team1": data.shotsTeam1,
        "shots_team2": data.shotsTeam2,
        "distance_covered_team1": data.distanceCoveredTeam1,
        "distance_covered_team2": data.distanceCoveredTeam2,
        "avg_speed_team1": data.avgSpeedTeam1,
        "avg_speed_team2": data.avgSpeedTeam2,
        "heatmap_data_team1": data.heatmapDataTeam1,
        "heatmap_data_team2": data.heatmapDataTeam2,
        "pass_network_team1": data.passNetworkTeam1,
        "pass_network_team2": data.passNetworkTeam2,
    }
    
    stats_id = create_statistics(stats_data)
    return {"id": stats_id}
