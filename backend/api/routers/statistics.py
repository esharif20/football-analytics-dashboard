"""
Statistics Router - Match statistics
Simplified - no authentication required
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import json

from api.services.database import get_analysis_by_id

router = APIRouter()

@router.get("/{analysis_id}")
async def get_statistics(analysis_id: int):
    """Get statistics for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Load from analytics data file if exists
    analytics_path = analysis.get("analytics_data_path")
    if analytics_path and os.path.exists(analytics_path):
        with open(analytics_path, 'r') as f:
            data = json.load(f)
            return data.get("statistics", {})
    
    return {"message": "No statistics available"}

@router.get("/{analysis_id}/heatmap")
async def get_heatmap(analysis_id: int, team: int = 1):
    """Get heatmap data for a team"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analytics_path = analysis.get("analytics_data_path")
    if analytics_path and os.path.exists(analytics_path):
        with open(analytics_path, 'r') as f:
            data = json.load(f)
            heatmaps = data.get("heatmaps", {})
            return heatmaps.get(f"team{team}", {})
    
    return {"message": "No heatmap data available"}

@router.get("/{analysis_id}/pass-network")
async def get_pass_network(analysis_id: int, team: int = 1):
    """Get pass network data for a team"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analytics_path = analysis.get("analytics_data_path")
    if analytics_path and os.path.exists(analytics_path):
        with open(analytics_path, 'r') as f:
            data = json.load(f)
            networks = data.get("pass_networks", {})
            return networks.get(f"team{team}", {})
    
    return {"message": "No pass network data available"}

@router.get("/{analysis_id}/download")
async def download_analytics(analysis_id: int):
    """Download full analytics as JSON file"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analytics_path = analysis.get("analytics_data_path")
    if analytics_path and os.path.exists(analytics_path):
        return FileResponse(
            analytics_path,
            media_type="application/json",
            filename=f"analytics_{analysis_id}.json"
        )
    
    raise HTTPException(status_code=404, detail="No analytics data available")
