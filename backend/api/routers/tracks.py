"""
Tracks Router - Player and ball tracking data
Simplified - no authentication required
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Any
import os
import json

from api.services.database import get_analysis_by_id

router = APIRouter()

@router.get("/{analysis_id}")
async def get_tracks(analysis_id: int):
    """Get tracking data for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Load from tracking data file if exists
    tracking_path = analysis.get("tracking_data_path")
    if tracking_path and os.path.exists(tracking_path):
        with open(tracking_path, 'r') as f:
            return json.load(f)
    
    return {"frames": [], "message": "No tracking data available"}

@router.get("/{analysis_id}/frame/{frame_number}")
async def get_track_frame(analysis_id: int, frame_number: int):
    """Get track data at specific frame"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    tracking_path = analysis.get("tracking_data_path")
    if tracking_path and os.path.exists(tracking_path):
        with open(tracking_path, 'r') as f:
            data = json.load(f)
            frames = data.get("frames", [])
            for frame in frames:
                if frame.get("frame_number") == frame_number:
                    return frame
    
    raise HTTPException(status_code=404, detail="Track not found for this frame")

@router.get("/{analysis_id}/download")
async def download_tracks(analysis_id: int):
    """Download tracking data as JSON file"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    tracking_path = analysis.get("tracking_data_path")
    if tracking_path and os.path.exists(tracking_path):
        return FileResponse(
            tracking_path,
            media_type="application/json",
            filename=f"tracks_{analysis_id}.json"
        )
    
    raise HTTPException(status_code=404, detail="No tracking data available")
