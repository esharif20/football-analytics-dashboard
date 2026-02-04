"""
Football Analytics - FastAPI Backend
Pure Python backend for local development without Node.js dependencies
"""
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import sys

# Add parent directory to path for pipeline imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import videos, analysis, events, tracks, statistics, auth
from api.services.database import init_db, get_db
from api.services.websocket_manager import manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    init_db()
    yield

app = FastAPI(
    title="Football Analytics API",
    description="Computer Vision Pipeline for Football Match Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(videos.router, prefix="/api/videos", tags=["Videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(tracks.router, prefix="/api/tracks", tags=["Tracks"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["Statistics"])

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle subscription messages
            if data.get("type") == "subscribe":
                analysis_id = data.get("analysisId")
                if analysis_id:
                    manager.subscribe(client_id, f"analysis:{analysis_id}")
            elif data.get("type") == "unsubscribe":
                analysis_id = data.get("analysisId")
                if analysis_id:
                    manager.unsubscribe(client_id, f"analysis:{analysis_id}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "football-analytics-api"}

# Serve static files (for production build of React frontend)
frontend_build_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist")
if os.path.exists(frontend_build_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_build_path, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React frontend for all non-API routes"""
        if full_path.startswith("api/") or full_path.startswith("ws"):
            raise HTTPException(status_code=404)
        
        file_path = os.path.join(frontend_build_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_build_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
