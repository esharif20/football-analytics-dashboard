"""
Authentication Router - Simplified (No login required)
Auto-creates a local user, no authentication needed
"""
from fastapi import APIRouter, Request

router = APIRouter()

# Default user - no login required
DEFAULT_USER = {
    "id": 1,
    "email": "local@localhost",
    "name": "Local User"
}

def get_current_user(request: Request = None) -> dict:
    """Always return the default local user - no auth required"""
    return DEFAULT_USER

def require_user(request: Request = None) -> dict:
    """Always return the default local user - no auth required"""
    return DEFAULT_USER

@router.get("/me")
async def get_me():
    """Get current user info - always returns local user"""
    return DEFAULT_USER

@router.post("/login")
async def login():
    """Login - always succeeds with local user"""
    return DEFAULT_USER

@router.post("/logout")
async def logout():
    """Logout - no-op since we don't track sessions"""
    return {"success": True}

@router.post("/auto-login")
async def auto_login():
    """Auto-login - returns local user"""
    return DEFAULT_USER
