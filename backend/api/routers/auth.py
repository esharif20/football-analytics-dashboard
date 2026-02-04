"""
Authentication Router - Local mode (no OAuth required)
"""
from fastapi import APIRouter, HTTPException, Depends, Response, Request
from pydantic import BaseModel
from typing import Optional
import hashlib
import secrets

from api.services.database import get_user_by_id, get_user_by_email, create_user

router = APIRouter()

# Simple session storage (in production, use Redis or similar)
sessions: dict = {}

class LoginRequest(BaseModel):
    email: str
    password: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str]

def get_current_user(request: Request) -> Optional[dict]:
    """Get current user from session cookie"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    
    user_id = sessions[session_id]
    return get_user_by_id(user_id)

def require_user(request: Request) -> dict:
    """Require authenticated user"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@router.get("/me")
async def get_me(request: Request):
    """Get current user info"""
    user = get_current_user(request)
    if not user:
        return None
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"]
    }

@router.post("/login")
async def login(request: LoginRequest, response: Response):
    """Login or create user (local mode - simplified auth)"""
    user = get_user_by_email(request.email)
    
    if not user:
        # Auto-create user in local mode
        user_id = create_user(request.email, request.email.split("@")[0])
        user = get_user_by_id(user_id)
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = user["id"]
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7  # 7 days
    )
    
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"]
    }

@router.post("/logout")
async def logout(request: Request, response: Response):
    """Logout current user"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    
    response.delete_cookie("session_id")
    return {"success": True}

@router.post("/auto-login")
async def auto_login(response: Response):
    """Auto-login as local user for development"""
    # Get or create local user
    user = get_user_by_id(1)
    if not user:
        from api.services.database import init_db
        init_db()
        user = get_user_by_id(1)
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = user["id"]
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7
    )
    
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"]
    }
