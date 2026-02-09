#!/usr/bin/env python3
"""
Users API - Handle user authentication and management with WorkOS
"""

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# WorkOS configuration
WORKOS_API_KEY = os.getenv('WORKOS_API_KEY')
WORKOS_CLIENT_ID = os.getenv('WORKOS_CLIENT_ID')

# Create router
router = APIRouter(prefix="/api/users", tags=["users"])


# Pydantic Models
VALID_SUBSCRIPTION_TIERS = ["scale", "platform", "custom", "dble_team"]


class User(BaseModel):
    """User model"""
    id: str = Field(alias="_id")
    workos_user_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    platform_access: bool = False
    active_subscription: bool = False
    subscription_tier: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        populate_by_name = True


class UserCreate(BaseModel):
    """User creation/update model"""
    workos_user_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class AuthCallbackRequest(BaseModel):
    """Request body for authentication callback"""
    code: str
    state: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """Request body for token refresh"""
    refresh_token: str


# Helper functions
def user_helper(user) -> dict:
    """Convert MongoDB user to dict"""
    return {
        "_id": str(user["_id"]),
        "workos_user_id": user["workos_user_id"],
        "email": user["email"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "roles": user.get("roles", []),
        "organizations": user.get("organizations", []),
        "platform_access": user.get("platform_access", False),
        "active_subscription": user.get("active_subscription", False),
        "subscription_tier": user.get("subscription_tier"),
        "active_workflow_limit": user.get("active_workflow_limit"),
        "created_at": user.get("created_at", datetime.utcnow()),
        "updated_at": user.get("updated_at", datetime.utcnow()),
        "last_login": user.get("last_login")
    }


def get_or_create_user(workos_user_id: str, email: str, first_name: Optional[str] = None, last_name: Optional[str] = None) -> dict:
    """
    Get existing user or create new one
    Returns user dict with MongoDB _id
    """
    # Check if user exists
    existing_user = db.users.find_one({"workos_user_id": workos_user_id})

    if existing_user:
        # Update last login and user info
        db.users.update_one(
            {"_id": existing_user["_id"]},
            {
                "$set": {
                    "last_login": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "email": email,  # Update email in case it changed
                    "first_name": first_name or existing_user.get("first_name"),
                    "last_name": last_name or existing_user.get("last_name")
                }
            }
        )
        return user_helper(db.users.find_one({"_id": existing_user["_id"]}))

    # Create new user
    new_user = {
        "workos_user_id": workos_user_id,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "roles": [],
        "active_workflow_limit": None,
        "platform_access": False,
        "active_subscription": False,
        "subscription_tier": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "last_login": datetime.utcnow()
    }

    result = db.users.insert_one(new_user)
    created_user = db.users.find_one({"_id": result.inserted_id})
    return user_helper(created_user)


# Routes

@router.post("/auth/callback")
async def auth_callback(request: AuthCallbackRequest):
    """
    Handle authentication callback from WorkOS
    Exchange code for user info and create/update user
    """
    try:
        # In a real implementation, you would:
        # 1. Exchange the code for access token using WorkOS SDK
        # 2. Get user profile from WorkOS
        # 3. Create/update user in database

        # For now, this is a placeholder that expects the frontend to send user info
        # You'll need to install workos package: pip install workos

        from workos import WorkOSClient

        workos = WorkOSClient(api_key=WORKOS_API_KEY, client_id=WORKOS_CLIENT_ID)

        # Exchange authorization code for profile
        profile = workos.user_management.authenticate_with_code(
            code=request.code
        )

        # Extract user info
        workos_user_id = profile.user.id
        email = profile.user.email
        first_name = profile.user.first_name
        last_name = profile.user.last_name

        # Get or create user
        user = get_or_create_user(
            workos_user_id=workos_user_id,
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        return {
            "success": True,
            "user": user,
            "access_token": profile.access_token,
            "refresh_token": profile.refresh_token
        }

    except Exception as e:
        print(f"Authentication error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


@router.post("/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh an expired access token using a refresh token
    """
    try:
        from workos import WorkOSClient

        workos = WorkOSClient(api_key=WORKOS_API_KEY, client_id=WORKOS_CLIENT_ID)

        # Use WorkOS to refresh the token
        auth_response = workos.user_management.authenticate_with_refresh_token(
            refresh_token=request.refresh_token
        )

        return {
            "success": True,
            "access_token": auth_response.access_token,
            "refresh_token": auth_response.refresh_token
        }

    except Exception as e:
        print(f"Token refresh error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")


@router.get("/me")
async def get_current_user(request: Request, workos_user_id: Optional[str] = None):
    """Get current user. Uses JWT identity if no query param provided."""
    try:
        if not workos_user_id:
            from auth import require_user_id
            workos_user_id = require_user_id(request)
        user = db.users.find_one({"workos_user_id": workos_user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user_helper(user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Get user by MongoDB ID"""
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user_helper(user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_all_users():
    """Get all users (admin only - add auth later)"""
    try:
        users = list(db.users.find())
        return [user_helper(user) for user in users]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RoleUpdate(BaseModel):
    """Request body for updating user roles"""
    roles: List[str]


@router.patch("/{user_id}/roles")
async def update_user_roles(user_id: str, body: RoleUpdate, request: Request):
    """Update a user's roles (admin only)"""
    try:
        from role_helpers import require_role
        require_role(request, db, ["admin"])

        valid_roles = ["admin", "fde", "fdm", "qa", "client"]
        for role in body.roles:
            if role not in valid_roles:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role: {role}. Valid roles: {', '.join(valid_roles)}"
                )

        result = db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"roles": body.roles, "updated_at": datetime.utcnow()}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        user = db.users.find_one({"_id": ObjectId(user_id)})
        return user_helper(user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
