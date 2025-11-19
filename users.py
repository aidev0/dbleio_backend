#!/usr/bin/env python3
"""
Users API - Handle user authentication and management with WorkOS
"""

from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'video_marketing_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# WorkOS configuration
WORKOS_API_KEY = os.getenv('WORKOS_API_KEY')
WORKOS_CLIENT_ID = os.getenv('WORKOS_CLIENT_ID')

# Create router
router = APIRouter(prefix="/api/users", tags=["users"])


# Pydantic Models
class User(BaseModel):
    """User model"""
    id: str = Field(alias="_id")
    workos_user_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
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


# Helper functions
def user_helper(user) -> dict:
    """Convert MongoDB user to dict"""
    return {
        "_id": str(user["_id"]),
        "workos_user_id": user["workos_user_id"],
        "email": user["email"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
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


@router.get("/me")
async def get_current_user(workos_user_id: str):
    """Get current user by WorkOS user ID"""
    try:
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
