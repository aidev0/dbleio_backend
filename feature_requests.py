"""
Feature Requests API
Handles feature/system request submissions from the landing page
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
import os
from pymongo import MongoClient

router = APIRouter(prefix="/api/feature-requests", tags=["feature-requests"])

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "dble_db")
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]


class FeatureRequest(BaseModel):
    name: str
    email: EmailStr
    company: Optional[str] = None
    feature_type: str
    description: Optional[str] = None
    plan: str


class FeatureRequestResponse(BaseModel):
    id: str
    message: str


@router.post("", response_model=FeatureRequestResponse)
async def create_feature_request(request: FeatureRequest):
    """
    Create a new feature/system request
    """
    try:
        # Create the document
        doc = {
            "name": request.name,
            "email": request.email,
            "company": request.company,
            "feature_type": request.feature_type,
            "description": request.description,
            "plan": request.plan,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Insert into database
        result = db.feature_requests.insert_one(doc)

        return FeatureRequestResponse(
            id=str(result.inserted_id),
            message="Request submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_feature_requests():
    """
    Get all feature requests (admin only - add auth later)
    """
    try:
        requests = list(db.feature_requests.find().sort("created_at", -1))
        for req in requests:
            req["_id"] = str(req["_id"])
        return requests
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{request_id}")
async def get_feature_request(request_id: str):
    """
    Get a specific feature request
    """
    from bson import ObjectId
    try:
        req = db.feature_requests.find_one({"_id": ObjectId(request_id)})
        if not req:
            raise HTTPException(status_code=404, detail="Request not found")
        req["_id"] = str(req["_id"])
        return req
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{request_id}/status")
async def update_request_status(request_id: str, status: str):
    """
    Update the status of a feature request
    """
    from bson import ObjectId
    valid_statuses = ["pending", "reviewing", "approved", "in_progress", "completed", "rejected"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    try:
        result = db.feature_requests.update_one(
            {"_id": ObjectId(request_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Request not found")
        return {"message": "Status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
