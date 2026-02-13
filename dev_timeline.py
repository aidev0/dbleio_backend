#!/usr/bin/env python3
"""
Development Workflow Timeline Entries API.
Chat messages, status updates, and activity feed for dev workflows.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

router = APIRouter(prefix="/api/dev/workflows/{workflow_id}/timeline", tags=["dev-timeline"])


# --- Models ---

class TimelineEntryCreate(BaseModel):
    card_type: str = "user_message"
    content: str
    visibility: str = "public"
    parent_entry_id: Optional[str] = None


class TimelineEntryUpdate(BaseModel):
    content: Optional[str] = None
    visibility: Optional[str] = None


# --- Helpers ---

def _entry_helper(doc) -> dict:
    if not doc:
        return {}
    return {
        "_id": str(doc["_id"]),
        "workflow_id": doc.get("workflow_id"),
        "card_type": doc.get("card_type"),
        "content": doc.get("content"),
        "author_id": doc.get("author_id"),
        "author_role": doc.get("author_role"),
        "visibility": doc.get("visibility", "public"),
        "parent_entry_id": doc.get("parent_entry_id"),
        "is_deleted": doc.get("is_deleted", False),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _verify_access(workflow_id: str, workos_user_id: str):
    workflow = db.dev_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    from role_helpers import verify_org_membership
    verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow


# --- Endpoints ---

@router.get("")
async def list_timeline_entries(workflow_id: str, request: Request, visibility: Optional[str] = None):
    """List timeline entries for a dev workflow."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_access(workflow_id, workos_user_id)

        query = {"workflow_id": workflow_id, "is_deleted": {"$ne": True}}
        if visibility:
            query["visibility"] = visibility

        entries = list(db.dev_timeline_entries.find(query).sort("created_at", 1))
        return [_entry_helper(e) for e in entries]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=201)
async def create_timeline_entry(workflow_id: str, body: TimelineEntryCreate, request: Request):
    """Create a timeline entry."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_access(workflow_id, workos_user_id)

        user = db.users.find_one({"workos_user_id": workos_user_id})
        author_role = "user"
        if user:
            roles = user.get("roles", [])
            if "admin" in roles:
                author_role = "admin"
            elif "fdm" in roles:
                author_role = "fdm"

        now = datetime.utcnow()
        doc = {
            "workflow_id": workflow_id,
            "card_type": body.card_type,
            "content": body.content,
            "author_id": workos_user_id,
            "author_role": author_role,
            "visibility": body.visibility,
            "parent_entry_id": body.parent_entry_id,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
        }
        result = db.dev_timeline_entries.insert_one(doc)
        entry = db.dev_timeline_entries.find_one({"_id": result.inserted_id})
        return _entry_helper(entry)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{entry_id}")
async def update_timeline_entry(workflow_id: str, entry_id: str, body: TimelineEntryUpdate, request: Request):
    """Update a timeline entry."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_access(workflow_id, workos_user_id)

        entry = db.dev_timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        if entry.get("author_id") != workos_user_id:
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user or "admin" not in user.get("roles", []):
                raise HTTPException(status_code=403, detail="Can only edit your own entries")

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.dev_timeline_entries.update_one({"_id": ObjectId(entry_id)}, {"$set": update_dict})

        updated = db.dev_timeline_entries.find_one({"_id": ObjectId(entry_id)})
        return _entry_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{entry_id}", status_code=204)
async def delete_timeline_entry(workflow_id: str, entry_id: str, request: Request):
    """Soft-delete a timeline entry."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_access(workflow_id, workos_user_id)

        entry = db.dev_timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        db.dev_timeline_entries.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
