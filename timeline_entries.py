#!/usr/bin/env python3
"""
Timeline Entries API - Chat timeline for developer workflow UI.
Supports user messages, AI messages, FDE messages, task cards, approvals, and status updates.
"""

import asyncio
import uuid
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Ensure indexes
db.timeline_entries.create_index("workflow_id")
db.timeline_entries.create_index([("workflow_id", ASCENDING), ("created_at", ASCENDING)])

router = APIRouter(
    prefix="/api/development/workflows/{workflow_id}/timeline",
    tags=["timeline"],
)


# --- Models ---

class TimelineEntryCreate(BaseModel):
    card_type: str = "user_message"
    content: str
    visibility: str = "public"
    todos: Optional[List[Dict]] = None
    parent_entry_id: Optional[str] = None

class TimelineEntryUpdate(BaseModel):
    content: Optional[str] = None
    visibility: Optional[str] = None
    todos: Optional[List[Dict]] = None

class TodoToggle(BaseModel):
    completed: bool


# --- Helpers ---

def entry_helper(doc, name_map=None) -> dict:
    """Convert MongoDB doc to response dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    # Resolve display name from user record if available
    if name_map and doc.get("author_id") in name_map:
        doc["author_name"] = name_map[doc["author_id"]]
    return doc


def _build_name_map(author_ids) -> dict:
    """Batch-resolve author_ids to display names from users collection."""
    if not author_ids:
        return {}
    users = db.users.find({"workos_user_id": {"$in": list(author_ids)}})
    name_map = {}
    for u in users:
        first = u.get("first_name", "")
        last = u.get("last_name", "")
        full = f"{first} {last}".strip()
        name = full or u.get("name") or u.get("email", "Unknown")
        name_map[u["workos_user_id"]] = name
    return name_map


def _get_user_info(request: Request, db_ref) -> dict:
    """Get user info from request for timeline entry authoring."""
    from auth import require_user_id
    workos_user_id = require_user_id(request)
    user = db_ref.users.find_one({"workos_user_id": workos_user_id})
    name = "Unknown"
    role = "fde"
    if user:
        first = user.get("first_name", "")
        last = user.get("last_name", "")
        full = f"{first} {last}".strip()
        name = full or user.get("name") or user.get("email", "Unknown")
        roles = user.get("roles", [])
        if "admin" in roles:
            role = "admin"
        elif "fde" in roles or "fdm" in roles:
            role = "fde"
        elif "client" in roles:
            role = "client"
        elif "qa" in roles:
            role = "qa"
    return {"user_id": workos_user_id, "name": name, "role": role}


def _verify_workflow_access(workflow_id: str, workos_user_id: str):
    """Verify user has access to this workflow's org."""
    from role_helpers import verify_org_membership
    workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow


# --- Endpoints ---

@router.get("")
async def list_timeline_entries(
    workflow_id: str,
    request: Request,
    visibility: Optional[str] = None,
):
    """List timeline entries for a workflow. Clients see public only."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        user_info = _get_user_info(request, db)
        query: Dict = {"workflow_id": workflow_id, "is_deleted": {"$ne": True}}

        # Clients can only see public entries
        if user_info["role"] == "client" or visibility == "public":
            query["visibility"] = "public"

        entries = list(
            db.timeline_entries.find(query).sort("created_at", ASCENDING)
        )
        # Resolve display names from user records
        author_ids = {e.get("author_id") for e in entries if e.get("author_id")}
        name_map = _build_name_map(author_ids)
        return [entry_helper(e, name_map) for e in entries]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=201)
async def create_timeline_entry(
    workflow_id: str,
    body: TimelineEntryCreate,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Create a timeline entry. Client user_messages trigger AI intake."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)
        user_info = _get_user_info(request, db)

        now = datetime.utcnow()
        entry_doc = {
            "workflow_id": workflow_id,
            "card_type": body.card_type,
            "content": body.content,
            "author_id": user_info["user_id"],
            "author_name": user_info["name"],
            "author_role": user_info["role"],
            "visibility": body.visibility,
            "todos": body.todos or [],
            "approval_data": None,
            "status_data": None,
            "parent_entry_id": body.parent_entry_id,
            "ai_model": None,
            "edited_by": None,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
        }
        result = db.timeline_entries.insert_one(entry_doc)
        entry_doc["_id"] = str(result.inserted_id)

        # If a client posts a user_message, trigger AI intake evaluation
        if user_info["role"] == "client" and body.card_type == "user_message":
            # Create a placeholder AI entry
            ai_placeholder = {
                "workflow_id": workflow_id,
                "card_type": "ai_message",
                "content": "",
                "author_id": "ai",
                "author_name": "AI Assistant",
                "author_role": "ai",
                "visibility": "internal",  # FDE reviews before publishing
                "todos": [],
                "approval_data": None,
                "status_data": None,
                "parent_entry_id": str(result.inserted_id),
                "ai_model": "claude-sonnet-4-5-20250929",
                "edited_by": None,
                "is_deleted": False,
                "processing": True,
                "created_at": now,
                "updated_at": now,
            }
            ai_result = db.timeline_entries.insert_one(ai_placeholder)
            ai_entry_id = str(ai_result.inserted_id)

            # Run AI intake in background
            background_tasks.add_task(
                _run_ai_intake, workflow_id, body.content, ai_entry_id, workflow
            )

        return entry_helper(entry_doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{entry_id}")
async def update_timeline_entry(
    workflow_id: str,
    entry_id: str,
    body: TimelineEntryUpdate,
    request: Request,
):
    """Update a timeline entry (FDE edits AI messages, toggles visibility)."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)
        user_info = _get_user_info(request, db)

        entry = db.timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        update_dict = {}
        if body.content is not None:
            update_dict["content"] = body.content
            update_dict["edited_by"] = user_info["user_id"]
        if body.visibility is not None:
            update_dict["visibility"] = body.visibility
        if body.todos is not None:
            update_dict["todos"] = body.todos

        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.timeline_entries.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": update_dict}
        )

        updated = db.timeline_entries.find_one({"_id": ObjectId(entry_id)})
        return entry_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{entry_id}")
async def delete_timeline_entry(
    workflow_id: str,
    entry_id: str,
    request: Request,
):
    """Soft delete a timeline entry."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        entry = db.timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        db.timeline_entries.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{entry_id}/publish")
async def publish_timeline_entry(
    workflow_id: str,
    entry_id: str,
    request: Request,
):
    """FDE publishes an internal AI message to the client (sets visibility to public)."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)
        require_role(request, db, ["admin", "fde"])

        entry = db.timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        db.timeline_entries.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": {"visibility": "public", "updated_at": datetime.utcnow()}}
        )

        updated = db.timeline_entries.find_one({"_id": ObjectId(entry_id)})
        return entry_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{entry_id}/todos/{todo_id}")
async def toggle_todo(
    workflow_id: str,
    entry_id: str,
    todo_id: str,
    body: TodoToggle,
    request: Request,
):
    """Toggle a todo item's completion status."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        entry = db.timeline_entries.find_one({"_id": ObjectId(entry_id), "workflow_id": workflow_id})
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")

        todos = entry.get("todos", [])
        found = False
        now = datetime.utcnow()
        for todo in todos:
            if todo.get("id") == todo_id:
                todo["completed"] = body.completed
                todo["completed_at"] = now if body.completed else None
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail="Todo not found")

        db.timeline_entries.update_one(
            {"_id": ObjectId(entry_id)},
            {"$set": {"todos": todos, "updated_at": now}}
        )

        updated = db.timeline_entries.find_one({"_id": ObjectId(entry_id)})
        return entry_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- AI Intake Background Task ---

async def _run_ai_intake(workflow_id: str, description: str, ai_entry_id: str, workflow: dict):
    """Background task: run AI intake agent and update the placeholder entry."""
    try:
        from agents.intake_agent import evaluate_request

        project = None
        if workflow.get("project_id"):
            project = db.projects.find_one({"_id": ObjectId(workflow["project_id"])})

        project_context = {}
        if project:
            project_context = {
                "name": project.get("name", ""),
                "description": project.get("description", ""),
            }

        result = await evaluate_request(description, project_context)

        # Update the AI placeholder entry with the result
        content = result.get("message", "")
        if result.get("questions"):
            content += "\n\n" + "\n".join(f"- {q}" for q in result["questions"])

        db.timeline_entries.update_one(
            {"_id": ObjectId(ai_entry_id)},
            {"$set": {
                "content": content,
                "processing": False,
                "updated_at": datetime.utcnow(),
            }, "$unset": {"processing": ""}}
        )
    except Exception as e:
        # Update with error message
        db.timeline_entries.update_one(
            {"_id": ObjectId(ai_entry_id)},
            {"$set": {
                "content": f"I encountered an issue processing your request. An FDE will follow up shortly.",
                "processing": False,
                "updated_at": datetime.utcnow(),
            }, "$unset": {"processing": ""}}
        )
        print(f"[INTAKE_AGENT] Error: {e}")
