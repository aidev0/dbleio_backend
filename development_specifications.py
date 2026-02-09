#!/usr/bin/env python3
"""
Development Specifications API - Manage feature specifications for workflows.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
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

router = APIRouter(prefix="/api/development/specifications", tags=["development-specifications"])


# --- Models ---

class SpecificationCreate(BaseModel):
    organization_id: str
    project_id: str
    title: str
    spec_text: str
    acceptance_criteria: Optional[str] = None
    target_repos: List[str] = []
    priority: str = "medium"  # low, medium, high, critical

class SpecificationUpdate(BaseModel):
    title: Optional[str] = None
    spec_text: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    target_repos: Optional[List[str]] = None
    priority: Optional[str] = None


# --- Helpers ---

def spec_helper(spec) -> dict:
    return {
        "_id": str(spec["_id"]),
        "organization_id": spec["organization_id"],
        "project_id": spec["project_id"],
        "title": spec["title"],
        "spec_text": spec["spec_text"],
        "acceptance_criteria": spec.get("acceptance_criteria"),
        "target_repos": spec.get("target_repos", []),
        "priority": spec.get("priority", "medium"),
        "created_by": spec.get("created_by"),
        "created_at": spec.get("created_at"),
        "updated_at": spec.get("updated_at"),
    }


# --- Endpoints ---

@router.post("", status_code=201)
async def create_specification(body: SpecificationCreate, request: Request):
    """Create a new specification (admin/fde only)."""
    try:
        from auth import require_user_id
        from role_helpers import require_role, verify_org_membership
        workos_user_id = require_user_id(request)

        user = require_role(request, db, ["admin", "fde"])
        verify_org_membership(db, body.organization_id, workos_user_id)

        now = datetime.utcnow()
        spec_doc = {
            "organization_id": body.organization_id,
            "project_id": body.project_id,
            "title": body.title,
            "spec_text": body.spec_text,
            "acceptance_criteria": body.acceptance_criteria,
            "target_repos": body.target_repos,
            "priority": body.priority,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.development_specifications.insert_one(spec_doc)
        spec = db.development_specifications.find_one({"_id": result.inserted_id})
        return spec_helper(spec)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{spec_id}")
async def get_specification(spec_id: str, request: Request):
    """Get a specification (org member)."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        spec = db.development_specifications.find_one({"_id": ObjectId(spec_id)})
        if not spec:
            raise HTTPException(status_code=404, detail="Specification not found")

        verify_org_membership(db, spec["organization_id"], workos_user_id)
        return spec_helper(spec)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{spec_id}")
async def update_specification(spec_id: str, body: SpecificationUpdate, request: Request):
    """Update a specification (admin/fde only)."""
    try:
        from auth import require_user_id
        from role_helpers import require_role, verify_org_membership
        workos_user_id = require_user_id(request)

        require_role(request, db, ["admin", "fde"])

        spec = db.development_specifications.find_one({"_id": ObjectId(spec_id)})
        if not spec:
            raise HTTPException(status_code=404, detail="Specification not found")

        verify_org_membership(db, spec["organization_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.development_specifications.update_one({"_id": ObjectId(spec_id)}, {"$set": update_dict})

        spec = db.development_specifications.find_one({"_id": ObjectId(spec_id)})
        return spec_helper(spec)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
