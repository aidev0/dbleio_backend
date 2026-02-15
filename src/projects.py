#!/usr/bin/env python3
"""
Projects API - Scoped to organizations for multi-tenancy.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
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
db.projects.create_index("organization_id")
db.projects.create_index(
    [("organization_id", ASCENDING), ("slug", ASCENDING)],
    unique=True
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


# --- Models ---

class RepoConfig(BaseModel):
    name: str
    url: str
    branch: Optional[str] = "main"
    path: Optional[str] = None  # subdirectory within the repo

class DeploymentTarget(BaseModel):
    name: str
    provider: str  # heroku, vercel, docker, custom
    url: Optional[str] = None
    health_check_url: Optional[str] = None
    config: Optional[Dict] = None

class DeploymentConfig(BaseModel):
    targets: List[DeploymentTarget] = []

class ProjectCreate(BaseModel):
    organization_id: str
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    repos: List[RepoConfig] = []
    deployment_config: Optional[DeploymentConfig] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    repos: Optional[List[RepoConfig]] = None
    deployment_config: Optional[DeploymentConfig] = None


# --- Helpers ---

def project_helper(project) -> dict:
    return {
        "_id": str(project["_id"]),
        "organization_id": project["organization_id"],
        "name": project["name"],
        "slug": project.get("slug"),
        "description": project.get("description"),
        "repos": project.get("repos", []),
        "deployment_config": project.get("deployment_config"),
        "created_by": project.get("created_by"),
        "created_at": project.get("created_at"),
        "updated_at": project.get("updated_at"),
    }


def _generate_project_slug(org_id: str, name: str) -> str:
    import re
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        slug = "project"
    base_slug = slug
    counter = 1
    while db.projects.find_one({"organization_id": org_id, "slug": slug}):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_project(body: ProjectCreate, request: Request):
    """Create a project within an organization."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        verify_org_membership(db, body.organization_id, workos_user_id)

        slug = body.slug if body.slug else _generate_project_slug(body.organization_id, body.name)

        existing = db.projects.find_one({
            "organization_id": body.organization_id,
            "slug": slug
        })
        if existing:
            raise HTTPException(status_code=409, detail=f"Project slug '{slug}' already exists in this org")

        now = datetime.utcnow()
        project_doc = {
            "organization_id": body.organization_id,
            "name": body.name,
            "slug": slug,
            "description": body.description,
            "repos": [r.model_dump() for r in body.repos] if body.repos else [],
            "deployment_config": body.deployment_config.model_dump() if body.deployment_config else None,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        result = db.projects.insert_one(project_doc)
        project = db.projects.find_one({"_id": result.inserted_id})
        return project_helper(project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_projects(request: Request, organization_id: Optional[str] = None):
    """List projects scoped to the user's organizations."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        # Get user's org IDs from embedded organizations
        user = db.users.find_one({"workos_user_id": workos_user_id})
        org_ids = [o["_id"] for o in (user.get("organizations", []) if user else [])]
        if not org_ids:
            return []

        query = {"organization_id": {"$in": org_ids}}
        if organization_id:
            if organization_id not in org_ids:
                raise HTTPException(status_code=403, detail="Not a member of this organization")
            query = {"organization_id": organization_id}

        projects = list(db.projects.find(query))
        return [project_helper(p) for p in projects]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}")
async def get_project(project_id: str, request: Request):
    """Get a single project (must be org member)."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        verify_org_membership(db, project["organization_id"], workos_user_id)
        return project_helper(project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate, request: Request):
    """Update a project (must be org member)."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        verify_org_membership(db, project["organization_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Serialize nested models
        if "repos" in update_dict and update_dict["repos"] is not None:
            update_dict["repos"] = [r if isinstance(r, dict) else r.model_dump() for r in update_dict["repos"]]
        if "deployment_config" in update_dict and update_dict["deployment_config"] is not None:
            dc = update_dict["deployment_config"]
            update_dict["deployment_config"] = dc if isinstance(dc, dict) else dc.model_dump()

        update_dict["updated_at"] = datetime.utcnow()
        db.projects.update_one({"_id": ObjectId(project_id)}, {"$set": update_dict})

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        return project_helper(project)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, request: Request):
    """Delete a project (must be org owner/admin)."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        verify_org_membership(db, project["organization_id"], workos_user_id, required_roles=["owner", "admin"])
        db.projects.delete_one({"_id": ObjectId(project_id)})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
