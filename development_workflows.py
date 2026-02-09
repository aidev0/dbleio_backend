#!/usr/bin/env python3
"""
Development Workflows API - CRUD, approval gates, and workflow management.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
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

router = APIRouter(prefix="/api/development/workflows", tags=["development-workflows"])


# --- Models ---

class WorkflowCreate(BaseModel):
    organization_id: str
    project_id: Optional[str] = None
    title: str
    description: Optional[str] = ""
    spec_title: str
    spec_text: str
    acceptance_criteria: Optional[str] = None
    target_repos: List[str] = []
    priority: str = "medium"
    agent_config: Optional[Dict] = None

class ApprovalRequest(BaseModel):
    approved: bool
    note: Optional[str] = ""

class WorkflowUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class RetryRequest(BaseModel):
    job_id: Optional[str] = None


# --- Endpoints ---

@router.post("", status_code=201)
async def create_workflow(body: WorkflowCreate, request: Request):
    """Create a new workflow with specification and first job."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        from development_orchestrator import create_workflow as orch_create_workflow

        workos_user_id = require_user_id(request)
        verify_org_membership(db, body.organization_id, workos_user_id)

        # Enforce active workflow limit
        user = db.users.find_one({"workos_user_id": workos_user_id})
        user_roles = user.get("roles", []) if user else []
        if "admin" not in user_roles:
            explicit_limit = user.get("active_workflow_limit") if user else None
            if explicit_limit is not None:
                limit = explicit_limit
            elif user and user.get("subscription_tier") == "dble_team":
                limit = 2
            else:
                limit = 1

            active_statuses = ["pending", "running", "waiting_approval"]
            active_count = db.development_workflows.count_documents({
                "created_by": workos_user_id,
                "status": {"$in": active_statuses},
            })
            if active_count >= limit:
                raise HTTPException(
                    status_code=403,
                    detail=f"Active workflow limit reached ({limit}). Complete or cancel an existing workflow before creating a new one.",
                )

        # Create specification first
        now = datetime.utcnow()
        spec_doc = {
            "organization_id": body.organization_id,
            "project_id": body.project_id,
            "title": body.spec_title,
            "spec_text": body.spec_text,
            "acceptance_criteria": body.acceptance_criteria,
            "target_repos": body.target_repos,
            "priority": body.priority,
            "created_by": workos_user_id,
            "created_at": now,
            "updated_at": now,
        }
        spec_result = db.development_specifications.insert_one(spec_doc)
        spec_id = str(spec_result.inserted_id)

        # Create workflow
        workflow = orch_create_workflow(
            db,
            organization_id=body.organization_id,
            project_id=body.project_id,
            specification_id=spec_id,
            title=body.title,
            description=body.description or "",
            agent_config=body.agent_config,
            created_by=workos_user_id,
        )
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_workflows(
    request: Request,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """List workflows scoped to user's organizations."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        # Get user's org IDs from embedded organizations
        user = db.users.find_one({"workos_user_id": workos_user_id})
        org_ids = [o["_id"] for o in (user.get("organizations", []) if user else [])]
        if not org_ids:
            return []

        query: Dict = {"organization_id": {"$in": org_ids}}
        if status:
            query["status"] = status
        if project_id:
            query["project_id"] = project_id

        workflows = list(db.development_workflows.find(query).sort("created_at", -1))
        for wf in workflows:
            wf["_id"] = str(wf["_id"])
        return workflows
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, request: Request):
    """Get a workflow with its nodes."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        from development_orchestrator import get_workflow as orch_get_workflow, get_workflow_nodes

        workos_user_id = require_user_id(request)

        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        nodes = get_workflow_nodes(db, workflow_id)
        workflow["nodes"] = nodes
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_workflow(workflow_id: str, body: WorkflowUpdate, request: Request):
    """Update workflow title/description. Any org member can edit."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)

        workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        updates = {"updated_at": datetime.utcnow()}
        if body.title is not None:
            updates["title"] = body.title
        if body.description is not None:
            updates["description"] = body.description

        db.development_workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": updates}
        )

        updated = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
        updated["_id"] = str(updated["_id"])
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, request: Request):
    """Delete a workflow. Admin only."""
    try:
        from auth import require_user_id
        from role_helpers import require_role

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin"])

        workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Delete related data
        db.development_workflow_nodes.delete_many({"workflow_id": workflow_id})
        db.development_workflow_jobs.delete_many({"workflow_id": workflow_id})
        db.development_events.delete_many({"workflow_id": workflow_id})
        db.timeline_entries.delete_many({"workflow_id": workflow_id})
        db.development_workflows.delete_one({"_id": ObjectId(workflow_id)})

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/nodes")
async def get_nodes(workflow_id: str, request: Request):
    """Get all nodes for a workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        from development_orchestrator import get_workflow as orch_get_workflow, get_workflow_nodes

        workos_user_id = require_user_id(request)
        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        return get_workflow_nodes(db, workflow_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/events")
async def get_events(workflow_id: str, request: Request, limit: int = 100):
    """Get events for a workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        from development_orchestrator import get_workflow as orch_get_workflow
        from development_events import get_workflow_events

        workos_user_id = require_user_id(request)
        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        return get_workflow_events(db, workflow_id, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/jobs")
async def get_jobs(workflow_id: str, request: Request):
    """Get all jobs for a workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        from development_orchestrator import get_workflow as orch_get_workflow, get_workflow_jobs

        workos_user_id = require_user_id(request)
        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        return get_workflow_jobs(db, workflow_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Approval Endpoints ---

@router.post("/{workflow_id}/approve-plan")
async def approve_plan(workflow_id: str, body: ApprovalRequest, request: Request):
    """FDE approves or rejects the plan."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_orchestrator import handle_approval

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "fde"])

        handle_approval(db, workflow_id, "fde_plan", body.approved,
                        note=body.note or "", approved_by=workos_user_id)
        return {"success": True, "approved": body.approved}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/approve-pr")
async def approve_pr(workflow_id: str, body: ApprovalRequest, request: Request):
    """FDE approves or rejects the PR."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_orchestrator import handle_approval

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "fde"])

        handle_approval(db, workflow_id, "fde_pr", body.approved,
                        note=body.note or "", approved_by=workos_user_id)
        return {"success": True, "approved": body.approved}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/approve-qa")
async def approve_qa(workflow_id: str, body: ApprovalRequest, request: Request):
    """QA approves or rejects the deployment."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_orchestrator import handle_approval

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "qa"])

        handle_approval(db, workflow_id, "qa", body.approved,
                        note=body.note or "", approved_by=workos_user_id)
        return {"success": True, "approved": body.approved}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/approve-client")
async def approve_client(workflow_id: str, body: ApprovalRequest, request: Request):
    """Client approves or provides feedback."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_orchestrator import handle_approval

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "client"])

        handle_approval(db, workflow_id, "client", body.approved,
                        note=body.note or "", approved_by=workos_user_id)
        return {"success": True, "approved": body.approved}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Retry / Cancel ---

@router.post("/{workflow_id}/retry")
async def retry_workflow(workflow_id: str, body: RetryRequest, request: Request):
    """Retry a failed workflow or specific job."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_orchestrator import create_job, get_workflow as orch_get_workflow
        from development_events import log_event

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "fde"])

        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if body.job_id:
            # Retry specific job
            job = db.development_workflow_jobs.find_one({"_id": ObjectId(body.job_id)})
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            db.development_workflow_jobs.update_one(
                {"_id": ObjectId(body.job_id)},
                {"$set": {"status": "queued", "claimed_by": None, "run_after": datetime.utcnow()}}
            )
        else:
            # Re-queue from current stage
            current_stage = workflow.get("current_stage", "")
            from development_orchestrator import PIPELINE_STAGES
            stage_def = next((s for s in PIPELINE_STAGES if s["name"] == current_stage), None)
            if stage_def and stage_def["job_type"]:
                create_job(db, workflow_id, stage_def["job_type"], stage_name=current_stage)

        # Reset workflow status
        db.development_workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": {"status": "running", "updated_at": datetime.utcnow()}}
        )

        log_event(db, workflow_id, "workflow_retried", "user", workos_user_id,
                  f"Workflow retried by {workos_user_id}")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str, request: Request):
    """Cancel a running workflow."""
    try:
        from auth import require_user_id
        from role_helpers import require_role
        from development_events import log_event

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin", "fde"])

        workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        db.development_workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}}
        )

        # Cancel pending jobs
        db.development_workflow_jobs.update_many(
            {"workflow_id": workflow_id, "status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "cancelled"}}
        )

        log_event(db, workflow_id, "workflow_cancelled", "user", workos_user_id,
                  f"Workflow cancelled by {workos_user_id}")

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
