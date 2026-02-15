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

class HumanInputRequest(BaseModel):
    input_data: Dict = {}

class ChatMessageRequest(BaseModel):
    message: str
    role: str = "user"

class ModelUpdateRequest(BaseModel):
    model_id: str


# --- Endpoints ---

@router.post("", status_code=201)
async def create_workflow(body: WorkflowCreate, request: Request):
    """Create a new workflow with specification and first job."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import create_workflow as orch_create_workflow

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
        from src.auth import require_user_id
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
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import get_workflow as orch_get_workflow, get_workflow_agents

        workos_user_id = require_user_id(request)

        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        agents = get_workflow_agents(db, workflow_id)
        workflow["agents"] = agents
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_workflow(workflow_id: str, body: WorkflowUpdate, request: Request):
    """Update workflow title/description. Any org member can edit."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

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
        from src.auth import require_user_id
        from src.role_helpers import require_role

        workos_user_id = require_user_id(request)
        require_role(request, db, ["admin"])

        workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Delete related data
        db.development_workflow_agents.delete_many({"workflow_id": workflow_id})
        db.development_workflow_jobs.delete_many({"workflow_id": workflow_id})
        db.development_events.delete_many({"workflow_id": workflow_id})
        db.timeline_entries.delete_many({"workflow_id": workflow_id})
        db.development_workflows.delete_one({"_id": ObjectId(workflow_id)})

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/agents")
async def get_agents(workflow_id: str, request: Request):
    """Get all agents for a workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import get_workflow as orch_get_workflow, get_workflow_agents

        workos_user_id = require_user_id(request)
        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        return get_workflow_agents(db, workflow_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/agents/{stage_name}")
async def get_agent(workflow_id: str, stage_name: str, request: Request):
    """Get a specific agent by stage name."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import get_workflow as orch_get_workflow

        workos_user_id = require_user_id(request)
        workflow = orch_get_workflow(db, workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        agent = db.development_workflow_agents.find_one({
            "workflow_id": workflow_id,
            "stage_name": stage_name,
        })
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent for stage '{stage_name}' not found")
        agent["_id"] = str(agent["_id"])
        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/events")
async def get_events(workflow_id: str, request: Request, limit: int = 100):
    """Get events for a workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import get_workflow as orch_get_workflow
        from src.development.events import get_workflow_events

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
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership
        from src.development.orchestrator import get_workflow as orch_get_workflow, get_workflow_jobs

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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.orchestrator import handle_approval

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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.orchestrator import handle_approval

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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.orchestrator import handle_approval

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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.orchestrator import handle_approval

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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.orchestrator import create_job, get_workflow as orch_get_workflow
        from src.development.events import log_event

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
            from src.development.orchestrator import PIPELINE_STAGES
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
        from src.auth import require_user_id
        from src.role_helpers import require_role
        from src.development.events import log_event

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


# --- Pipeline Control ---

def _verify_workflow_access(workflow_id: str, workos_user_id: str):
    """Verify the workflow exists and user has access."""
    workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    from src.role_helpers import verify_org_membership
    verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow


@router.post("/{workflow_id}/run")
async def run_pipeline(workflow_id: str, request: Request):
    """Run the pipeline from the current stage."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.run_pipeline(actor_id=workos_user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/advance")
async def advance_stage(workflow_id: str, request: Request):
    """Advance one stage in the pipeline."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.advance(actor_id=workos_user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_name}/approve")
async def approve_stage(workflow_id: str, stage_name: str, body: ApprovalRequest, request: Request):
    """Approve or reject a human review stage."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)

        if body.approved:
            result = await orchestrator.approve(stage_name, workos_user_id, body.note)
        else:
            result = await orchestrator.reject(stage_name, workos_user_id, body.note)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_name}/input")
async def submit_stage_input(workflow_id: str, stage_name: str, body: HumanInputRequest, request: Request):
    """Submit human input for a stage."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.submit_human_input(stage_name, workos_user_id, body.input_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Agent Model Selection ---

@router.get("/{workflow_id}/agents/{stage_name}/model")
async def get_agent_model(workflow_id: str, stage_name: str, request: Request):
    """Get the current model for a specific agent."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        model_id = orchestrator.get_stage_model(stage_name)

        from src.development.models import get_model
        model = get_model(model_id) if model_id else None

        return {
            "stage_name": stage_name,
            "model_id": model_id,
            "model_name": model.name if model else None,
            "model_provider": model.provider if model else None,
            "model_category": model.category if model else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{workflow_id}/agents/{stage_name}/model")
async def set_agent_model(workflow_id: str, stage_name: str, body: ModelUpdateRequest, request: Request):
    """Set the model for a specific agent."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = orchestrator.set_stage_model(stage_name, body.model_id)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Available Models ---

@router.get("/models/available")
async def get_available_models(request: Request):
    """Get all available models grouped by category."""
    try:
        from src.auth import require_user_id
        require_user_id(request)

        from src.development.models import (
            get_available_llms, get_available_clis,
            get_available_video_models, get_available_image_models,
            get_replicate_models,
        )
        return {
            "llm": get_available_llms(),
            "cli": get_available_clis(),
            "video": get_available_video_models(),
            "image": get_available_image_models(),
            "replicate": get_replicate_models(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Chat ---

@router.post("/{workflow_id}/chat")
async def send_chat_message(workflow_id: str, body: ChatMessageRequest, request: Request):
    """Send a chat message in the workflow context."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.handle_chat_message(workos_user_id, body.message, body.role)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- State & History ---

@router.get("/{workflow_id}/state")
async def get_workflow_state(workflow_id: str, request: Request):
    """Get the full workflow state snapshot."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.state import DevWorkflowStateStore
        state = DevWorkflowStateStore.load(workflow_id)
        return state or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/agent-states")
async def get_agent_states(workflow_id: str, request: Request):
    """Get all agent states for a workflow."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.state import DevAgentStateStore
        agents = DevAgentStateStore.list_agents(workflow_id)
        return agents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/transitions")
async def get_transitions(workflow_id: str, request: Request):
    """Get the transition audit trail."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.state import DevWorkflowState
        state = DevWorkflowState(workflow_id)
        return state.get_transition_history()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/sessions")
async def get_user_sessions(workflow_id: str, request: Request):
    """Get all active user sessions for a workflow."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from src.development.state import DevUserSession
        sessions = DevUserSession.get_active_sessions(workflow_id)
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
