#!/usr/bin/env python3
"""
Development Workflows API routes.
Exposes the development orchestrator to the frontend with model selection support.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
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

router = APIRouter(prefix="/api/dev/workflows", tags=["dev-workflows"])


# --- Models ---

class DevWorkflowCreate(BaseModel):
    project_id: str
    title: str
    description: Optional[str] = None
    config: Optional[dict] = None
    model_overrides: Optional[Dict[str, str]] = None  # stage_key → model_id


class DevWorkflowUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    config: Optional[dict] = None


class ApprovalRequest(BaseModel):
    approved: bool
    note: Optional[str] = None


class HumanInputRequest(BaseModel):
    input_data: dict = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    message: str
    role: str = "user"


class ModelUpdateRequest(BaseModel):
    model_id: str


# --- Helpers ---

def _workflow_helper(doc) -> dict:
    if not doc:
        return {}
    return {
        "_id": str(doc["_id"]),
        "project_id": doc.get("project_id"),
        "organization_id": doc.get("organization_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "status": doc.get("status"),
        "current_stage": doc.get("current_stage"),
        "current_stage_index": doc.get("current_stage_index", 0),
        "config": doc.get("config", {}),
        "created_by": doc.get("created_by"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _node_helper(doc) -> dict:
    if not doc:
        return {}
    return {
        "_id": str(doc["_id"]),
        "workflow_id": doc.get("workflow_id"),
        "stage_key": doc.get("stage_key"),
        "stage_index": doc.get("stage_index"),
        "stage_type": doc.get("stage_type"),
        "status": doc.get("status"),
        "model_id": doc.get("model_id"),
        "allowed_model_categories": doc.get("allowed_model_categories", []),
        "input_data": doc.get("input_data", {}),
        "output_data": doc.get("output_data", {}),
        "error": doc.get("error"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _verify_workflow_access(workflow_id: str, workos_user_id: str):
    workflow = db.dev_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    from role_helpers import verify_org_membership
    verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow


# --- CRUD ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_dev_workflow(body: DevWorkflowCreate, request: Request):
    """Create a new development workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        project = db.projects.find_one({"_id": ObjectId(body.project_id)})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        verify_org_membership(db, project["organization_id"], workos_user_id)

        from development.orchestrator import DevOrchestrator
        workflow = DevOrchestrator.create_workflow(
            project_id=body.project_id,
            title=body.title,
            description=body.description,
            created_by=workos_user_id,
            config=body.config,
            model_overrides=body.model_overrides,
        )
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_dev_workflows(request: Request, project_id: Optional[str] = None,
                              organization_id: Optional[str] = None):
    """List development workflows."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        query = {}
        if project_id:
            project = db.projects.find_one({"_id": ObjectId(project_id)})
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            from role_helpers import verify_org_membership
            verify_org_membership(db, project["organization_id"], workos_user_id)
            query["project_id"] = project_id
        elif organization_id:
            from role_helpers import verify_org_membership
            verify_org_membership(db, organization_id, workos_user_id)
            query["organization_id"] = organization_id

        workflows = list(db.dev_workflows.find(query).sort("created_at", -1))
        return [_workflow_helper(w) for w in workflows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_dev_workflow(workflow_id: str, request: Request):
    """Get a single dev workflow with its nodes."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        result = _workflow_helper(workflow)
        nodes = list(db.dev_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        result["nodes"] = [_node_helper(n) for n in nodes]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_dev_workflow(workflow_id: str, body: DevWorkflowUpdate, request: Request):
    """Update a dev workflow."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_dict["updated_at"] = datetime.utcnow()
        db.dev_workflows.update_one({"_id": ObjectId(workflow_id)}, {"$set": update_dict})

        updated = db.dev_workflows.find_one({"_id": ObjectId(workflow_id)})
        return _workflow_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}", status_code=204)
async def delete_dev_workflow(workflow_id: str, request: Request):
    """Delete a dev workflow and all associated data."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        verify_org_membership(db, workflow["organization_id"], workos_user_id, required_roles=["owner"])

        db.dev_workflows.delete_one({"_id": ObjectId(workflow_id)})
        db.dev_workflow_nodes.delete_many({"workflow_id": workflow_id})
        db.dev_workflow_transitions.delete_many({"workflow_id": workflow_id})
        db.dev_workflow_states.delete_many({"workflow_id": workflow_id})
        db.dev_agent_states.delete_many({"workflow_id": workflow_id})
        db.dev_user_sessions.delete_many({"workflow_id": workflow_id})
        db.dev_timeline_entries.delete_many({"workflow_id": workflow_id})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Pipeline Control ---

@router.get("/{workflow_id}/nodes")
async def get_nodes(workflow_id: str, request: Request):
    """Get all pipeline nodes for a workflow."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        nodes = list(db.dev_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        return [_node_helper(n) for n in nodes]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/run")
async def run_pipeline(workflow_id: str, request: Request):
    """Run the pipeline from the current stage."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.advance(actor_id=workos_user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_key}/approve")
async def approve_stage(workflow_id: str, stage_key: str, body: ApprovalRequest, request: Request):
    """Approve or reject a human review stage."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)

        if body.approved:
            result = await orchestrator.approve(stage_key, workos_user_id, body.note)
        else:
            result = await orchestrator.reject(stage_key, workos_user_id, body.note)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/stages/{stage_key}/input")
async def submit_stage_input(workflow_id: str, stage_key: str, body: HumanInputRequest, request: Request):
    """Submit human input for a stage."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = await orchestrator.submit_human_input(stage_key, workos_user_id, body.input_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Model Selection ---

@router.get("/{workflow_id}/stages/{stage_key}/model")
async def get_stage_model(workflow_id: str, stage_key: str, request: Request):
    """Get the current model for a specific stage."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        model_id = orchestrator.get_stage_model(stage_key)

        from development.models import get_model
        model = get_model(model_id) if model_id else None

        return {
            "stage_key": stage_key,
            "model_id": model_id,
            "model_name": model.name if model else None,
            "model_provider": model.provider if model else None,
            "model_category": model.category if model else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{workflow_id}/stages/{stage_key}/model")
async def set_stage_model(workflow_id: str, stage_key: str, body: ModelUpdateRequest, request: Request):
    """Set the model for a specific stage."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
        orchestrator = DevOrchestrator(workflow_id)
        result = orchestrator.set_stage_model(stage_key, body.model_id)

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
        from auth import require_user_id
        require_user_id(request)

        from development.models import (
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.orchestrator import DevOrchestrator
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.state import DevWorkflowStateStore
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.state import DevAgentStateStore
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.state import DevWorkflowState
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
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from development.state import DevUserSession
        sessions = DevUserSession.get_active_sessions(workflow_id)
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
