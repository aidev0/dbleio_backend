#!/usr/bin/env python3
"""
Content Workflows API routes.
Exposes the content generation orchestrator to the frontend.
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

router = APIRouter(prefix="/api/content/workflows", tags=["content-workflows"])


# --- Models ---

class ContentWorkflowCreate(BaseModel):
    brand_id: str
    title: str
    description: Optional[str] = None
    config: Optional[dict] = None


class ContentWorkflowUpdate(BaseModel):
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


class GenerateConceptsRequest(BaseModel):
    num: int = Field(ge=1, le=50)
    tone: str


# --- Helpers ---

def _workflow_helper(doc) -> dict:
    if not doc:
        return {}
    return {
        "_id": str(doc["_id"]),
        "brand_id": doc.get("brand_id"),
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
        "input_data": doc.get("input_data", {}),
        "output_data": doc.get("output_data", {}),
        "error": doc.get("error"),
        "started_at": doc.get("started_at"),
        "completed_at": doc.get("completed_at"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


# --- Endpoints ---

@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
async def create_content_workflow(body: ContentWorkflowCreate, request: Request):
    """Create a new content generation workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        brand = db.brands.find_one({"_id": ObjectId(body.brand_id)})
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")

        verify_org_membership(db, brand["organization_id"], workos_user_id)

        from content_generation.orchestrator import ContentOrchestrator
        workflow = ContentOrchestrator.create_workflow(
            brand_id=body.brand_id,
            title=body.title,
            description=body.description,
            created_by=workos_user_id,
            config=body.config,
        )
        return workflow
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_content_workflows(request: Request, brand_id: Optional[str] = None,
                                  organization_id: Optional[str] = None):
    """List content workflows."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)

        query = {}
        if brand_id:
            brand = db.brands.find_one({"_id": ObjectId(brand_id)})
            if not brand:
                raise HTTPException(status_code=404, detail="Brand not found")
            from role_helpers import verify_org_membership
            verify_org_membership(db, brand["organization_id"], workos_user_id)
            query["brand_id"] = brand_id
        elif organization_id:
            from role_helpers import verify_org_membership
            verify_org_membership(db, organization_id, workos_user_id)
            query["organization_id"] = organization_id
        else:
            user = db.users.find_one({"workos_user_id": workos_user_id})
            if not user:
                return []
            org_ids = [o["_id"] for o in user.get("organizations", [])]
            if "admin" not in user.get("roles", []):
                query["organization_id"] = {"$in": org_ids}

        workflows = list(db.content_workflows.find(query).sort("created_at", -1))
        return [_workflow_helper(w) for w in workflows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_content_workflow(workflow_id: str, request: Request):
    """Get a single content workflow with its nodes."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        result = _workflow_helper(workflow)
        nodes = list(db.content_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        result["nodes"] = [_node_helper(n) for n in nodes]
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_content_workflow(workflow_id: str, body: ContentWorkflowUpdate, request: Request):
    """Update a content workflow."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id)

        update_dict = body.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Deep-merge config so stage_settings updates don't nuke other config keys
        if "config" in update_dict and workflow.get("config"):
            merged_config = {**workflow["config"], **update_dict["config"]}
            update_dict["config"] = merged_config

        update_dict["updated_at"] = datetime.utcnow()
        db.content_workflows.update_one({"_id": ObjectId(workflow_id)}, {"$set": update_dict})

        updated = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        return _workflow_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}", status_code=204)
async def delete_content_workflow(workflow_id: str, request: Request):
    """Delete a content workflow and all associated data."""
    try:
        from auth import require_user_id
        from role_helpers import verify_org_membership
        workos_user_id = require_user_id(request)

        workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        verify_org_membership(db, workflow["organization_id"], workos_user_id, required_roles=["owner"])

        db.content_workflows.delete_one({"_id": ObjectId(workflow_id)})
        db.content_workflow_nodes.delete_many({"workflow_id": workflow_id})
        db.content_workflow_transitions.delete_many({"workflow_id": workflow_id})
        db.content_workflow_states.delete_many({"workflow_id": workflow_id})
        db.content_agent_states.delete_many({"workflow_id": workflow_id})
        db.content_user_sessions.delete_many({"workflow_id": workflow_id})
        db.content_timeline_entries.delete_many({"workflow_id": workflow_id})
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

        nodes = list(db.content_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
        return [_node_helper(n) for n in nodes]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/run")
async def run_pipeline(workflow_id: str, request: Request):
    """Run the pipeline from the current stage (auto-advances through agent stages)."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
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

        from content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
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

        from content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)

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

        from content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.submit_human_input(stage_key, workos_user_id, body.input_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/chat")
async def send_chat_message(workflow_id: str, body: ChatMessageRequest, request: Request):
    """Send a chat message in the workflow context."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from content_generation.orchestrator import ContentOrchestrator
        orchestrator = ContentOrchestrator(workflow_id)
        result = await orchestrator.handle_chat_message(workos_user_id, body.message, body.role)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/state")
async def get_workflow_state(workflow_id: str, request: Request):
    """Get the full workflow state snapshot."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        _verify_workflow_access(workflow_id, workos_user_id)

        from content_generation.state import WorkflowStateStore
        state = WorkflowStateStore.load(workflow_id)
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

        from content_generation.state import AgentStateStore
        agents = AgentStateStore.list_agents(workflow_id)
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

        from content_generation.state import WorkflowState
        state = WorkflowState(workflow_id)
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

        from content_generation.state import UserSession
        sessions = UserSession.get_active_sessions(workflow_id)
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Concept Generation ---

@router.post("/{workflow_id}/generate-concepts")
async def generate_concepts(workflow_id: str, body: GenerateConceptsRequest, request: Request):
    """Use AI to generate content concepts with full workflow context."""
    try:
        from auth import require_user_id
        workos_user_id = require_user_id(request)
        workflow = _verify_workflow_access(workflow_id, workos_user_id)

        # Gather context
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow.get("brand_id") else None
        campaign = None
        strategy = None
        audience = None

        config = workflow.get("config") or {}
        campaign_id = config.get("campaign_id")
        if campaign_id:
            campaign = db.campaigns.find_one({"_id": ObjectId(campaign_id)})
            # Get first strategy for this campaign
            strategy = db.strategies.find_one({"campaign_id": campaign_id})

        if brand:
            audience = db.audiences.find_one({"brand_id": workflow["brand_id"]})

        # Build context string
        context_parts = []
        if brand:
            context_parts.append(f"Brand: {brand.get('name', '')} — {brand.get('industry', '')}. {brand.get('description', '')}. Product: {brand.get('product', '')}")
        if campaign:
            context_parts.append(f"Campaign: {campaign.get('name', '')}. Goal: {campaign.get('campaign_goal', '')}. Platform: {campaign.get('platform', '')}")
        if strategy:
            context_parts.append(f"Strategy: {strategy.get('name', '')}. Budget: {strategy.get('budget_amount', '')} {strategy.get('budget_type', '')}. Objectives: {strategy.get('objectives', '')}")
        if audience:
            context_parts.append(f"Audience: {audience.get('name', '')}. Demographics: {audience.get('demographics', '')}. Interests: {audience.get('interests', '')}")

        # Check prior stage outputs from workflow state
        try:
            from content_generation.state import WorkflowStateStore
            state = WorkflowStateStore.load(workflow_id)
            if state:
                if state.get("research"):
                    context_parts.append(f"Research findings: {str(state['research'])[:500]}")
                if state.get("strategy_assets"):
                    context_parts.append(f"Strategy assets: {str(state['strategy_assets'])[:500]}")
        except Exception:
            pass

        context_str = "\n".join(context_parts) if context_parts else "No additional context available."

        system_prompt = f"""You are a creative content strategist. Generate content concepts based on the following context.

{context_str}

Return a JSON array of exactly {body.num} concepts with tone "{body.tone}". Each concept must have:
- "title": a catchy concept title
- "hook": the opening hook (1-2 sentences)
- "script": a script outline (3-5 bullet points)
- "messaging": an array of 2-3 key messaging points

Return ONLY valid JSON: {{"concepts": [...]}}"""

        import anthropic
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="Anthropic API key not configured")

        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Generate {body.num} content concepts with a {body.tone} tone."}],
        )

        assistant_text = response.content[0].text

        # Track usage
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        from chat import save_llm_usage
        save_llm_usage(
            user_id=workos_user_id,
            campaign_id=campaign_id,
            provider="anthropic",
            model_name="claude-sonnet-4-20250514",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            mode="concept_generation",
        )

        # Parse JSON from response
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', assistant_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"concepts": []}

        # Tag each concept with the tone
        for concept in result.get("concepts", []):
            concept["tone"] = body.tone

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Helpers ---

def _verify_workflow_access(workflow_id: str, workos_user_id: str):
    """Verify user has access to this workflow's organization."""
    workflow = db.content_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    from role_helpers import verify_org_membership
    verify_org_membership(db, workflow["organization_id"], workos_user_id)
    return workflow
