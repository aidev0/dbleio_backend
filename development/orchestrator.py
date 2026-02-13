"""
Development Orchestrator

The central coordinator that bridges:
1. UI Chat (user/client messages)
2. FDM/FDE Review (team member actions)
3. Workflow Executor (runs agents as tools, manages state machine)

Adds model selection per stage — each stage can use a different LLM/CLI tool.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from bson import ObjectId
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

from development.state import DevWorkflowState, DevWorkflowStateStore, DevAgentStateStore, DevUserSession
from development.workflows.pipeline import (
    DEV_PIPELINE_STAGES, DEV_STAGE_MAP, DEV_STAGE_KEYS,
    get_dev_next_stage, get_dev_reject_target, get_dev_stage_index,
)
from development.agents import get_dev_agent, DEV_AGENT_REGISTRY
from development.models import get_model, get_models_by_category, ALL_MODELS


class DevOrchestrator:
    """
    Main orchestrator for development workflows.
    Supports per-stage model selection.
    """

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.state = DevWorkflowState(workflow_id)

    # --- Workflow Lifecycle ---

    @staticmethod
    def create_workflow(project_id: str, title: str, created_by: str,
                        description: str = None, config: dict = None,
                        model_overrides: dict = None) -> dict:
        """
        Create a new dev workflow and initialize all stage nodes.

        Args:
            model_overrides: Optional dict mapping stage_key → model_id to override defaults.
                e.g., {"planner": "gpt-5-2", "developer": "gemini-cli"}
        """
        now = datetime.utcnow()

        project = db.projects.find_one({"_id": ObjectId(project_id)})
        if not project:
            raise ValueError("Project not found")

        workflow_config = config or {}
        if model_overrides:
            workflow_config["model_overrides"] = model_overrides

        doc = {
            "project_id": project_id,
            "organization_id": project.get("organization_id"),
            "title": title,
            "description": description,
            "status": "pending",
            "current_stage": DEV_STAGE_KEYS[0],
            "current_stage_index": 0,
            "config": workflow_config,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        result = db.dev_workflows.insert_one(doc)
        workflow_id = str(result.inserted_id)

        # Initialize all stage nodes as pending
        nodes = []
        for stage in DEV_PIPELINE_STAGES:
            # Determine model for this stage
            stage_model = None
            if model_overrides and stage.key in model_overrides:
                stage_model = model_overrides[stage.key]
            elif stage.default_model_id:
                stage_model = stage.default_model_id

            node = {
                "workflow_id": workflow_id,
                "stage_key": stage.key,
                "stage_index": get_dev_stage_index(stage.key),
                "stage_type": stage.stage_type,
                "status": "pending",
                "model_id": stage_model,
                "allowed_model_categories": stage.allowed_model_categories,
                "input_data": {},
                "output_data": {},
                "created_at": now,
                "updated_at": now,
            }
            nodes.append(node)
        if nodes:
            db.dev_workflow_nodes.insert_many(nodes)

        # Initialize workflow state snapshot
        DevWorkflowStateStore.save(workflow_id, {
            "current_stage": DEV_STAGE_KEYS[0],
            "status": "pending",
            "stage_outputs": {},
            "approval_history": [],
            "feedback_loops": [],
            "model_overrides": model_overrides or {},
        })

        workflow = db.dev_workflows.find_one({"_id": result.inserted_id})
        return _workflow_helper(workflow)

    # --- Model Management ---

    def get_stage_model(self, stage_key: str) -> Optional[str]:
        """Get the configured model_id for a specific stage."""
        node = db.dev_workflow_nodes.find_one({
            "workflow_id": self.workflow_id,
            "stage_key": stage_key,
        })
        if node and node.get("model_id"):
            return node["model_id"]
        stage_def = DEV_STAGE_MAP.get(stage_key)
        return stage_def.default_model_id if stage_def else None

    def set_stage_model(self, stage_key: str, model_id: str) -> dict:
        """Update the model for a specific stage (before it runs)."""
        model = get_model(model_id)
        if not model:
            return {"error": f"Unknown model: {model_id}"}

        stage_def = DEV_STAGE_MAP.get(stage_key)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_key}"}

        if model.category not in stage_def.allowed_model_categories:
            return {"error": f"Model category '{model.category}' not allowed for stage '{stage_key}'. Allowed: {stage_def.allowed_model_categories}"}

        db.dev_workflow_nodes.update_one(
            {"workflow_id": self.workflow_id, "stage_key": stage_key},
            {"$set": {"model_id": model_id, "updated_at": datetime.utcnow()}}
        )

        # Also update in state snapshot
        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("model_overrides", {})[stage_key] = model_id
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        return {"status": "updated", "stage": stage_key, "model_id": model_id}

    # --- Stage Execution ---

    async def advance(self, actor_id: str = None) -> dict:
        """
        Advance the workflow to the next stage.
        Agent/auto stages run automatically; human stages wait.
        """
        self.state.reload()
        current_key = self.state.current_stage
        stage_def = DEV_STAGE_MAP.get(current_key)

        if not stage_def:
            return {"error": f"Unknown stage: {current_key}"}

        if self.state.status in ("completed", "cancelled", "failed"):
            return {"error": f"Workflow is {self.state.status}"}

        # Human stages wait for approval/input
        if stage_def.stage_type == "human":
            return {
                "status": "waiting_human",
                "stage": current_key,
                "label": stage_def.label,
                "approval_required": stage_def.approval_required,
                "message": f"Waiting for human action at '{stage_def.label}'",
            }

        # Agent/auto stages — execute
        if stage_def.agent_name:
            return await self._execute_agent_stage(current_key, stage_def, actor_id)

        return await self._transition_next(current_key, actor_id, {})

    async def _execute_agent_stage(self, stage_key: str, stage_def, actor_id: str = None) -> dict:
        """Execute an agent for the current stage with the configured model."""
        self.state.update_node(stage_key, "running")
        self.state.set_status("running", actor_id)

        try:
            # Get the model override for this stage
            model_id = self.get_stage_model(stage_key)
            agent = get_dev_agent(stage_def.agent_name, self.workflow_id, model_id=model_id)

            context = self._build_agent_context()
            input_data = self._get_stage_input(stage_key)

            output = await agent.run(input_data, context)

            self.state.update_node(stage_key, "completed", output_data=output)

            # Save to workflow state snapshot
            state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
            state_data.setdefault("stage_outputs", {})[stage_key] = output
            DevWorkflowStateStore.save(self.workflow_id, state_data)

            return await self._transition_next(stage_key, actor_id, output)

        except Exception as e:
            self.state.update_node(stage_key, "failed", error=str(e))
            self.state.set_status("failed", actor_id)
            return {"error": str(e), "stage": stage_key}

    async def _transition_next(self, current_key: str, actor_id: str, output: dict) -> dict:
        """Move to the next stage in the pipeline."""
        next_key = get_dev_next_stage(current_key)

        if next_key is None:
            self.state.set_status("completed", actor_id)
            self.state.transition_to(current_key, trigger="completed", actor_id=actor_id)
            return {"status": "completed", "message": "Pipeline complete"}

        self.state.transition_to(next_key, trigger="auto", actor_id=actor_id)

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data["current_stage"] = next_key
        state_data["status"] = "running"
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        return {
            "status": "advanced",
            "from_stage": current_key,
            "to_stage": next_key,
            "to_label": DEV_STAGE_MAP[next_key].label,
        }

    # --- Human Actions ---

    async def approve(self, stage_key: str, actor_id: str, note: str = None) -> dict:
        """Approve a human review stage and advance."""
        stage_def = DEV_STAGE_MAP.get(stage_key)
        if not stage_def or not stage_def.approval_required:
            return {"error": f"Stage '{stage_key}' does not require approval"}

        self.state.update_node(stage_key, "completed", output_data={
            "approved": True,
            "approved_by": actor_id,
            "note": note,
            "approved_at": datetime.utcnow().isoformat(),
        })

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("approval_history", []).append({
            "stage": stage_key,
            "approved": True,
            "actor_id": actor_id,
            "note": note,
            "timestamp": datetime.utcnow().isoformat(),
        })
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        return await self._transition_next(stage_key, actor_id, {"approved": True})

    async def reject(self, stage_key: str, actor_id: str, note: str = None) -> dict:
        """Reject at a human review stage — triggers feedback loop."""
        stage_def = DEV_STAGE_MAP.get(stage_key)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_key}"}

        reject_target = get_dev_reject_target(stage_key)
        if not reject_target:
            return {"error": f"Stage '{stage_key}' has no rejection target"}

        self.state.update_node(stage_key, "completed", output_data={
            "approved": False,
            "rejected_by": actor_id,
            "note": note,
            "rejected_at": datetime.utcnow().isoformat(),
        })

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("approval_history", []).append({
            "stage": stage_key,
            "approved": False,
            "actor_id": actor_id,
            "note": note,
            "timestamp": datetime.utcnow().isoformat(),
        })
        state_data.setdefault("feedback_loops", []).append({
            "from_stage": stage_key,
            "to_stage": reject_target,
            "reason": note or "rejected",
            "timestamp": datetime.utcnow().isoformat(),
        })
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        self.state.transition_to(reject_target, trigger="rejection", actor_id=actor_id,
                                  metadata={"reason": note, "from_stage": stage_key})
        self.state.update_node(reject_target, "pending")

        return {
            "status": "rejected",
            "from_stage": stage_key,
            "to_stage": reject_target,
            "to_label": DEV_STAGE_MAP[reject_target].label,
            "note": note,
        }

    async def submit_human_input(self, stage_key: str, actor_id: str, input_data: dict) -> dict:
        """Submit input for a human stage (e.g., spec, FDM edits)."""
        stage_def = DEV_STAGE_MAP.get(stage_key)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_key}"}

        self.state.update_node(stage_key, "completed", output_data=input_data)

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("stage_outputs", {})[stage_key] = input_data
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        if stage_def.approval_required:
            return {"status": "input_received", "stage": stage_key, "awaiting_approval": True}

        return await self._transition_next(stage_key, actor_id, input_data)

    # --- Chat ---

    async def handle_chat_message(self, user_id: str, message: str, role: str = "user") -> dict:
        """Handle a chat message from the UI."""
        session = DevUserSession.get_or_create(self.workflow_id, user_id)

        chat_context = session.get("chat_context", [])
        chat_context.append({
            "role": role,
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
        })
        DevUserSession.update_chat_context(self.workflow_id, user_id, chat_context)

        db.dev_timeline_entries.insert_one({
            "workflow_id": self.workflow_id,
            "card_type": f"{role}_message",
            "content": message,
            "author_id": user_id,
            "author_role": role,
            "visibility": "public" if role in ("user", "client") else "internal",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })

        return {
            "status": "message_received",
            "workflow_id": self.workflow_id,
            "current_stage": self.state.current_stage,
        }

    # --- Context ---

    def _build_agent_context(self) -> dict:
        """Build the full context dict passed to agents."""
        workflow = db.dev_workflows.find_one({"_id": ObjectId(self.workflow_id)})
        project = db.projects.find_one({"_id": ObjectId(workflow["project_id"])}) if workflow else None

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)

        return {
            "workflow_id": self.workflow_id,
            "project": _safe_serialize(project) if project else {},
            "stage_outputs": state_data.get("stage_outputs", {}),
            "approval_history": state_data.get("approval_history", []),
            "feedback_loops": state_data.get("feedback_loops", []),
            "model_overrides": state_data.get("model_overrides", {}),
            "config": workflow.get("config", {}) if workflow else {},
        }

    def _get_stage_input(self, stage_key: str) -> dict:
        idx = get_dev_stage_index(stage_key)
        if idx == 0:
            return {}
        prev_key = DEV_STAGE_KEYS[idx - 1]
        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        return state_data.get("stage_outputs", {}).get(prev_key, {})

    # --- Run Full Pipeline ---

    async def run_pipeline(self, actor_id: str = None) -> dict:
        """Run from current stage, auto-advancing through agent/auto stages, stopping at human stages."""
        self.state.set_status("running", actor_id)
        results = []

        while True:
            result = await self.advance(actor_id)
            results.append(result)

            if result.get("status") in ("waiting_human", "completed", "error"):
                break
            if result.get("error"):
                break

        return {
            "workflow_id": self.workflow_id,
            "steps_executed": len(results),
            "final_result": results[-1] if results else {},
            "current_stage": self.state.current_stage,
            "workflow_status": self.state.status,
        }


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


def _safe_serialize(doc) -> dict:
    if not doc:
        return {}
    result = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result
