"""
Content Generation Orchestrator

The central coordinator that bridges:
1. UI Chat (user/client messages)
2. FDM Review (team member actions)
3. Workflow Executor (runs agents as tools, manages state machine)

Responsibilities:
- Receives messages from chat UI and routes them to the right handler
- Manages the workflow state machine (transitions, approvals, rejections)
- Invokes agents as tools for each pipeline stage
- Persists all state (workflow_states, agent_states, user_sessions)
- Handles feedback loops (QA reject → concepts, RL → research)
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

from src.content_generation.state import WorkflowState, WorkflowStateStore, AgentStateStore, UserSession
from src.content_generation.workflows.pipeline import (
    PIPELINE_STAGES, STAGE_MAP, STAGE_KEYS,
    get_next_stage, get_reject_target, get_stage_index,
)
from src.content_generation.agents import get_agent, AGENT_REGISTRY


class ContentOrchestrator:
    """
    Main orchestrator for content generation workflows.
    Acts as the single entry point for all workflow operations.
    """

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.state = WorkflowState(workflow_id)

    # --- Workflow Lifecycle ---

    @staticmethod
    def create_workflow(brand_id: str, title: str, created_by: str,
                        description: str = None, config: dict = None) -> dict:
        """Create a new content workflow and initialize all stage nodes."""
        now = datetime.utcnow()

        # Verify brand exists
        brand = db.brands.find_one({"_id": ObjectId(brand_id)})
        if not brand:
            raise ValueError("Brand not found")

        doc = {
            "brand_id": brand_id,
            "organization_id": brand["organization_id"],
            "title": title,
            "description": description,
            "status": "pending",
            "current_stage": STAGE_KEYS[0],
            "current_stage_index": 0,
            "config": config or {},
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        result = db.content_workflows.insert_one(doc)
        workflow_id = str(result.inserted_id)

        # Initialize all stage nodes as pending
        nodes = []
        for stage in PIPELINE_STAGES:
            node = {
                "workflow_id": workflow_id,
                "stage_key": stage.key,
                "stage_index": get_stage_index(stage.key),
                "stage_type": stage.stage_type,
                "status": "pending",
                "input_data": {},
                "output_data": {},
                "created_at": now,
                "updated_at": now,
            }
            nodes.append(node)
        if nodes:
            db.content_workflow_nodes.insert_many(nodes)

        # Initialize workflow state snapshot
        WorkflowStateStore.save(workflow_id, {
            "current_stage": STAGE_KEYS[0],
            "status": "pending",
            "stage_outputs": {},
            "approval_history": [],
            "feedback_loops": [],
        })

        workflow = db.content_workflows.find_one({"_id": result.inserted_id})
        return _workflow_helper(workflow)

    # --- Stage Execution ---

    async def advance(self, actor_id: str = None) -> dict:
        """
        Attempt to advance the workflow to the next stage.
        If current stage is an agent/auto stage, execute the agent.
        If current stage is human, wait for approval/input.
        Returns the updated workflow state.
        """
        self.state.reload()
        current_key = self.state.current_stage
        stage_def = STAGE_MAP.get(current_key)

        if not stage_def:
            return {"error": f"Unknown stage: {current_key}"}

        if self.state.status in ("completed", "cancelled", "failed"):
            return {"error": f"Workflow is {self.state.status}"}

        # For human stages, we don't auto-advance — wait for approval
        if stage_def.stage_type == "human":
            return {
                "status": "waiting_human",
                "stage": current_key,
                "label": stage_def.label,
                "approval_required": stage_def.approval_required,
                "message": f"Waiting for human action at '{stage_def.label}'",
            }

        # For agent/auto stages, execute the agent
        if stage_def.agent_name:
            return await self._execute_agent_stage(current_key, stage_def, actor_id)

        # If no agent, just advance
        return await self._transition_next(current_key, actor_id, {})

    async def _execute_agent_stage(self, stage_key: str, stage_def, actor_id: str = None) -> dict:
        """Execute an agent for the current stage."""
        # Mark node as running
        self.state.update_node(stage_key, "running")
        self.state.set_status("running", actor_id)

        try:
            agent = get_agent(stage_def.agent_name, self.workflow_id)

            # Build context with all prior stage outputs
            context = self._build_agent_context()
            input_data = self._get_stage_input(stage_key)

            # Run the agent
            output = await agent.run(input_data, context)

            # Mark node as completed
            self.state.update_node(stage_key, "completed", output_data=output)

            # Save to workflow state snapshot
            state_data = WorkflowStateStore.get_state_data(self.workflow_id)
            state_data.setdefault("stage_outputs", {})[stage_key] = output
            WorkflowStateStore.save(self.workflow_id, state_data)

            # Transition to next stage
            return await self._transition_next(stage_key, actor_id, output)

        except Exception as e:
            self.state.update_node(stage_key, "failed", error=str(e))
            self.state.set_status("failed", actor_id)
            return {"error": str(e), "stage": stage_key}

    async def _transition_next(self, current_key: str, actor_id: str, output: dict) -> dict:
        """Move to the next stage in the pipeline."""
        next_key = get_next_stage(current_key)

        if next_key is None:
            # Pipeline complete
            self.state.set_status("completed", actor_id)
            self.state.transition_to(current_key, trigger="completed", actor_id=actor_id)
            return {"status": "completed", "message": "Pipeline complete"}

        self.state.transition_to(next_key, trigger="auto", actor_id=actor_id)

        # Update workflow state snapshot
        state_data = WorkflowStateStore.get_state_data(self.workflow_id)
        state_data["current_stage"] = next_key
        state_data["status"] = "running"
        WorkflowStateStore.save(self.workflow_id, state_data)

        return {
            "status": "advanced",
            "from_stage": current_key,
            "to_stage": next_key,
            "to_label": STAGE_MAP[next_key].label,
        }

    # --- Human Actions (Approvals / Rejections) ---

    async def approve(self, stage_key: str, actor_id: str, note: str = None) -> dict:
        """Approve a human review stage and advance."""
        stage_def = STAGE_MAP.get(stage_key)
        if not stage_def or not stage_def.approval_required:
            return {"error": f"Stage '{stage_key}' does not require approval"}

        self.state.update_node(stage_key, "completed", output_data={
            "approved": True,
            "approved_by": actor_id,
            "note": note,
            "approved_at": datetime.utcnow().isoformat(),
        })

        # Record in state
        state_data = WorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("approval_history", []).append({
            "stage": stage_key,
            "approved": True,
            "actor_id": actor_id,
            "note": note,
            "timestamp": datetime.utcnow().isoformat(),
        })
        WorkflowStateStore.save(self.workflow_id, state_data)

        return await self._transition_next(stage_key, actor_id, {"approved": True})

    async def reject(self, stage_key: str, actor_id: str, note: str = None) -> dict:
        """Reject at a human review stage — triggers feedback loop."""
        stage_def = STAGE_MAP.get(stage_key)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_key}"}

        reject_target = get_reject_target(stage_key)
        if not reject_target:
            return {"error": f"Stage '{stage_key}' has no rejection target"}

        self.state.update_node(stage_key, "completed", output_data={
            "approved": False,
            "rejected_by": actor_id,
            "note": note,
            "rejected_at": datetime.utcnow().isoformat(),
        })

        # Record feedback loop
        state_data = WorkflowStateStore.get_state_data(self.workflow_id)
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
        WorkflowStateStore.save(self.workflow_id, state_data)

        # Transition back to reject target
        self.state.transition_to(reject_target, trigger="rejection", actor_id=actor_id,
                                  metadata={"reason": note, "from_stage": stage_key})

        # Reset the target stage node to pending
        self.state.update_node(reject_target, "pending")

        return {
            "status": "rejected",
            "from_stage": stage_key,
            "to_stage": reject_target,
            "to_label": STAGE_MAP[reject_target].label,
            "note": note,
        }

    # --- Submit Human Input (for non-approval human stages) ---

    async def submit_human_input(self, stage_key: str, actor_id: str, input_data: dict) -> dict:
        """Submit input for a human stage (e.g., strategy brief, FDM edits)."""
        stage_def = STAGE_MAP.get(stage_key)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_key}"}

        self.state.update_node(stage_key, "completed", output_data=input_data)

        state_data = WorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("stage_outputs", {})[stage_key] = input_data
        WorkflowStateStore.save(self.workflow_id, state_data)

        if stage_def.approval_required:
            return {"status": "input_received", "stage": stage_key, "awaiting_approval": True}

        return await self._transition_next(stage_key, actor_id, input_data)

    # --- Chat Message Handling ---

    async def handle_chat_message(self, user_id: str, message: str, role: str = "user") -> dict:
        """
        Handle a chat message from the UI.
        Routes to the appropriate handler based on the current stage and user role.
        """
        session = UserSession.get_or_create(self.workflow_id, user_id)

        # Append message to session context
        chat_context = session.get("chat_context", [])
        chat_context.append({
            "role": role,
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
        })
        UserSession.update_chat_context(self.workflow_id, user_id, chat_context)

        # Store in timeline
        db.content_timeline_entries.insert_one({
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

    # --- Context Building ---

    def _build_agent_context(self) -> dict:
        """Build the full context dict passed to agents."""
        workflow = db.content_workflows.find_one({"_id": ObjectId(self.workflow_id)})
        brand = db.brands.find_one({"_id": ObjectId(workflow["brand_id"])}) if workflow else None
        audiences = list(db.audiences.find({"brand_id": workflow["brand_id"]})) if workflow else []

        # Get all stage outputs
        state_data = WorkflowStateStore.get_state_data(self.workflow_id)

        return {
            "workflow_id": self.workflow_id,
            "brand": _safe_serialize(brand) if brand else {},
            "audiences": [_safe_serialize(a) for a in audiences],
            "stage_outputs": state_data.get("stage_outputs", {}),
            "approval_history": state_data.get("approval_history", []),
            "feedback_loops": state_data.get("feedback_loops", []),
            "config": workflow.get("config", {}) if workflow else {},
        }

    def _get_stage_input(self, stage_key: str) -> dict:
        """Get the input data for a stage (typically the prior stage's output)."""
        idx = get_stage_index(stage_key)
        if idx == 0:
            return {}
        prev_key = STAGE_KEYS[idx - 1]
        state_data = WorkflowStateStore.get_state_data(self.workflow_id)
        return state_data.get("stage_outputs", {}).get(prev_key, {})

    # --- Run Full Pipeline (auto-advance through agent stages) ---

    async def run_pipeline(self, actor_id: str = None) -> dict:
        """
        Run the pipeline from the current stage, auto-advancing through
        agent/auto stages and stopping at human stages.
        """
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
    """Convert MongoDB workflow doc to response dict."""
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


def _safe_serialize(doc) -> dict:
    """Convert a MongoDB doc to a JSON-safe dict."""
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
