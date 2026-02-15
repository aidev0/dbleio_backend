"""
Development Orchestrator

The central coordinator that bridges:
1. UI Chat (user/client messages)
2. FDM/FDE Review (team member actions)
3. Workflow Executor (runs agents as tools, manages state machine)

Adds model selection per stage — each stage can use a different LLM/CLI tool.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from bson import ObjectId
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

from src.development.state import DevWorkflowState, DevWorkflowStateStore, DevAgentStateStore, DevUserSession
from src.development.workflows.pipeline import (
    DEV_PIPELINE_STAGES, DEV_STAGE_MAP, DEV_STAGE_KEYS,
    get_dev_next_stage, get_dev_reject_target, get_dev_stage_index,
)
from src.development.agents import get_dev_agent, DEV_AGENT_REGISTRY
from src.development.models import get_model, get_models_by_category, ALL_MODELS
from src.development.events import log_event


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
            model_overrides: Optional dict mapping stage_name → model_id to override defaults.
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
        result = db.development_workflows.insert_one(doc)
        workflow_id = str(result.inserted_id)

        # Initialize all agents as pending
        agents = []
        for stage in DEV_PIPELINE_STAGES:
            # Determine model for this stage
            stage_model = None
            if model_overrides and stage.name in model_overrides:
                stage_model = model_overrides[stage.name]
            elif stage.default_model_id:
                stage_model = stage.default_model_id

            agent = {
                "workflow_id": workflow_id,
                "stage_name": stage.name,
                "stage_index": get_dev_stage_index(stage.name),
                "node_type": stage.node_type,
                "agent_type": stage.agent_name,
                "status": "pending",
                "model_id": stage_model,
                "allowed_model_categories": stage.allowed_model_categories,
                "input_data": {},
                "output_data": {},
                "created_at": now,
                "updated_at": now,
            }
            agents.append(agent)
        if agents:
            db.development_workflow_agents.insert_many(agents)

        # Initialize workflow state snapshot
        DevWorkflowStateStore.save(workflow_id, {
            "current_stage": DEV_STAGE_KEYS[0],
            "status": "pending",
            "stage_outputs": {},
            "approval_history": [],
            "feedback_loops": [],
            "model_overrides": model_overrides or {},
        })

        workflow = db.development_workflows.find_one({"_id": result.inserted_id})
        return _workflow_helper(workflow)

    # --- Model Management ---

    def get_stage_model(self, stage_name: str) -> Optional[str]:
        """Get the configured model_id for a specific stage's agent."""
        agent = db.development_workflow_agents.find_one({
            "workflow_id": self.workflow_id,
            "stage_name": stage_name,
        })
        if agent and agent.get("model_id"):
            return agent["model_id"]
        stage_def = DEV_STAGE_MAP.get(stage_name)
        return stage_def.default_model_id if stage_def else None

    def set_stage_model(self, stage_name: str, model_id: str) -> dict:
        """Update the model for a specific stage's agent (before it runs)."""
        model = get_model(model_id)
        if not model:
            return {"error": f"Unknown model: {model_id}"}

        stage_def = DEV_STAGE_MAP.get(stage_name)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_name}"}

        if model.category not in stage_def.allowed_model_categories:
            return {"error": f"Model category '{model.category}' not allowed for stage '{stage_name}'. Allowed: {stage_def.allowed_model_categories}"}

        db.development_workflow_agents.update_one(
            {"workflow_id": self.workflow_id, "stage_name": stage_name},
            {"$set": {"model_id": model_id, "updated_at": datetime.utcnow()}}
        )

        # Also update in state snapshot
        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("model_overrides", {})[stage_name] = model_id
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        return {"status": "updated", "stage": stage_name, "model_id": model_id}

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
        if stage_def.node_type == "human":
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

    async def _execute_agent_stage(self, stage_name: str, stage_def, actor_id: str = None) -> dict:
        """Execute an agent for the current stage with the configured model."""
        self.state.update_node(stage_name, "running")
        self.state.set_status("running", actor_id)

        try:
            # Get the model override for this stage
            model_id = self.get_stage_model(stage_name)
            agent = get_dev_agent(stage_def.agent_name, self.workflow_id, model_id=model_id)

            context = self._build_agent_context()
            input_data = self._get_stage_input(stage_name)

            output = await agent.run(input_data, context)

            self.state.update_node(stage_name, "completed", output_data=output)

            # Save to workflow state snapshot
            state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
            state_data.setdefault("stage_outputs", {})[stage_name] = output
            DevWorkflowStateStore.save(self.workflow_id, state_data)

            return await self._transition_next(stage_name, actor_id, output)

        except Exception as e:
            self.state.update_node(stage_name, "failed", error=str(e))
            self.state.set_status("failed", actor_id)
            return {"error": str(e), "stage": stage_name}

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

    async def approve(self, stage_name: str, actor_id: str, note: str = None) -> dict:
        """Approve a human review stage and advance."""
        stage_def = DEV_STAGE_MAP.get(stage_name)
        if not stage_def or not stage_def.approval_required:
            return {"error": f"Stage '{stage_name}' does not require approval"}

        self.state.update_node(stage_name, "completed", output_data={
            "approved": True,
            "approved_by": actor_id,
            "note": note,
            "approved_at": datetime.utcnow().isoformat(),
        })

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("approval_history", []).append({
            "stage": stage_name,
            "approved": True,
            "actor_id": actor_id,
            "note": note,
            "timestamp": datetime.utcnow().isoformat(),
        })
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        return await self._transition_next(stage_name, actor_id, {"approved": True})

    async def reject(self, stage_name: str, actor_id: str, note: str = None) -> dict:
        """Reject at a human review stage — triggers feedback loop."""
        stage_def = DEV_STAGE_MAP.get(stage_name)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_name}"}

        reject_target = get_dev_reject_target(stage_name)
        if not reject_target:
            return {"error": f"Stage '{stage_name}' has no rejection target"}

        self.state.update_node(stage_name, "completed", output_data={
            "approved": False,
            "rejected_by": actor_id,
            "note": note,
            "rejected_at": datetime.utcnow().isoformat(),
        })

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("approval_history", []).append({
            "stage": stage_name,
            "approved": False,
            "actor_id": actor_id,
            "note": note,
            "timestamp": datetime.utcnow().isoformat(),
        })
        state_data.setdefault("feedback_loops", []).append({
            "from_stage": stage_name,
            "to_stage": reject_target,
            "reason": note or "rejected",
            "timestamp": datetime.utcnow().isoformat(),
        })
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        self.state.transition_to(reject_target, trigger="rejection", actor_id=actor_id,
                                  metadata={"reason": note, "from_stage": stage_name})
        self.state.update_node(reject_target, "pending")

        return {
            "status": "rejected",
            "from_stage": stage_name,
            "to_stage": reject_target,
            "to_label": DEV_STAGE_MAP[reject_target].label,
            "note": note,
        }

    async def submit_human_input(self, stage_name: str, actor_id: str, input_data: dict) -> dict:
        """Submit input for a human stage (e.g., spec, FDM edits)."""
        stage_def = DEV_STAGE_MAP.get(stage_name)
        if not stage_def:
            return {"error": f"Unknown stage: {stage_name}"}

        self.state.update_node(stage_name, "completed", output_data=input_data)

        state_data = DevWorkflowStateStore.get_state_data(self.workflow_id)
        state_data.setdefault("stage_outputs", {})[stage_name] = input_data
        DevWorkflowStateStore.save(self.workflow_id, state_data)

        if stage_def.approval_required:
            return {"status": "input_received", "stage": stage_name, "awaiting_approval": True}

        return await self._transition_next(stage_name, actor_id, input_data)

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

        db.timeline_entries.insert_one({
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
        workflow = db.development_workflows.find_one({"_id": ObjectId(self.workflow_id)})
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

    def _get_stage_input(self, stage_name: str) -> dict:
        idx = get_dev_stage_index(stage_name)
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


# ===========================================================================
# Legacy standalone functions (used by old router endpoints)
# ===========================================================================

# 13-stage pipeline definition (flat dict format for legacy endpoints)
PIPELINE_STAGES = [
    {"name": "spec_intake",    "node_type": "human",  "agent_type": "notify",             "job_type": "notify",       "gate": None},
    {"name": "setup",          "node_type": "human",  "agent_type": "setup",              "job_type": "setup",        "gate": None},
    {"name": "planner",        "node_type": "agent",  "agent_type": "planner_agent",      "job_type": "plan",         "gate": None},
    {"name": "plan_reviewer",  "node_type": "agent",  "agent_type": "plan_reviewer_agent","job_type": "review",       "gate": None},
    {"name": "plan_approval",  "node_type": "human",  "agent_type": None,                 "job_type": None,           "gate": "fde_plan"},
    {"name": "developer",      "node_type": "agent",  "agent_type": "developer_agent",    "job_type": "implement",    "gate": None},
    {"name": "code_reviewer",  "node_type": "agent",  "agent_type": "code_reviewer_agent","job_type": "review",       "gate": None},
    {"name": "validator",      "node_type": "agent",  "agent_type": "validator_agent",    "job_type": "validate",     "gate": None},
    {"name": "commit_pr",      "node_type": "human",  "agent_type": "commit_pr_agent",   "job_type": "commit_pr",    "gate": None},
    {"name": "deployer",       "node_type": "human",  "agent_type": "deployer_agent",     "job_type": "deploy",       "gate": None},
    {"name": "qa_review",      "node_type": "human",  "agent_type": None,                 "job_type": None,           "gate": "qa"},
    {"name": "client_review",  "node_type": "human",  "agent_type": None,                 "job_type": None,           "gate": "client"},
    {"name": "done",           "node_type": "human",  "agent_type": None,                 "job_type": None,           "gate": None},
]

MAX_REVIEW_ITERATIONS = 3

STAGE_LABELS = {
    "spec_intake": "Spec Intake",
    "setup": "Setup",
    "planner": "Planner",
    "plan_reviewer": "Plan Review",
    "plan_approval": "Plan Approval",
    "developer": "Developer",
    "code_reviewer": "Code Review",
    "validator": "Validator",
    "commit_pr": "Commit & PR",
    "deployer": "Deploy",
    "qa_review": "QA Review",
    "client_review": "Client Review",
    "done": "Approved & Completed",
}


def ensure_indexes(db_ref):
    db_ref.development_workflows.create_index("organization_id")
    db_ref.development_workflows.create_index("project_id")
    db_ref.development_workflows.create_index("status")
    db_ref.development_workflow_agents.create_index("workflow_id")
    db_ref.development_workflow_jobs.create_index([("status", ASCENDING), ("run_after", ASCENDING)])
    db_ref.development_workflow_approvals.create_index([("workflow_id", ASCENDING), ("approval_type", ASCENDING)])


def create_workflow(
    db_ref,
    organization_id: str,
    title: str,
    specification_id: str = "",
    project_id: Optional[str] = None,
    description: str = "",
    agent_config: Optional[Dict] = None,
    created_by: str = "",
) -> dict:
    """
    Create a new workflow with 13 nodes (all pending) and queue the first job.
    Returns the workflow document.
    """
    now = datetime.utcnow()

    workflow_doc = {
        "organization_id": organization_id,
        "project_id": project_id,
        "specification_id": specification_id,
        "title": title,
        "description": description,
        "status": "pending",
        "current_stage": "spec_intake",
        "current_stage_index": 0,
        "agent_config": agent_config or {},
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
    }
    result = db_ref.development_workflows.insert_one(workflow_doc)
    workflow_id = str(result.inserted_id)

    # Create 13 agents
    for idx, stage in enumerate(PIPELINE_STAGES):
        agent_doc = {
            "workflow_id": workflow_id,
            "stage_name": stage["name"],
            "stage_index": idx,
            "node_type": stage["node_type"],
            "agent_type": stage["agent_type"],
            "model_id": None,
            "allowed_model_categories": [],
            "status": "pending",
            "iteration": 0,
            "input_data": {},
            "output_data": {},
            "error": None,
            "started_at": None,
            "completed_at": None,
            "created_at": now,
            "updated_at": now,
        }
        db_ref.development_workflow_agents.insert_one(agent_doc)

    # Queue the first job (spec_intake is a notify job)
    create_job(db_ref, workflow_id, "notify", stage_name="spec_intake")

    log_event(db_ref, workflow_id, "workflow_created", "system", created_by,
              f"Workflow '{title}' created")

    # Push timeline update
    push_timeline_update(db_ref, workflow_id, "spec_intake", "running",
                         f"Workflow '{title}' started", visibility="public")

    workflow_doc["_id"] = workflow_id
    return workflow_doc


def transition_workflow(db_ref, workflow_id: str, new_status: str, new_stage: str, new_stage_index: int):
    """Atomically update workflow status and current stage."""
    now = datetime.utcnow()
    db_ref.development_workflows.update_one(
        {"_id": ObjectId(workflow_id)},
        {"$set": {
            "status": new_status,
            "current_stage": new_stage,
            "current_stage_index": new_stage_index,
            "updated_at": now,
        }}
    )
    log_event(db_ref, workflow_id, "stage_transition", "system", "orchestrator",
              f"Transitioned to {new_stage} (status={new_status})")

    # Push timeline update
    label = STAGE_LABELS.get(new_stage, new_stage)
    push_timeline_update(db_ref, workflow_id, new_stage, new_status,
                         f"{label}: {new_status}")


def create_job(
    db_ref,
    workflow_id: str,
    job_type: str,
    stage_name: str = "",
    input_data: Optional[Dict] = None,
    delay_seconds: int = 0,
) -> str:
    """Create a queued job for the worker to pick up. Returns job ID."""
    now = datetime.utcnow()
    job_doc = {
        "workflow_id": workflow_id,
        "job_type": job_type,
        "stage_name": stage_name,
        "status": "queued",
        "input_data": input_data or {},
        "output_data": {},
        "attempt": 0,
        "max_attempts": 3,
        "run_after": now + timedelta(seconds=delay_seconds),
        "claimed_by": None,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    result = db_ref.development_workflow_jobs.insert_one(job_doc)
    return str(result.inserted_id)


def handle_stage_completion(db_ref, workflow_id: str, completed_stage_name: str, output_data: Dict = None):
    """
    Called when a stage finishes. Determines the next action:
    advance to next job, wait for human gate, or mark workflow complete.
    """
    workflow = db_ref.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        return

    # Find current stage index
    current_idx = None
    for idx, stage in enumerate(PIPELINE_STAGES):
        if stage["name"] == completed_stage_name:
            current_idx = idx
            break
    if current_idx is None:
        return

    # Update the node
    db_ref.development_workflow_agents.update_one(
        {"workflow_id": workflow_id, "stage_name": completed_stage_name},
        {"$set": {
            "status": "completed",
            "output_data": output_data or {},
            "completed_at": datetime.utcnow(),
        }}
    )

    next_idx = current_idx + 1

    # Check if workflow is done
    if next_idx >= len(PIPELINE_STAGES):
        transition_workflow(db_ref, workflow_id, "completed", completed_stage_name, current_idx)
        log_event(db_ref, workflow_id, "workflow_completed", "system", "orchestrator",
                  "Workflow completed successfully")
        push_timeline_update(db_ref, workflow_id, completed_stage_name, "completed",
                             "Workflow completed successfully!")
        return

    next_stage = PIPELINE_STAGES[next_idx]
    now = datetime.utcnow()

    # Terminal "done" stage — mark completed immediately
    if next_stage["name"] == "done":
        db_ref.development_workflow_agents.update_one(
            {"workflow_id": workflow_id, "stage_name": "done"},
            {"$set": {"status": "completed", "started_at": now, "completed_at": now}}
        )
        transition_workflow(db_ref, workflow_id, "completed", "done", next_idx)
        log_event(db_ref, workflow_id, "workflow_completed", "system", "orchestrator",
                  "Workflow completed successfully")
        push_timeline_update(db_ref, workflow_id, "done", "completed",
                             "Workflow completed successfully!")
        return

    # Update node to running
    db_ref.development_workflow_agents.update_one(
        {"workflow_id": workflow_id, "stage_name": next_stage["name"]},
        {"$set": {"status": "running", "started_at": now}}
    )

    # If next stage has a human gate, transition and wait
    if next_stage["gate"]:
        transition_workflow(db_ref, workflow_id, "waiting_approval", next_stage["name"], next_idx)
        db_ref.development_workflow_agents.update_one(
            {"workflow_id": workflow_id, "stage_name": next_stage["name"]},
            {"$set": {"status": "waiting_approval"}}
        )
        log_event(db_ref, workflow_id, "waiting_approval", "system", "orchestrator",
                  f"Waiting for {next_stage['gate']} approval at {next_stage['name']}")
        return

    # Otherwise queue the next job
    transition_workflow(db_ref, workflow_id, "running", next_stage["name"], next_idx)
    if next_stage["job_type"]:
        create_job(db_ref, workflow_id, next_stage["job_type"], stage_name=next_stage["name"],
                   input_data=output_data)


def handle_approval(db_ref, workflow_id: str, approval_type: str, approved: bool,
                    note: str = "", approved_by: str = ""):
    """
    Process a human gate decision. Advance or loop back.
    approval_type: fde_plan, fde_pr, qa, client
    """
    workflow = db_ref.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if not workflow:
        return

    now = datetime.utcnow()
    approval_doc = {
        "workflow_id": workflow_id,
        "approval_type": approval_type,
        "approved": approved,
        "note": note,
        "approved_by": approved_by,
        "created_at": now,
    }
    db_ref.development_workflow_approvals.insert_one(approval_doc)

    current_stage = workflow["current_stage"]
    current_idx = workflow.get("current_stage_index", 0)

    if approved:
        log_event(db_ref, workflow_id, "approval_granted", "user", approved_by,
                  f"{approval_type} approved at {current_stage}")
        handle_stage_completion(db_ref, workflow_id, current_stage, {"approval": approval_type, "note": note})
    else:
        # Loop back based on approval type
        loop_back_stage = _get_loop_back_stage(approval_type)
        log_event(db_ref, workflow_id, "approval_rejected", "user", approved_by,
                  f"{approval_type} rejected at {current_stage}, looping back to {loop_back_stage}")

        # Find loop back index
        for idx, stage in enumerate(PIPELINE_STAGES):
            if stage["name"] == loop_back_stage:
                # Increment iteration on loop-back node
                db_ref.development_workflow_agents.update_one(
                    {"workflow_id": workflow_id, "stage_name": loop_back_stage},
                    {"$inc": {"iteration": 1}, "$set": {"status": "running", "started_at": now}}
                )
                transition_workflow(db_ref, workflow_id, "running", loop_back_stage, idx)
                stage_def = PIPELINE_STAGES[idx]
                if stage_def["job_type"]:
                    create_job(db_ref, workflow_id, stage_def["job_type"], stage_name=loop_back_stage,
                               input_data={"feedback": note, "loop_back_from": current_stage})
                break


def handle_review_feedback(db_ref, workflow_id: str, reviewer_stage: str, passed: bool,
                           feedback: str = "", output_data: Dict = None):
    """
    Process agent review result. If passed, advance. If failed, loop back
    (up to MAX_REVIEW_ITERATIONS).
    """
    node = db_ref.development_workflow_agents.find_one({
        "workflow_id": workflow_id,
        "stage_name": reviewer_stage,
    })
    if not node:
        return

    iteration = node.get("iteration", 0)

    if passed or iteration >= MAX_REVIEW_ITERATIONS:
        if not passed:
            log_event(db_ref, workflow_id, "max_iterations", "system", "orchestrator",
                      f"Max iterations ({MAX_REVIEW_ITERATIONS}) reached at {reviewer_stage}, advancing anyway")
        handle_stage_completion(db_ref, workflow_id, reviewer_stage, output_data)
    else:
        # Loop back
        loop_back_stage = _get_review_loop_back(reviewer_stage)
        log_event(db_ref, workflow_id, "review_rejected", "agent", reviewer_stage,
                  f"Review failed at {reviewer_stage}, looping back to {loop_back_stage}")

        now = datetime.utcnow()
        for idx, stage in enumerate(PIPELINE_STAGES):
            if stage["name"] == loop_back_stage:
                db_ref.development_workflow_agents.update_one(
                    {"workflow_id": workflow_id, "stage_name": loop_back_stage},
                    {"$inc": {"iteration": 1}, "$set": {"status": "running", "started_at": now}}
                )
                transition_workflow(db_ref, workflow_id, "running", loop_back_stage, idx)
                stage_def = PIPELINE_STAGES[idx]
                if stage_def["job_type"]:
                    create_job(db_ref, workflow_id, stage_def["job_type"], stage_name=loop_back_stage,
                               input_data={"feedback": feedback, "loop_back_from": reviewer_stage})
                break


def _get_loop_back_stage(approval_type: str) -> str:
    """Map approval types to their loop-back target."""
    return {
        "fde_plan": "planner",
        "fde_pr": "developer",
        "qa": "developer",
        "client": "developer",
    }.get(approval_type, "planner")


def _get_review_loop_back(reviewer_stage: str) -> str:
    """Map reviewer stages to their loop-back target."""
    return {
        "plan_reviewer": "planner",
        "code_reviewer": "developer",
    }.get(reviewer_stage, "planner")


# --- Timeline Bridge ---

def push_timeline_update(db_ref, workflow_id: str, stage: str, status: str,
                         message: str, visibility: str = "public"):
    """Create a status_update timeline entry from orchestrator events."""
    now = datetime.utcnow()
    label = STAGE_LABELS.get(stage, stage)

    # Look up actual workflow creator
    author_id = "system"
    author_name = "dble"
    wf = db_ref.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if wf and wf.get("created_by"):
        user = db_ref.users.find_one({"workos_user_id": wf["created_by"]})
        if user:
            first = user.get("first_name", "")
            last = user.get("last_name", "")
            full = f"{first} {last}".strip()
            author_name = full or user.get("email", "dble")
            author_id = wf["created_by"]

    entry = {
        "workflow_id": workflow_id,
        "card_type": "status_update",
        "content": message,
        "author_id": author_id,
        "author_name": author_name,
        "author_role": "system",
        "visibility": visibility,
        "todos": [],
        "approval_data": None,
        "status_data": {"stage": stage, "status": status, "message": message},
        "parent_entry_id": None,
        "ai_model": None,
        "edited_by": None,
        "is_deleted": False,
        "created_at": now,
        "updated_at": now,
    }
    db_ref.timeline_entries.insert_one(entry)


# --- Query helpers ---

def get_workflow(db_ref, workflow_id: str) -> Optional[dict]:
    wf = db_ref.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if wf:
        wf["_id"] = str(wf["_id"])
    return wf


def get_workflow_agents(db_ref, workflow_id: str) -> List[dict]:
    agents = list(db_ref.development_workflow_agents.find({"workflow_id": workflow_id}).sort("stage_index", 1))
    for a in agents:
        a["_id"] = str(a["_id"])
    return agents


# Backward-compatible alias
get_workflow_nodes = get_workflow_agents


def get_workflow_jobs(db_ref, workflow_id: str) -> List[dict]:
    jobs = list(db_ref.development_workflow_jobs.find({"workflow_id": workflow_id}).sort("created_at", -1))
    for j in jobs:
        j["_id"] = str(j["_id"])
    return jobs
