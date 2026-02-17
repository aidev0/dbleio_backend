"""
Robust State Management for Content Generation Workflows

Persists all workflow state and user sessions to MongoDB.
Supports:
- Workflow execution state (current stage, history, outputs per stage)
- User session state (who is interacting, chat context, preferences)
- Stage transition history (audit trail)
- Rollback and replay capabilities
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

# Ensure indexes
db.content_workflows.create_index("brand_id")
db.content_workflow_nodes.create_index([("workflow_id", 1), ("stage_key", 1)])
db.content_workflow_transitions.create_index("workflow_id")
db.content_workflow_states.create_index("workflow_id", unique=True)
db.content_agent_states.create_index([("workflow_id", 1), ("agent_name", 1)])
db.content_user_sessions.create_index([("workflow_id", 1), ("user_id", 1)])
db.content_user_sessions.create_index("user_id")


# --- Workflow State ---

class WorkflowState:
    """Manages persistent state for a single content workflow."""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self._doc = None

    def _load(self) -> dict:
        if self._doc is None:
            self._doc = db.content_workflows.find_one({"_id": ObjectId(self.workflow_id)})
        return self._doc

    def reload(self):
        self._doc = None
        return self._load()

    @property
    def current_stage(self) -> str:
        doc = self._load()
        return doc.get("current_stage", "strategy_assets") if doc else "strategy_assets"

    @property
    def status(self) -> str:
        doc = self._load()
        return doc.get("status", "pending") if doc else "pending"

    def get_stage_output(self, stage_key: str) -> Optional[dict]:
        """Get the output data for a specific stage."""
        node = db.content_workflow_nodes.find_one({
            "workflow_id": self.workflow_id,
            "stage_key": stage_key,
        })
        return node.get("output_data") if node else None

    def get_all_stage_outputs(self) -> Dict[str, Any]:
        """Get all stage outputs as a dict keyed by stage_key."""
        nodes = db.content_workflow_nodes.find({"workflow_id": self.workflow_id})
        return {n["stage_key"]: n.get("output_data", {}) for n in nodes}

    def transition_to(self, next_stage: str, trigger: str = "auto", actor_id: str = None, metadata: dict = None):
        """
        Transition the workflow to a new stage.
        Records the transition in the audit trail.
        """
        now = datetime.utcnow()
        prev_stage = self.current_stage

        # Record transition
        db.content_workflow_transitions.insert_one({
            "workflow_id": self.workflow_id,
            "from_stage": prev_stage,
            "to_stage": next_stage,
            "trigger": trigger,  # "auto" | "approval" | "rejection" | "feedback_loop" | "manual"
            "actor_id": actor_id,
            "metadata": metadata or {},
            "timestamp": now,
        })

        # Update workflow
        from src.content_generation.workflows.pipeline import get_stage_index
        db.content_workflows.update_one(
            {"_id": ObjectId(self.workflow_id)},
            {"$set": {
                "current_stage": next_stage,
                "current_stage_index": get_stage_index(next_stage),
                "updated_at": now,
            }}
        )
        self._doc = None  # Invalidate cache

    def set_status(self, status: str, actor_id: str = None):
        """Set workflow status: pending | running | paused | completed | failed | cancelled"""
        now = datetime.utcnow()
        db.content_workflows.update_one(
            {"_id": ObjectId(self.workflow_id)},
            {"$set": {"status": status, "updated_at": now}}
        )
        db.content_workflow_transitions.insert_one({
            "workflow_id": self.workflow_id,
            "from_stage": self.current_stage,
            "to_stage": self.current_stage,
            "trigger": "status_change",
            "actor_id": actor_id,
            "metadata": {"new_status": status},
            "timestamp": now,
        })
        self._doc = None

    def update_node(self, stage_key: str, status: str, output_data: dict = None, error: str = None):
        """Create or update a stage node with its execution result."""
        now = datetime.utcnow()
        update_fields = {
            "status": status,
            "updated_at": now,
        }
        if output_data is not None:
            update_fields["output_data"] = output_data
        if error is not None:
            update_fields["error"] = error
        if status == "running":
            update_fields["started_at"] = now
        if status in ("completed", "failed"):
            update_fields["completed_at"] = now

        from src.content_generation.workflows.pipeline import get_stage_index, STAGE_MAP
        stage_def = STAGE_MAP.get(stage_key)

        db.content_workflow_nodes.update_one(
            {"workflow_id": self.workflow_id, "stage_key": stage_key},
            {"$set": update_fields,
             "$setOnInsert": {
                 "workflow_id": self.workflow_id,
                 "stage_key": stage_key,
                 "stage_index": get_stage_index(stage_key),
                 "stage_type": stage_def.stage_type if stage_def else "agent",
                 "created_at": now,
             }},
            upsert=True,
        )

    def get_transition_history(self) -> List[dict]:
        """Get full audit trail of stage transitions."""
        transitions = list(db.content_workflow_transitions.find(
            {"workflow_id": self.workflow_id}
        ).sort("timestamp", 1))
        for t in transitions:
            t["_id"] = str(t["_id"])
        return transitions


# --- Workflow States (persistent snapshot of full workflow context) ---

class WorkflowStateStore:
    """
    Persists the full runtime state of a workflow to content_workflow_states.
    This is the canonical state the orchestrator uses to resume, replay, or debug.
    """

    @staticmethod
    def save(workflow_id: str, state_data: dict):
        """Save or update the full workflow state snapshot."""
        now = datetime.utcnow()
        db.content_workflow_states.update_one(
            {"workflow_id": workflow_id},
            {"$set": {
                "workflow_id": workflow_id,
                "state": state_data,
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    @staticmethod
    def load(workflow_id: str) -> Optional[dict]:
        """Load the full workflow state snapshot."""
        doc = db.content_workflow_states.find_one({"workflow_id": workflow_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def get_state_data(workflow_id: str) -> dict:
        """Get just the state dict (empty dict if not found)."""
        doc = db.content_workflow_states.find_one({"workflow_id": workflow_id})
        return doc.get("state", {}) if doc else {}


# --- Agent States (per-agent execution state within a workflow) ---

class AgentStateStore:
    """
    Persists individual agent execution state.
    Each agent can store its own context, memory, intermediate results,
    and conversation history so it can resume or be inspected.
    """

    @staticmethod
    def save(workflow_id: str, agent_name: str, state_data: dict):
        """Save or update an agent's state."""
        now = datetime.utcnow()
        db.content_agent_states.update_one(
            {"workflow_id": workflow_id, "agent_name": agent_name},
            {"$set": {
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "state": state_data,
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    @staticmethod
    def load(workflow_id: str, agent_name: str) -> Optional[dict]:
        """Load an agent's state."""
        doc = db.content_agent_states.find_one({
            "workflow_id": workflow_id,
            "agent_name": agent_name,
        })
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def get_state_data(workflow_id: str, agent_name: str) -> dict:
        """Get just the state dict for an agent."""
        doc = db.content_agent_states.find_one({
            "workflow_id": workflow_id,
            "agent_name": agent_name,
        })
        return doc.get("state", {}) if doc else {}

    @staticmethod
    def list_agents(workflow_id: str) -> List[dict]:
        """List all agent states for a workflow."""
        agents = list(db.content_agent_states.find({"workflow_id": workflow_id}))
        for a in agents:
            a["_id"] = str(a["_id"])
        return agents

    @staticmethod
    def clear(workflow_id: str, agent_name: str):
        """Clear an agent's state (for restart)."""
        db.content_agent_states.delete_one({
            "workflow_id": workflow_id,
            "agent_name": agent_name,
        })


# --- User Sessions ---

class UserSession:
    """Manages persistent user session state for a workflow interaction."""

    @staticmethod
    def get_or_create(workflow_id: str, user_id: str) -> dict:
        """Get or create a user session for a workflow."""
        now = datetime.utcnow()
        session = db.content_user_sessions.find_one({
            "workflow_id": workflow_id,
            "user_id": user_id,
        })
        if not session:
            doc = {
                "workflow_id": workflow_id,
                "user_id": user_id,
                "chat_context": [],  # Recent chat messages for context window
                "preferences": {},
                "last_viewed_stage": None,
                "last_active_at": now,
                "created_at": now,
                "updated_at": now,
            }
            result = db.content_user_sessions.insert_one(doc)
            session = db.content_user_sessions.find_one({"_id": result.inserted_id})

        session["_id"] = str(session["_id"])
        return session

    @staticmethod
    def update_chat_context(workflow_id: str, user_id: str, messages: List[dict]):
        """Update the chat context for a user session (keeps last N messages)."""
        MAX_CONTEXT = 50
        now = datetime.utcnow()
        db.content_user_sessions.update_one(
            {"workflow_id": workflow_id, "user_id": user_id},
            {"$set": {
                "chat_context": messages[-MAX_CONTEXT:],
                "last_active_at": now,
                "updated_at": now,
            }}
        )

    @staticmethod
    def update_preferences(workflow_id: str, user_id: str, preferences: dict):
        """Merge new preferences into the session."""
        now = datetime.utcnow()
        set_fields = {f"preferences.{k}": v for k, v in preferences.items()}
        set_fields["updated_at"] = now
        set_fields["last_active_at"] = now
        db.content_user_sessions.update_one(
            {"workflow_id": workflow_id, "user_id": user_id},
            {"$set": set_fields}
        )

    @staticmethod
    def set_last_viewed_stage(workflow_id: str, user_id: str, stage_key: str):
        now = datetime.utcnow()
        db.content_user_sessions.update_one(
            {"workflow_id": workflow_id, "user_id": user_id},
            {"$set": {
                "last_viewed_stage": stage_key,
                "last_active_at": now,
                "updated_at": now,
            }}
        )

    @staticmethod
    def get_active_sessions(workflow_id: str) -> List[dict]:
        """Get all active user sessions for a workflow."""
        sessions = list(db.content_user_sessions.find({"workflow_id": workflow_id}))
        for s in sessions:
            s["_id"] = str(s["_id"])
        return sessions

    @staticmethod
    def get_user_sessions(user_id: str) -> List[dict]:
        """Get all sessions for a user across workflows."""
        sessions = list(db.content_user_sessions.find({"user_id": user_id}).sort("last_active_at", -1))
        for s in sessions:
            s["_id"] = str(s["_id"])
        return sessions
