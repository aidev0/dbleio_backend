"""
State Management for Development Workflows

Persists workflow state, agent states, and user sessions to MongoDB.
Mirrors content_generation/state.py but uses dev-specific collections.
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
db.development_workflows.create_index("organization_id")
db.development_workflow_agents.create_index([("workflow_id", 1), ("stage_name", 1)])
db.development_workflow_transitions.create_index("workflow_id")
db.development_workflow_states.create_index("workflow_id", unique=True)
db.development_agent_states.create_index([("workflow_id", 1), ("agent_name", 1)])
db.development_user_sessions.create_index([("workflow_id", 1), ("user_id", 1)])


class DevWorkflowState:
    """Manages persistent state for a single development workflow."""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self._doc = None

    def _load(self) -> dict:
        if self._doc is None:
            self._doc = db.development_workflows.find_one({"_id": ObjectId(self.workflow_id)})
        return self._doc

    def reload(self):
        self._doc = None
        return self._load()

    @property
    def current_stage(self) -> str:
        doc = self._load()
        return doc.get("current_stage", "spec_intake") if doc else "spec_intake"

    @property
    def status(self) -> str:
        doc = self._load()
        return doc.get("status", "pending") if doc else "pending"

    def get_stage_output(self, stage_name: str) -> Optional[dict]:
        node = db.development_workflow_agents.find_one({
            "workflow_id": self.workflow_id,
            "stage_name": stage_name,
        })
        return node.get("output_data") if node else None

    def get_all_stage_outputs(self) -> Dict[str, Any]:
        agents = db.development_workflow_agents.find({"workflow_id": self.workflow_id})
        return {a["stage_name"]: a.get("output_data", {}) for a in agents}

    def transition_to(self, next_stage: str, trigger: str = "auto", actor_id: str = None, metadata: dict = None):
        now = datetime.utcnow()
        prev_stage = self.current_stage

        db.development_workflow_transitions.insert_one({
            "workflow_id": self.workflow_id,
            "from_stage": prev_stage,
            "to_stage": next_stage,
            "trigger": trigger,
            "actor_id": actor_id,
            "metadata": metadata or {},
            "timestamp": now,
        })

        from development.workflows.pipeline import get_dev_stage_index
        db.development_workflows.update_one(
            {"_id": ObjectId(self.workflow_id)},
            {"$set": {
                "current_stage": next_stage,
                "current_stage_index": get_dev_stage_index(next_stage),
                "updated_at": now,
            }}
        )
        self._doc = None

    def set_status(self, status: str, actor_id: str = None):
        now = datetime.utcnow()
        db.development_workflows.update_one(
            {"_id": ObjectId(self.workflow_id)},
            {"$set": {"status": status, "updated_at": now}}
        )
        self._doc = None

    def update_node(self, stage_name: str, status: str, output_data: dict = None, error: str = None):
        now = datetime.utcnow()
        update_fields = {"status": status, "updated_at": now}
        if output_data is not None:
            update_fields["output_data"] = output_data
        if error is not None:
            update_fields["error"] = error
        if status == "running":
            update_fields["started_at"] = now
        if status in ("completed", "failed"):
            update_fields["completed_at"] = now

        from development.workflows.pipeline import get_dev_stage_index, DEV_STAGE_MAP
        stage_def = DEV_STAGE_MAP.get(stage_name)

        db.development_workflow_agents.update_one(
            {"workflow_id": self.workflow_id, "stage_name": stage_name},
            {"$set": update_fields,
             "$setOnInsert": {
                 "workflow_id": self.workflow_id,
                 "stage_name": stage_name,
                 "stage_index": get_dev_stage_index(stage_name),
                 "node_type": stage_def.node_type if stage_def else "agent",
                 "created_at": now,
             }},
            upsert=True,
        )

    def get_transition_history(self) -> List[dict]:
        transitions = list(db.development_workflow_transitions.find(
            {"workflow_id": self.workflow_id}
        ).sort("timestamp", 1))
        for t in transitions:
            t["_id"] = str(t["_id"])
        return transitions


class DevWorkflowStateStore:
    @staticmethod
    def save(workflow_id: str, state_data: dict):
        now = datetime.utcnow()
        db.development_workflow_states.update_one(
            {"workflow_id": workflow_id},
            {"$set": {"workflow_id": workflow_id, "state": state_data, "updated_at": now},
             "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    @staticmethod
    def load(workflow_id: str) -> Optional[dict]:
        doc = db.development_workflow_states.find_one({"workflow_id": workflow_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def get_state_data(workflow_id: str) -> dict:
        doc = db.development_workflow_states.find_one({"workflow_id": workflow_id})
        return doc.get("state", {}) if doc else {}


class DevAgentStateStore:
    @staticmethod
    def save(workflow_id: str, agent_name: str, state_data: dict):
        now = datetime.utcnow()
        db.development_agent_states.update_one(
            {"workflow_id": workflow_id, "agent_name": agent_name},
            {"$set": {"workflow_id": workflow_id, "agent_name": agent_name, "state": state_data, "updated_at": now},
             "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    @staticmethod
    def load(workflow_id: str, agent_name: str) -> Optional[dict]:
        doc = db.development_agent_states.find_one({"workflow_id": workflow_id, "agent_name": agent_name})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    @staticmethod
    def get_state_data(workflow_id: str, agent_name: str) -> dict:
        doc = db.development_agent_states.find_one({"workflow_id": workflow_id, "agent_name": agent_name})
        return doc.get("state", {}) if doc else {}

    @staticmethod
    def list_agents(workflow_id: str) -> List[dict]:
        agents = list(db.development_agent_states.find({"workflow_id": workflow_id}))
        for a in agents:
            a["_id"] = str(a["_id"])
        return agents

    @staticmethod
    def clear(workflow_id: str, agent_name: str):
        db.development_agent_states.delete_one({"workflow_id": workflow_id, "agent_name": agent_name})


class DevUserSession:
    @staticmethod
    def get_or_create(workflow_id: str, user_id: str) -> dict:
        now = datetime.utcnow()
        session = db.development_user_sessions.find_one({"workflow_id": workflow_id, "user_id": user_id})
        if not session:
            doc = {
                "workflow_id": workflow_id,
                "user_id": user_id,
                "chat_context": [],
                "preferences": {},
                "last_viewed_stage": None,
                "last_active_at": now,
                "created_at": now,
                "updated_at": now,
            }
            result = db.development_user_sessions.insert_one(doc)
            session = db.development_user_sessions.find_one({"_id": result.inserted_id})
        session["_id"] = str(session["_id"])
        return session

    @staticmethod
    def update_chat_context(workflow_id: str, user_id: str, messages: List[dict]):
        MAX_CONTEXT = 50
        now = datetime.utcnow()
        db.development_user_sessions.update_one(
            {"workflow_id": workflow_id, "user_id": user_id},
            {"$set": {"chat_context": messages[-MAX_CONTEXT:], "last_active_at": now, "updated_at": now}}
        )

    @staticmethod
    def get_active_sessions(workflow_id: str) -> List[dict]:
        sessions = list(db.development_user_sessions.find({"workflow_id": workflow_id}))
        for s in sessions:
            s["_id"] = str(s["_id"])
        return sessions
