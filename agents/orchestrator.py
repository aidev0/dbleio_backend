#!/usr/bin/env python3
"""
Development Orchestrator - Manages workflow lifecycle, stage transitions, jobs, and approvals.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from bson import ObjectId
from pymongo import ASCENDING

from agents.events import log_event


# 12-stage pipeline definition
PIPELINE_STAGES = [
    {"name": "spec_intake",    "node_type": "auto",  "job_type": "notify",       "gate": None},
    {"name": "setup",          "node_type": "auto",  "job_type": "setup",        "gate": None},
    {"name": "planner",        "node_type": "agent", "job_type": "plan",         "gate": None},
    {"name": "plan_reviewer",  "node_type": "agent", "job_type": "review",       "gate": None},
    {"name": "plan_approval",  "node_type": "human", "job_type": None,           "gate": "fde_plan"},
    {"name": "developer",      "node_type": "agent", "job_type": "implement",    "gate": None},
    {"name": "code_reviewer",  "node_type": "agent", "job_type": "review",       "gate": None},
    {"name": "validator",      "node_type": "agent", "job_type": "validate",     "gate": None},
    {"name": "commit_pr",      "node_type": "auto",  "job_type": "commit_pr",    "gate": None},
    {"name": "deployer",       "node_type": "auto",  "job_type": "deploy",       "gate": None},
    {"name": "qa_review",      "node_type": "human", "job_type": None,           "gate": "qa"},
    {"name": "client_review",  "node_type": "human", "job_type": None,           "gate": "client"},
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
}


def ensure_indexes(db):
    db.development_workflows.create_index("organization_id")
    db.development_workflows.create_index("project_id")
    db.development_workflows.create_index("status")
    db.development_workflow_nodes.create_index("workflow_id")
    db.development_workflow_jobs.create_index([("status", ASCENDING), ("run_after", ASCENDING)])
    db.development_workflow_approvals.create_index([("workflow_id", ASCENDING), ("approval_type", ASCENDING)])


def create_workflow(
    db,
    organization_id: str,
    title: str,
    specification_id: str = "",
    project_id: Optional[str] = None,
    description: str = "",
    agent_config: Optional[Dict] = None,
    created_by: str = "",
) -> dict:
    """
    Create a new workflow with 12 nodes (all pending) and queue the first job.
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
    result = db.development_workflows.insert_one(workflow_doc)
    workflow_id = str(result.inserted_id)

    # Create 12 nodes
    for idx, stage in enumerate(PIPELINE_STAGES):
        node_doc = {
            "workflow_id": workflow_id,
            "stage_name": stage["name"],
            "stage_index": idx,
            "node_type": stage["node_type"],
            "status": "pending",
            "iteration": 0,
            "input_data": {},
            "output_data": {},
            "started_at": None,
            "completed_at": None,
        }
        db.development_workflow_nodes.insert_one(node_doc)

    # Queue the first job (spec_intake is a notify job)
    create_job(db, workflow_id, "notify", stage_name="spec_intake")

    log_event(db, workflow_id, "workflow_created", "system", created_by,
              f"Workflow '{title}' created")

    # Push timeline update
    push_timeline_update(db, workflow_id, "spec_intake", "running",
                         f"Workflow '{title}' started", visibility="public")

    workflow_doc["_id"] = workflow_id
    return workflow_doc


def transition_workflow(db, workflow_id: str, new_status: str, new_stage: str, new_stage_index: int):
    """Atomically update workflow status and current stage."""
    now = datetime.utcnow()
    db.development_workflows.update_one(
        {"_id": ObjectId(workflow_id)},
        {"$set": {
            "status": new_status,
            "current_stage": new_stage,
            "current_stage_index": new_stage_index,
            "updated_at": now,
        }}
    )
    log_event(db, workflow_id, "stage_transition", "system", "orchestrator",
              f"Transitioned to {new_stage} (status={new_status})")

    # Push timeline update
    label = STAGE_LABELS.get(new_stage, new_stage)
    push_timeline_update(db, workflow_id, new_stage, new_status,
                         f"{label}: {new_status}")


def create_job(
    db,
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
    result = db.development_workflow_jobs.insert_one(job_doc)
    return str(result.inserted_id)


def handle_stage_completion(db, workflow_id: str, completed_stage_name: str, output_data: Dict = None):
    """
    Called when a stage finishes. Determines the next action:
    advance to next job, wait for human gate, or mark workflow complete.
    """
    workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
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
    db.development_workflow_nodes.update_one(
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
        transition_workflow(db, workflow_id, "completed", completed_stage_name, current_idx)
        log_event(db, workflow_id, "workflow_completed", "system", "orchestrator",
                  "Workflow completed successfully")
        push_timeline_update(db, workflow_id, completed_stage_name, "completed",
                             "Workflow completed successfully!")
        return

    next_stage = PIPELINE_STAGES[next_idx]

    # Update node to running
    db.development_workflow_nodes.update_one(
        {"workflow_id": workflow_id, "stage_name": next_stage["name"]},
        {"$set": {"status": "running", "started_at": datetime.utcnow()}}
    )

    # If next stage has a human gate, transition and wait
    if next_stage["gate"]:
        transition_workflow(db, workflow_id, "waiting_approval", next_stage["name"], next_idx)
        db.development_workflow_nodes.update_one(
            {"workflow_id": workflow_id, "stage_name": next_stage["name"]},
            {"$set": {"status": "waiting_approval"}}
        )
        log_event(db, workflow_id, "waiting_approval", "system", "orchestrator",
                  f"Waiting for {next_stage['gate']} approval at {next_stage['name']}")
        return

    # Otherwise queue the next job
    transition_workflow(db, workflow_id, "running", next_stage["name"], next_idx)
    if next_stage["job_type"]:
        create_job(db, workflow_id, next_stage["job_type"], stage_name=next_stage["name"],
                   input_data=output_data)


def handle_approval(db, workflow_id: str, approval_type: str, approved: bool,
                    note: str = "", approved_by: str = ""):
    """
    Process a human gate decision. Advance or loop back.
    approval_type: fde_plan, fde_pr, qa, client
    """
    workflow = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
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
    db.development_workflow_approvals.insert_one(approval_doc)

    current_stage = workflow["current_stage"]
    current_idx = workflow.get("current_stage_index", 0)

    if approved:
        log_event(db, workflow_id, "approval_granted", "user", approved_by,
                  f"{approval_type} approved at {current_stage}")
        handle_stage_completion(db, workflow_id, current_stage, {"approval": approval_type, "note": note})
    else:
        # Loop back based on approval type
        loop_back_stage = _get_loop_back_stage(approval_type)
        log_event(db, workflow_id, "approval_rejected", "user", approved_by,
                  f"{approval_type} rejected at {current_stage}, looping back to {loop_back_stage}")

        # Find loop back index
        for idx, stage in enumerate(PIPELINE_STAGES):
            if stage["name"] == loop_back_stage:
                # Increment iteration on loop-back node
                db.development_workflow_nodes.update_one(
                    {"workflow_id": workflow_id, "stage_name": loop_back_stage},
                    {"$inc": {"iteration": 1}, "$set": {"status": "running", "started_at": now}}
                )
                transition_workflow(db, workflow_id, "running", loop_back_stage, idx)
                stage_def = PIPELINE_STAGES[idx]
                if stage_def["job_type"]:
                    create_job(db, workflow_id, stage_def["job_type"], stage_name=loop_back_stage,
                               input_data={"feedback": note, "loop_back_from": current_stage})
                break


def handle_review_feedback(db, workflow_id: str, reviewer_stage: str, passed: bool,
                           feedback: str = "", output_data: Dict = None):
    """
    Process agent review result. If passed, advance. If failed, loop back
    (up to MAX_REVIEW_ITERATIONS).
    """
    node = db.development_workflow_nodes.find_one({
        "workflow_id": workflow_id,
        "stage_name": reviewer_stage,
    })
    if not node:
        return

    iteration = node.get("iteration", 0)

    if passed or iteration >= MAX_REVIEW_ITERATIONS:
        if not passed:
            log_event(db, workflow_id, "max_iterations", "system", "orchestrator",
                      f"Max iterations ({MAX_REVIEW_ITERATIONS}) reached at {reviewer_stage}, advancing anyway")
        handle_stage_completion(db, workflow_id, reviewer_stage, output_data)
    else:
        # Loop back
        loop_back_stage = _get_review_loop_back(reviewer_stage)
        log_event(db, workflow_id, "review_rejected", "agent", reviewer_stage,
                  f"Review failed at {reviewer_stage}, looping back to {loop_back_stage}")

        now = datetime.utcnow()
        for idx, stage in enumerate(PIPELINE_STAGES):
            if stage["name"] == loop_back_stage:
                db.development_workflow_nodes.update_one(
                    {"workflow_id": workflow_id, "stage_name": loop_back_stage},
                    {"$inc": {"iteration": 1}, "$set": {"status": "running", "started_at": now}}
                )
                transition_workflow(db, workflow_id, "running", loop_back_stage, idx)
                stage_def = PIPELINE_STAGES[idx]
                if stage_def["job_type"]:
                    create_job(db, workflow_id, stage_def["job_type"], stage_name=loop_back_stage,
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

def push_timeline_update(db, workflow_id: str, stage: str, status: str,
                         message: str, visibility: str = "public"):
    """Create a status_update timeline entry from orchestrator events."""
    now = datetime.utcnow()
    label = STAGE_LABELS.get(stage, stage)

    # Look up actual workflow creator
    author_id = "system"
    author_name = "dble"
    wf = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if wf and wf.get("created_by"):
        user = db.users.find_one({"workos_user_id": wf["created_by"]})
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
    db.timeline_entries.insert_one(entry)


# --- Query helpers ---

def get_workflow(db, workflow_id: str) -> Optional[dict]:
    wf = db.development_workflows.find_one({"_id": ObjectId(workflow_id)})
    if wf:
        wf["_id"] = str(wf["_id"])
    return wf


def get_workflow_nodes(db, workflow_id: str) -> List[dict]:
    nodes = list(db.development_workflow_nodes.find({"workflow_id": workflow_id}).sort("stage_index", 1))
    for n in nodes:
        n["_id"] = str(n["_id"])
    return nodes


def get_workflow_jobs(db, workflow_id: str) -> List[dict]:
    jobs = list(db.development_workflow_jobs.find({"workflow_id": workflow_id}).sort("created_at", -1))
    for j in jobs:
        j["_id"] = str(j["_id"])
    return jobs
