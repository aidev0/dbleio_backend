#!/usr/bin/env python3
"""
Development Workflow Events - Immutable event log for workflow activity.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId


def ensure_indexes(db):
    db.development_workflow_events.create_index("workflow_id")
    db.development_workflow_events.create_index([("workflow_id", 1), ("timestamp", 1)])


def log_event(
    db,
    workflow_id: str,
    event_type: str,
    actor_type: str,  # "system", "agent", "user"
    actor_id: str,
    message: str,
    node_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Log a workflow event. Returns the event ID."""
    event = {
        "workflow_id": workflow_id,
        "event_type": event_type,
        "actor_type": actor_type,
        "actor_id": actor_id,
        "message": message,
        "node_id": node_id,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow(),
    }
    result = db.development_workflow_events.insert_one(event)
    return str(result.inserted_id)


def get_workflow_events(
    db,
    workflow_id: str,
    limit: int = 100,
    event_type: Optional[str] = None,
) -> List[dict]:
    """Get events for a workflow, most recent first."""
    query: Dict[str, Any] = {"workflow_id": workflow_id}
    if event_type:
        query["event_type"] = event_type

    events = list(
        db.development_workflow_events
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )
    for e in events:
        e["_id"] = str(e["_id"])
    return events
