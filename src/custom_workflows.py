#!/usr/bin/env python3
"""
Custom Workflows API - CRUD for customer-facing workflows with DB-driven graph nodes and edges.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict
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

router = APIRouter(prefix="/api/custom-workflows", tags=["custom-workflows"])


# --- Models ---

class WorkflowCreate(BaseModel):
    organization_id: str
    title: str
    description: Optional[str] = ""
    status: str = "draft"
    source_dev_workflow_id: Optional[str] = None

class WorkflowUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class NodeCreate(BaseModel):
    label: str
    node_type: str = "step"
    status: str = "pending"
    position_x: float = 0
    position_y: float = 0
    config: Optional[Dict] = None

class NodeUpdate(BaseModel):
    label: Optional[str] = None
    node_type: Optional[str] = None
    status: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    config: Optional[Dict] = None

class EdgeCreate(BaseModel):
    source_node_id: str
    target_node_id: str
    label: Optional[str] = ""
    condition: Optional[str] = None
    edge_type: str = "default"


# --- Helpers ---

def workflow_helper(doc) -> dict:
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc

def node_helper(doc) -> dict:
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc

def edge_helper(doc) -> dict:
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


# --- Workflow Endpoints ---

@router.get("")
async def list_workflows(request: Request, status: Optional[str] = None):
    """List custom workflows for user's organizations."""
    try:
        from src.auth import require_user_id
        workos_user_id = require_user_id(request)

        user = db.users.find_one({"workos_user_id": workos_user_id})
        org_ids = [o["_id"] for o in (user.get("organizations", []) if user else [])]
        if not org_ids:
            return []

        query: Dict = {
            "organization_id": {"$in": org_ids},
            "is_deleted": {"$ne": True},
        }
        if status:
            query["status"] = status

        workflows = list(db.custom_workflows.find(query).sort("created_at", -1))
        return [workflow_helper(wf) for wf in workflows]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=201)
async def create_workflow(body: WorkflowCreate, request: Request):
    """Create a custom workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        verify_org_membership(db, body.organization_id, workos_user_id)

        now = datetime.utcnow()
        doc = {
            "organization_id": body.organization_id,
            "title": body.title,
            "description": body.description or "",
            "status": body.status,
            "source_dev_workflow_id": body.source_dev_workflow_id,
            "created_by": workos_user_id,
            "is_deleted": False,
            "created_at": now,
            "updated_at": now,
        }
        result = db.custom_workflows.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, request: Request):
    """Get a single custom workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)
        return workflow_helper(wf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}")
async def update_workflow(workflow_id: str, body: WorkflowUpdate, request: Request):
    """Update a custom workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        updates = {"updated_at": datetime.utcnow()}
        if body.title is not None:
            updates["title"] = body.title
        if body.description is not None:
            updates["description"] = body.description
        if body.status is not None:
            updates["status"] = body.status

        db.custom_workflows.update_one({"_id": ObjectId(workflow_id)}, {"$set": updates})
        updated = db.custom_workflows.find_one({"_id": ObjectId(workflow_id)})
        return workflow_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, request: Request):
    """Soft delete a custom workflow."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        db.custom_workflows.update_one(
            {"_id": ObjectId(workflow_id)},
            {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Graph Endpoints ---

@router.get("/{workflow_id}/graph")
async def get_graph(workflow_id: str, request: Request):
    """Get all nodes and edges for a workflow's graph."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        nodes = list(db.custom_workflow_nodes.find({"workflow_id": workflow_id}))
        edges = list(db.custom_workflow_edges.find({"workflow_id": workflow_id}))
        return {
            "nodes": [node_helper(n) for n in nodes],
            "edges": [edge_helper(e) for e in edges],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/nodes", status_code=201)
async def create_node(workflow_id: str, body: NodeCreate, request: Request):
    """Add a node to a workflow graph."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        now = datetime.utcnow()
        doc = {
            "workflow_id": workflow_id,
            "label": body.label,
            "node_type": body.node_type,
            "status": body.status,
            "position_x": body.position_x,
            "position_y": body.position_y,
            "config": body.config or {},
            "created_at": now,
            "updated_at": now,
        }
        result = db.custom_workflow_nodes.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{workflow_id}/nodes/{node_id}")
async def update_node(workflow_id: str, node_id: str, body: NodeUpdate, request: Request):
    """Update a graph node."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        node = db.custom_workflow_nodes.find_one({"_id": ObjectId(node_id), "workflow_id": workflow_id})
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")

        updates = {"updated_at": datetime.utcnow()}
        if body.label is not None:
            updates["label"] = body.label
        if body.node_type is not None:
            updates["node_type"] = body.node_type
        if body.status is not None:
            updates["status"] = body.status
        if body.position_x is not None:
            updates["position_x"] = body.position_x
        if body.position_y is not None:
            updates["position_y"] = body.position_y
        if body.config is not None:
            updates["config"] = body.config

        db.custom_workflow_nodes.update_one({"_id": ObjectId(node_id)}, {"$set": updates})
        updated = db.custom_workflow_nodes.find_one({"_id": ObjectId(node_id)})
        return node_helper(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}/nodes/{node_id}")
async def delete_node(workflow_id: str, node_id: str, request: Request):
    """Delete a graph node and its connected edges."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        node = db.custom_workflow_nodes.find_one({"_id": ObjectId(node_id), "workflow_id": workflow_id})
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")

        # Delete connected edges
        db.custom_workflow_edges.delete_many({
            "workflow_id": workflow_id,
            "$or": [{"source_node_id": node_id}, {"target_node_id": node_id}],
        })
        db.custom_workflow_nodes.delete_one({"_id": ObjectId(node_id)})
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/edges", status_code=201)
async def create_edge(workflow_id: str, body: EdgeCreate, request: Request):
    """Add an edge to a workflow graph."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        doc = {
            "workflow_id": workflow_id,
            "source_node_id": body.source_node_id,
            "target_node_id": body.target_node_id,
            "label": body.label or "",
            "condition": body.condition,
            "edge_type": body.edge_type,
            "created_at": datetime.utcnow(),
        }
        result = db.custom_workflow_edges.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}/edges/{edge_id}")
async def delete_edge(workflow_id: str, edge_id: str, request: Request):
    """Delete a graph edge."""
    try:
        from src.auth import require_user_id
        from src.role_helpers import verify_org_membership

        workos_user_id = require_user_id(request)
        wf = db.custom_workflows.find_one({"_id": ObjectId(workflow_id), "is_deleted": {"$ne": True}})
        if not wf:
            raise HTTPException(status_code=404, detail="Workflow not found")
        verify_org_membership(db, wf["organization_id"], workos_user_id)

        edge = db.custom_workflow_edges.find_one({"_id": ObjectId(edge_id), "workflow_id": workflow_id})
        if not edge:
            raise HTTPException(status_code=404, detail="Edge not found")

        db.custom_workflow_edges.delete_one({"_id": ObjectId(edge_id)})
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
