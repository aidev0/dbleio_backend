#!/usr/bin/env python3
"""
Task status API routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from task_manager import task_manager

# Create router
router = APIRouter(prefix="/api/tasks", tags=["tasks"])

class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress: int
    message: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """Get status of a background task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed or failed task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task_manager.delete_task(task_id)
    return {"success": True, "message": "Task deleted"}
