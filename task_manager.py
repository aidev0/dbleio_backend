#!/usr/bin/env python3
"""
Background task management system
Tracks status of long-running async tasks
"""

from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import threading

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskManager:
    """Singleton task manager for tracking background tasks"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks = {}
        return cls._instance

    def create_task(self, task_id: str, task_type: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new task with pending status"""
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": TaskStatus.PENDING,
            "progress": 0,
            "message": "Task created",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "result": None,
            "error": None
        }
        self._tasks[task_id] = task
        return task

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ):
        """Update task status and progress"""
        if task_id not in self._tasks:
            return None

        task = self._tasks[task_id]

        if status is not None:
            task["status"] = status
        if progress is not None:
            task["progress"] = min(100, max(0, progress))
        if message is not None:
            task["message"] = message
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error
            task["status"] = TaskStatus.FAILED

        task["updated_at"] = datetime.utcnow().isoformat()

        return task

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return self._tasks.get(task_id)

    def delete_task(self, task_id: str):
        """Delete a task"""
        if task_id in self._tasks:
            del self._tasks[task_id]

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up tasks older than specified hours"""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_delete = []

        for task_id, task in self._tasks.items():
            task_time = datetime.fromisoformat(task["updated_at"])
            if task_time < cutoff and task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                to_delete.append(task_id)

        for task_id in to_delete:
            del self._tasks[task_id]

        return len(to_delete)

# Global instance
task_manager = TaskManager()
