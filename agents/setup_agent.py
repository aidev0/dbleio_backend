#!/usr/bin/env python3
"""
Development Setup Agent - Manages workspace creation, git operations, and cleanup.
"""

import asyncio
import os
import shutil
from datetime import datetime
from typing import Dict, Optional


WORKSPACE_BASE = "/tmp/dble-workflow"


def workspace_path(workflow_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, workflow_id)


async def _run_git(args: list, cwd: str) -> str:
    """Run a git command and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr.decode()[:500]}")
    return stdout.decode().strip()


async def setup_workspace(db, workflow_id: str, project: dict) -> Dict:
    """
    Create workspace directory, clone repos, checkout feature branch, push, and record branches.
    Returns dict with workspace_path and branch info.
    """
    ws_path = workspace_path(workflow_id)
    os.makedirs(ws_path, exist_ok=True)

    repos = project.get("repos", [])
    branches = []
    feature_branch = f"dble/workflow-{workflow_id[:8]}"

    for repo in repos:
        repo_url = repo.get("url", "")
        repo_name = repo.get("name", "repo")
        base_branch = repo.get("branch", "main")
        repo_dir = os.path.join(ws_path, repo_name)

        # Clone
        await _run_git(["clone", "--depth", "50", "-b", base_branch, repo_url, repo_dir], ws_path)

        # Create and push feature branch
        await _run_git(["checkout", "-b", feature_branch], repo_dir)
        await _run_git(["push", "-u", "origin", feature_branch], repo_dir)

        branch_record = {
            "workflow_id": workflow_id,
            "repo_name": repo_name,
            "repo_url": repo_url,
            "base_branch": base_branch,
            "feature_branch": feature_branch,
            "repo_path": repo_dir,
            "created_at": datetime.utcnow(),
        }
        db.development_branches.insert_one(branch_record)
        branches.append({
            "repo_name": repo_name,
            "feature_branch": feature_branch,
            "base_branch": base_branch,
            "repo_path": repo_dir,
        })

    return {
        "workspace_path": ws_path,
        "feature_branch": feature_branch,
        "branches": branches,
    }


def cleanup_workspace(workflow_id: str):
    """Delete the workspace directory for a workflow."""
    ws_path = workspace_path(workflow_id)
    if os.path.exists(ws_path):
        shutil.rmtree(ws_path, ignore_errors=True)
