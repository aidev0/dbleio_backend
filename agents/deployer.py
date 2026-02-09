#!/usr/bin/env python3
"""
Deployer - Handles deployment to various providers (Heroku, Vercel, Docker, custom).
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Optional
import httpx


async def deploy(db, workflow_id: str, project: dict, workspace_path: str, commit_sha: str) -> Dict:
    """
    Deploy based on project's deployment_config.
    Routes to the appropriate provider handler.
    Returns deployment result dict.
    """
    deployment_config = project.get("deployment_config") or {}
    targets = deployment_config.get("targets", [])

    if not targets:
        return {"status": "skipped", "message": "No deployment targets configured"}

    results = []
    for target in targets:
        provider = target.get("provider", "")
        handlers = {
            "heroku": _deploy_heroku,
            "vercel": _deploy_vercel,
            "docker": _deploy_docker,
            "custom": _deploy_custom_script,
        }
        handler = handlers.get(provider)
        if not handler:
            results.append({"target": target.get("name"), "status": "skipped", "error": f"Unknown provider: {provider}"})
            continue

        try:
            result = await handler(target, workspace_path, commit_sha)
            record = create_deployment_record(db, workflow_id, target, commit_sha, result)
            results.append(result)
        except Exception as e:
            error_result = {"target": target.get("name"), "status": "failed", "error": str(e)}
            create_deployment_record(db, workflow_id, target, commit_sha, error_result)
            results.append(error_result)

    # Run health checks
    for i, target in enumerate(targets):
        health_url = target.get("health_check_url")
        if health_url and results[i].get("status") == "success":
            healthy = await run_health_check(health_url)
            results[i]["healthy"] = healthy

    return {"deployments": results}


async def _deploy_heroku(target: dict, workspace_path: str, commit_sha: str) -> Dict:
    """Deploy to Heroku using git push."""
    config = target.get("config", {})
    app_name = config.get("app_name", "")
    if not app_name:
        raise ValueError("Heroku deployment requires config.app_name")

    repo_path = _find_repo_path(workspace_path, config)

    proc = await asyncio.create_subprocess_exec(
        "git", "push", f"https://git.heroku.com/{app_name}.git", "HEAD:main", "--force",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

    if proc.returncode != 0:
        raise RuntimeError(f"Heroku deploy failed: {stderr.decode()[:500]}")

    return {
        "target": target.get("name"),
        "status": "success",
        "provider": "heroku",
        "url": target.get("url", f"https://{app_name}.herokuapp.com"),
        "commit_sha": commit_sha,
    }


async def _deploy_vercel(target: dict, workspace_path: str, commit_sha: str) -> Dict:
    """Deploy to Vercel using CLI."""
    config = target.get("config", {})
    repo_path = _find_repo_path(workspace_path, config)

    cmd = ["vercel", "--prod", "--yes"]
    if config.get("token"):
        cmd.extend(["--token", config["token"]])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

    deploy_url = stdout.decode().strip().split('\n')[-1]

    return {
        "target": target.get("name"),
        "status": "success" if proc.returncode == 0 else "failed",
        "provider": "vercel",
        "url": deploy_url or target.get("url", ""),
        "commit_sha": commit_sha,
    }


async def _deploy_docker(target: dict, workspace_path: str, commit_sha: str) -> Dict:
    """Build and push Docker image."""
    config = target.get("config", {})
    image = config.get("image", "")
    if not image:
        raise ValueError("Docker deployment requires config.image")

    repo_path = _find_repo_path(workspace_path, config)
    tag = f"{image}:{commit_sha[:8]}"

    # Build
    proc = await asyncio.create_subprocess_exec(
        "docker", "build", "-t", tag, ".",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"Docker build failed: {stderr.decode()[:500]}")

    # Push
    proc = await asyncio.create_subprocess_exec(
        "docker", "push", tag,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Docker push failed: {stderr.decode()[:500]}")

    return {
        "target": target.get("name"),
        "status": "success",
        "provider": "docker",
        "image": tag,
        "commit_sha": commit_sha,
    }


async def _deploy_custom_script(target: dict, workspace_path: str, commit_sha: str) -> Dict:
    """Run a custom deploy script."""
    config = target.get("config", {})
    script = config.get("script", "")
    if not script:
        raise ValueError("Custom deployment requires config.script")

    repo_path = _find_repo_path(workspace_path, config)

    proc = await asyncio.create_subprocess_shell(
        script,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "COMMIT_SHA": commit_sha, "WORKSPACE": repo_path},
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

    return {
        "target": target.get("name"),
        "status": "success" if proc.returncode == 0 else "failed",
        "provider": "custom",
        "output": stdout.decode()[:1000],
        "commit_sha": commit_sha,
    }


async def run_health_check(url: str, max_retries: int = 3) -> bool:
    """GET a health URL with retries. Returns True if 2xx response."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(url)
                if 200 <= response.status_code < 300:
                    return True
            except Exception:
                pass
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
    return False


def create_deployment_record(db, workflow_id: str, target: dict, commit_sha: str, result: Dict) -> str:
    """Create an immutable deployment record."""
    record = {
        "workflow_id": workflow_id,
        "target_name": target.get("name", ""),
        "provider": target.get("provider", ""),
        "commit_sha": commit_sha,
        "status": result.get("status", "unknown"),
        "url": result.get("url", ""),
        "error": result.get("error"),
        "metadata": result,
        "created_at": datetime.utcnow(),
    }
    result_insert = db.development_deployment_records.insert_one(record)
    return str(result_insert.inserted_id)


def _find_repo_path(workspace_path: str, config: dict) -> str:
    """Find the repo directory within the workspace."""
    repo_name = config.get("repo_name", "")
    if repo_name:
        path = os.path.join(workspace_path, repo_name)
        if os.path.isdir(path):
            return path
    # Fallback: find first directory in workspace
    if os.path.isdir(workspace_path):
        for entry in os.listdir(workspace_path):
            full = os.path.join(workspace_path, entry)
            if os.path.isdir(full) and not entry.startswith('.'):
                return full
    return workspace_path
