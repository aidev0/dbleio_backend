#!/usr/bin/env python3
"""
Development Worker - Polls for queued jobs and executes them.
Run as a standalone process: python -m agents.worker
"""

import asyncio
import os
import sys
import uuid
import traceback
from datetime import datetime, timedelta
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'dble_db')
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB_NAME]

WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"
POLL_INTERVAL = 5  # seconds


async def poll_and_execute():
    """Main worker loop: poll for jobs, claim, execute."""
    print(f"[{WORKER_ID}] Development worker started, polling every {POLL_INTERVAL}s")

    # Ensure indexes
    from agents.orchestrator import ensure_indexes as orch_indexes
    from agents.events import ensure_indexes as event_indexes
    orch_indexes(db)
    event_indexes(db)

    while True:
        try:
            job = claim_job()
            if job:
                await execute_job(job)
            else:
                await asyncio.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print(f"[{WORKER_ID}] Shutting down")
            break
        except Exception as e:
            print(f"[{WORKER_ID}] Poll error: {e}")
            traceback.print_exc()
            await asyncio.sleep(POLL_INTERVAL)


def claim_job():
    """Attempt to claim a queued job using atomic find_one_and_update."""
    now = datetime.utcnow()
    job = db.development_workflow_jobs.find_one_and_update(
        {
            "status": "queued",
            "run_after": {"$lte": now},
        },
        {
            "$set": {
                "status": "running",
                "claimed_by": WORKER_ID,
                "started_at": now,
            }
        },
        sort=[("run_after", 1)],
        return_document=True,
    )
    if job:
        job["_id"] = str(job["_id"])
        print(f"[{WORKER_ID}] Claimed job {job['_id']} (type={job['job_type']}, stage={job.get('stage_name', '')})")
    return job


async def execute_job(job: dict):
    """Route job to the appropriate handler and handle result/failures."""
    from agents.orchestrator import handle_stage_completion, handle_review_feedback, create_job
    from agents.events import log_event

    job_id = job["_id"]
    workflow_id = job["workflow_id"]
    job_type = job["job_type"]
    stage_name = job.get("stage_name", "")

    handlers = {
        "setup": handle_setup_job,
        "plan": handle_plan_job,
        "review": handle_review_job,
        "implement": handle_implement_job,
        "validate": handle_validate_job,
        "commit_pr": handle_commit_pr_job,
        "deploy": handle_deploy_job,
        "notify": handle_notify_job,
    }

    handler = handlers.get(job_type)
    if not handler:
        _fail_job(job_id, f"Unknown job type: {job_type}")
        return

    try:
        log_event(db, workflow_id, "job_started", "system", WORKER_ID,
                  f"Job {job_type} started for stage {stage_name}", node_id=stage_name)

        output = await handler(job)

        # Mark job completed
        db.development_workflow_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": "completed", "output_data": output or {}, "completed_at": datetime.utcnow()}}
        )

        log_event(db, workflow_id, "job_completed", "system", WORKER_ID,
                  f"Job {job_type} completed for stage {stage_name}", node_id=stage_name)

        # Advance the workflow
        if job_type == "review":
            passed = output.get("passed", True)
            handle_review_feedback(db, workflow_id, stage_name, passed,
                                   feedback=output.get("feedback", ""), output_data=output)
        else:
            handle_stage_completion(db, workflow_id, stage_name, output)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[{WORKER_ID}] Job {job_id} failed: {error_msg}")
        traceback.print_exc()

        attempt = job.get("attempt", 0) + 1
        max_attempts = job.get("max_attempts", 3)

        if attempt < max_attempts:
            # Re-queue with exponential backoff
            delay = 30 * (2 ** (attempt - 1))
            db.development_workflow_jobs.update_one(
                {"_id": ObjectId(job_id)},
                {"$set": {
                    "status": "queued",
                    "claimed_by": None,
                    "attempt": attempt,
                    "run_after": datetime.utcnow() + timedelta(seconds=delay),
                    "error": error_msg,
                }}
            )
            log_event(db, workflow_id, "job_retrying", "system", WORKER_ID,
                      f"Job {job_type} failed (attempt {attempt}/{max_attempts}), retrying in {delay}s",
                      node_id=stage_name, metadata={"error": error_msg})
        else:
            _fail_job(job_id, error_msg)
            # Mark workflow as failed
            db.development_workflows.update_one(
                {"_id": ObjectId(workflow_id)},
                {"$set": {"status": "failed", "updated_at": datetime.utcnow()}}
            )
            log_event(db, workflow_id, "workflow_failed", "system", WORKER_ID,
                      f"Workflow failed at stage {stage_name}: {error_msg}",
                      node_id=stage_name, metadata={"error": error_msg})


def _fail_job(job_id: str, error: str):
    db.development_workflow_jobs.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": {"status": "failed", "error": error, "completed_at": datetime.utcnow()}}
    )


# --- Job Handlers ---

async def handle_setup_job(job: dict) -> dict:
    """Clone repos, create feature branches."""
    from agents.setup_agent import setup_workspace
    from agents.orchestrator import get_workflow

    workflow = get_workflow(db, job["workflow_id"])
    project = db.projects.find_one({"_id": ObjectId(workflow["project_id"])})
    if not project:
        raise ValueError(f"Project {workflow['project_id']} not found")

    result = await setup_workspace(db, job["workflow_id"], project)
    return result


async def handle_plan_job(job: dict) -> dict:
    """Run planning agent to create implementation plan."""
    from agents.executor import execute_agent, AgentConfig, AgentProvider

    workflow, spec, agent_config = _load_workflow_context(job)
    config = _build_agent_config(agent_config)

    prompt = _build_plan_prompt(spec, job.get("input_data", {}))
    results = await execute_agent(config, prompt, stage="plan")

    best = results[0] if results else None
    if not best or not best.success:
        raise RuntimeError(f"Planning agent failed: {best.error if best else 'no result'}")

    return {
        "plan": best.intent.explanation if best.intent else best.raw_output,
        "provider": best.provider.value,
        "duration_ms": best.duration_ms,
    }


async def handle_review_job(job: dict) -> dict:
    """Run review agent on plan or code."""
    from agents.executor import execute_agent, AgentConfig

    workflow, spec, agent_config = _load_workflow_context(job)
    config = _build_agent_config(agent_config)

    stage_name = job.get("stage_name", "")
    input_data = job.get("input_data", {})

    if stage_name == "plan_reviewer":
        prompt = _build_plan_review_prompt(input_data)
    else:
        prompt = _build_code_review_prompt(input_data)

    results = await execute_agent(config, prompt, stage="review")

    best = results[0] if results else None
    if not best or not best.success:
        raise RuntimeError(f"Review agent failed: {best.error if best else 'no result'}")

    output_text = best.intent.explanation if best.intent else best.raw_output
    passed = _parse_review_verdict(output_text)

    return {
        "passed": passed,
        "feedback": output_text,
        "provider": best.provider.value,
        "duration_ms": best.duration_ms,
    }


async def handle_implement_job(job: dict) -> dict:
    """Run development agent to write code."""
    from agents.executor import execute_agent, AgentConfig

    workflow, spec, agent_config = _load_workflow_context(job)
    config = _build_agent_config(agent_config)
    workspace = _get_workspace_path(job["workflow_id"])

    input_data = job.get("input_data", {})
    prompt = _build_implement_prompt(spec, input_data)
    results = await execute_agent(config, prompt, workspace_path=workspace, stage="implement")

    best = results[0] if results else None
    if not best or not best.success:
        raise RuntimeError(f"Implementation agent failed: {best.error if best else 'no result'}")

    return {
        "explanation": best.intent.explanation if best.intent else "",
        "file_edits": [{"path": fe.path, "action": fe.action} for fe in (best.intent.file_edits if best.intent else [])],
        "commands_run": best.intent.commands if best.intent else [],
        "provider": best.provider.value,
        "duration_ms": best.duration_ms,
    }


async def handle_validate_job(job: dict) -> dict:
    """Run validation (tests, linting) in the workspace."""
    from agents.executor import execute_agent, AgentConfig

    workflow, spec, agent_config = _load_workflow_context(job)
    config = _build_agent_config(agent_config)
    workspace = _get_workspace_path(job["workflow_id"])

    prompt = "Run the test suite and linting for this project. Report pass/fail results."
    results = await execute_agent(config, prompt, workspace_path=workspace, stage="validate")

    best = results[0] if results else None
    if not best or not best.success:
        raise RuntimeError(f"Validation agent failed: {best.error if best else 'no result'}")

    return {
        "validation_output": best.intent.explanation if best.intent else best.raw_output,
        "provider": best.provider.value,
        "duration_ms": best.duration_ms,
    }


async def handle_commit_pr_job(job: dict) -> dict:
    """Commit changes and create a PR."""
    import asyncio
    workspace = _get_workspace_path(job["workflow_id"])

    branches = list(db.development_branches.find({"workflow_id": job["workflow_id"]}))
    pr_urls = []

    for branch in branches:
        repo_path = branch.get("repo_path", "")
        feature_branch = branch.get("feature_branch", "")
        base_branch = branch.get("base_branch", "main")

        if not os.path.isdir(repo_path):
            continue

        # Stage, commit, push
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A",
            cwd=repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        workflow = get_workflow_for_job(job)
        commit_msg = f"feat: {workflow.get('title', 'workflow changes')}"
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", commit_msg, "--allow-empty",
            cwd=repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        proc = await asyncio.create_subprocess_exec(
            "git", "push", "origin", feature_branch,
            cwd=repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        # Create PR via gh CLI
        proc = await asyncio.create_subprocess_exec(
            "gh", "pr", "create",
            "--title", commit_msg,
            "--body", f"Automated PR from DBLE workflow {job['workflow_id']}",
            "--base", base_branch,
            "--head", feature_branch,
            cwd=repo_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        pr_url = stdout.decode().strip()
        if pr_url:
            pr_urls.append(pr_url)

    return {"pr_urls": pr_urls}


async def handle_deploy_job(job: dict) -> dict:
    """Deploy using the deployer module."""
    from agents.deployer import deploy
    from agents.orchestrator import get_workflow

    workflow = get_workflow(db, job["workflow_id"])
    project = db.projects.find_one({"_id": ObjectId(workflow["project_id"])})
    workspace = _get_workspace_path(job["workflow_id"])

    commit_sha = job.get("input_data", {}).get("commit_sha", "HEAD")
    result = await deploy(db, job["workflow_id"], project, workspace, commit_sha)
    return result


async def handle_notify_job(job: dict) -> dict:
    """Send notifications (placeholder — logs the event)."""
    from agents.events import log_event
    log_event(db, job["workflow_id"], "notification", "system", WORKER_ID,
              f"Notification for stage {job.get('stage_name', '')}")
    return {"notified": True}


# --- Helpers ---

def _load_workflow_context(job):
    """Load workflow, specification, and agent config for a job."""
    from agents.orchestrator import get_workflow
    workflow = get_workflow(db, job["workflow_id"])
    spec = db.development_specifications.find_one({"_id": ObjectId(workflow["specification_id"])}) if workflow else None
    agent_config = workflow.get("agent_config", {}) if workflow else {}
    return workflow, spec, agent_config


def get_workflow_for_job(job):
    from agents.orchestrator import get_workflow
    return get_workflow(db, job["workflow_id"])


def _build_agent_config(agent_config_dict):
    from agents.executor import AgentConfig, AgentProvider
    providers = agent_config_dict.get("providers", ["claude_api"])
    return AgentConfig(
        providers=[AgentProvider(p) for p in providers],
        mode=agent_config_dict.get("mode", "first"),
        timeout=agent_config_dict.get("timeout", 300),
    )


def _get_workspace_path(workflow_id: str) -> str:
    from agents.setup_agent import workspace_path
    return workspace_path(workflow_id)


def _build_plan_prompt(spec, input_data):
    feedback = input_data.get("feedback", "")
    spec_text = spec.get("spec_text", "") if spec else ""
    title = spec.get("title", "") if spec else ""
    acceptance = spec.get("acceptance_criteria", "") if spec else ""

    prompt = f"""Create a detailed implementation plan for the following specification.

Title: {title}
Specification: {spec_text}
Acceptance Criteria: {acceptance}
"""
    if feedback:
        prompt += f"\nPrevious feedback to address:\n{feedback}\n"
    prompt += "\nProvide a step-by-step plan with file changes, architecture decisions, and testing strategy."
    return prompt


def _build_plan_review_prompt(input_data):
    plan = input_data.get("plan", "")
    return f"""Review the following implementation plan. Check for:
1. Completeness - does it cover all requirements?
2. Correctness - is the approach technically sound?
3. Risk - are there any concerns?

If the plan is acceptable, start your response with "APPROVED".
If changes are needed, start with "CHANGES_NEEDED" and explain what needs to change.

Plan:
{plan}
"""


def _build_code_review_prompt(input_data):
    explanation = input_data.get("explanation", "")
    file_edits = input_data.get("file_edits", [])
    return f"""Review the following code changes. Check for:
1. Correctness
2. Security vulnerabilities
3. Performance issues
4. Code quality

If the code is acceptable, start your response with "APPROVED".
If changes are needed, start with "CHANGES_NEEDED" and explain.

Changes summary: {explanation}
Files modified: {', '.join(f.get('path', '') for f in file_edits)}
"""


def _build_implement_prompt(spec, input_data):
    feedback = input_data.get("feedback", "")
    plan = input_data.get("plan", "")
    spec_text = spec.get("spec_text", "") if spec else ""
    title = spec.get("title", "") if spec else ""

    prompt = f"""Implement the following specification.

Title: {title}
Specification: {spec_text}

Implementation Plan:
{plan}
"""
    if feedback:
        prompt += f"\nFeedback to address:\n{feedback}\n"
    prompt += "\nWrite the code. Use ```filename.ext blocks to show file contents."
    return prompt


def _parse_review_verdict(text: str) -> bool:
    """Parse whether a review passed based on the agent output."""
    first_line = text.strip().split('\n')[0].upper()
    return "APPROVED" in first_line


# --- Workspace cleanup ---

async def cleanup_terminal_workflows():
    """Periodically clean up workspaces for completed/failed workflows."""
    from agents.setup_agent import cleanup_workspace
    terminal_workflows = list(db.development_workflows.find({
        "status": {"$in": ["completed", "failed", "cancelled"]}
    }))
    for wf in terminal_workflows:
        wf_id = str(wf["_id"])
        ws_path = _get_workspace_path(wf_id)
        if os.path.exists(ws_path):
            cleanup_workspace(wf_id)
            print(f"[{WORKER_ID}] Cleaned up workspace for workflow {wf_id}")


if __name__ == "__main__":
    asyncio.run(poll_and_execute())
