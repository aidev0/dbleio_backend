"""
Development Pipeline Definition

13-stage development pipeline (mirrors existing dev workflow stages).
Each stage can be configured with a specific LLM or CLI model.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DevStageDefinition:
    name: str
    label: str
    node_type: str  # "human" | "agent"
    description: str
    agent_name: Optional[str] = None
    approval_required: bool = False
    reject_target: Optional[str] = None
    # Model selection: which model categories are allowed at this stage
    allowed_model_categories: List[str] = field(default_factory=lambda: ["llm"])
    default_model_id: Optional[str] = None  # Default model for this stage


DEV_PIPELINE_STAGES: List[DevStageDefinition] = [
    DevStageDefinition(
        name="spec_intake",
        label="Spec Intake",
        node_type="human",
        description="Client/PM submits the specification: requirements, acceptance criteria, target repos.",
        agent_name="spec_intake_agent",
        allowed_model_categories=["llm"],
        default_model_id="claude-sonnet-4-5",
    ),
    DevStageDefinition(
        name="setup",
        label="Setup",
        node_type="agent",
        description="Clone repos, set up branches, install dependencies, prepare environment.",
        agent_name="setup_agent",
        allowed_model_categories=["cli"],
        default_model_id="claude-code",
    ),
    DevStageDefinition(
        name="planner",
        label="Planner",
        node_type="agent",
        description="AI analyzes the spec and codebase, creates an implementation plan.",
        agent_name="planner_agent",
        allowed_model_categories=["llm"],
        default_model_id="claude-opus-4-6",
    ),
    DevStageDefinition(
        name="plan_reviewer",
        label="Plan Review",
        node_type="agent",
        description="AI reviews the plan for completeness, risks, and edge cases.",
        agent_name="plan_reviewer_agent",
        allowed_model_categories=["llm"],
        default_model_id="gemini-3-pro",
    ),
    DevStageDefinition(
        name="plan_approval",
        label="Plan Approval",
        node_type="human",
        description="FDE/FDM reviews and approves or rejects the plan.",
        approval_required=True,
        reject_target="planner",
    ),
    DevStageDefinition(
        name="developer",
        label="Developer",
        node_type="agent",
        description="AI writes the code based on the approved plan.",
        agent_name="developer_agent",
        allowed_model_categories=["cli", "llm"],
        default_model_id="claude-code",
    ),
    DevStageDefinition(
        name="code_reviewer",
        label="Code Review",
        node_type="agent",
        description="AI reviews the code for quality, bugs, security, and best practices.",
        agent_name="code_reviewer_agent",
        allowed_model_categories=["llm"],
        default_model_id="claude-sonnet-4-5",
    ),
    DevStageDefinition(
        name="validator",
        label="Validator",
        node_type="agent",
        description="Run tests, linters, type checks, and validate the implementation.",
        agent_name="validator_agent",
        allowed_model_categories=["cli"],
        default_model_id="claude-code",
    ),
    DevStageDefinition(
        name="commit_pr",
        label="Commit & PR",
        node_type="human",
        description="Commit changes, push branch, create pull request.",
        agent_name="commit_pr_agent",
        allowed_model_categories=["cli"],
        default_model_id="claude-code",
    ),
    DevStageDefinition(
        name="deployer",
        label="Deploy",
        node_type="human",
        description="Deploy to staging/preview environment for review.",
        agent_name="deployer_agent",
        allowed_model_categories=["cli"],
        default_model_id="claude-code",
    ),
    DevStageDefinition(
        name="qa_review",
        label="QA Review",
        node_type="human",
        description="QA team tests the deployment, verifies acceptance criteria.",
        approval_required=True,
        reject_target="developer",
    ),
    DevStageDefinition(
        name="client_review",
        label="Client Review",
        node_type="human",
        description="Client reviews the feature and gives final approval.",
        approval_required=True,
        reject_target="developer",
    ),
    DevStageDefinition(
        name="done",
        label="Done",
        node_type="human",
        description="Merge PR, deploy to production, close ticket.",
    ),
]

DEV_STAGE_KEYS = [s.name for s in DEV_PIPELINE_STAGES]
DEV_STAGE_MAP: Dict[str, DevStageDefinition] = {s.name: s for s in DEV_PIPELINE_STAGES}
DEV_STAGE_LABELS: Dict[str, str] = {s.name: s.label for s in DEV_PIPELINE_STAGES}


def get_dev_stage_index(key: str) -> int:
    return DEV_STAGE_KEYS.index(key)


def get_dev_next_stage(current_key: str) -> Optional[str]:
    idx = get_dev_stage_index(current_key)
    if idx + 1 < len(DEV_STAGE_KEYS):
        return DEV_STAGE_KEYS[idx + 1]
    return None


def get_dev_reject_target(current_key: str) -> Optional[str]:
    stage = DEV_STAGE_MAP.get(current_key)
    return stage.reject_target if stage else None
