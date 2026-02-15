"""
Development Pipeline Agents

Each agent handles a specific pipeline stage with configurable model selection.
Agents persist their own state via DevAgentStateStore.
"""

from src.development.agents.base import DevBaseAgent
from src.development.agents.spec_intake import SpecIntakeAgent
from src.development.agents.setup import SetupAgent
from src.development.agents.planner import PlannerAgent
from src.development.agents.plan_reviewer import PlanReviewerAgent
from src.development.agents.developer import DeveloperAgent
from src.development.agents.code_reviewer import CodeReviewerAgent
from src.development.agents.validator import ValidatorAgent
from src.development.agents.commit_pr import CommitPRAgent
from src.development.agents.deployer import DeployerAgent
from src.development.agents.done import DoneAgent

# Registry: agent_name → class
DEV_AGENT_REGISTRY = {
    "spec_intake_agent": SpecIntakeAgent,
    "setup_agent": SetupAgent,
    "planner_agent": PlannerAgent,
    "plan_reviewer_agent": PlanReviewerAgent,
    "developer_agent": DeveloperAgent,
    "code_reviewer_agent": CodeReviewerAgent,
    "validator_agent": ValidatorAgent,
    "commit_pr_agent": CommitPRAgent,
    "deployer_agent": DeployerAgent,
    "done_agent": DoneAgent,
}


def get_dev_agent(agent_name: str, workflow_id: str, model_id: str = None) -> DevBaseAgent:
    """Factory: instantiate a dev agent by name, optionally overriding the model."""
    cls = DEV_AGENT_REGISTRY.get(agent_name)
    if not cls:
        raise ValueError(f"Unknown dev agent: {agent_name}")
    return cls(workflow_id=workflow_id, model_id=model_id)
