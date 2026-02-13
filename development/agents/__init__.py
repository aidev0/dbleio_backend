"""
Development Pipeline Agents

Each agent handles a specific pipeline stage with configurable model selection.
Agents persist their own state via DevAgentStateStore.
"""

from development.agents.base import DevBaseAgent
from development.agents.spec_intake import SpecIntakeAgent
from development.agents.setup import SetupAgent
from development.agents.planner import PlannerAgent
from development.agents.plan_reviewer import PlanReviewerAgent
from development.agents.developer import DeveloperAgent
from development.agents.code_reviewer import CodeReviewerAgent
from development.agents.validator import ValidatorAgent
from development.agents.commit_pr import CommitPRAgent
from development.agents.deployer import DeployerAgent
from development.agents.done import DoneAgent

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
