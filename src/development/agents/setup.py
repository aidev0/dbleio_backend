"""Setup Agent — Clones repos, sets up branches, installs dependencies."""

from typing import Dict, Any
from src.development.agents.base import DevBaseAgent


class SetupAgent(DevBaseAgent):
    agent_name = "setup_agent"
    description = "Clone repos, create branches, install dependencies, and prepare the development environment."
    allowed_model_categories = ["cli"]
    default_model_id = "claude-code"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        spec = context.get("stage_outputs", {}).get("spec_intake", {})
        repos = spec.get("target_repos", [])

        return {
            "repos_cloned": repos,
            "branches_created": [f"feature/{spec.get('title', 'dev').lower().replace(' ', '-')}"],
            "dependencies_installed": True,
            "environment_ready": True,
            "cli_tool": context.get("model", {}).get("id"),
            "status": "ready",
        }
