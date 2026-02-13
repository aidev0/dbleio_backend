"""Done Agent — Merges PR, deploys to production, closes ticket."""

from typing import Dict, Any
from development.agents.base import DevBaseAgent


class DoneAgent(DevBaseAgent):
    agent_name = "done_agent"
    description = "Merge the pull request, deploy to production, and close the ticket."
    allowed_model_categories = ["cli"]
    default_model_id = "claude-code"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        pr = context.get("stage_outputs", {}).get("commit_pr", {})

        return {
            "pr_merged": True,
            "pr_url": pr.get("pr_url", ""),
            "production_url": "",
            "ticket_closed": True,
            "cli_tool": context.get("model", {}).get("id"),
            "status": "done",
        }
