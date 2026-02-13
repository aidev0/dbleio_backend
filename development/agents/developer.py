"""Developer Agent — Writes code based on the approved plan."""

from typing import Dict, Any
from development.agents.base import DevBaseAgent


class DeveloperAgent(DevBaseAgent):
    agent_name = "developer_agent"
    description = "Write code based on the approved implementation plan using CLI or LLM tools."
    allowed_model_categories = ["cli", "llm"]
    default_model_id = "claude-code"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        plan = context.get("stage_outputs", {}).get("planner", {})

        return {
            "files_modified": plan.get("files_to_modify", []),
            "files_created": plan.get("files_to_create", []),
            "lines_added": 0,
            "lines_removed": 0,
            "cli_tool": context.get("model", {}).get("id"),
            "status": "implemented",
        }
