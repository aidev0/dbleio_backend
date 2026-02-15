"""Spec Intake Agent — Parses and structures the specification from PM/client."""

from typing import Dict, Any
from src.development.agents.base import DevBaseAgent


class SpecIntakeAgent(DevBaseAgent):
    agent_name = "spec_intake_agent"
    description = "Parse client specification into structured requirements, acceptance criteria, and target repos."
    allowed_model_categories = ["llm"]
    default_model_id = "claude-sonnet-4-5"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": input_data.get("title", ""),
            "description": input_data.get("description", ""),
            "requirements": input_data.get("requirements", []),
            "acceptance_criteria": input_data.get("acceptance_criteria", []),
            "target_repos": input_data.get("target_repos", []),
            "priority": input_data.get("priority", "medium"),
            "model_used": context.get("model", {}).get("id"),
            "status": "parsed",
        }
