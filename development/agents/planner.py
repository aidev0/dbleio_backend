"""Planner Agent — Analyzes spec + codebase and creates an implementation plan."""

from typing import Dict, Any
from development.agents.base import DevBaseAgent


class PlannerAgent(DevBaseAgent):
    agent_name = "planner_agent"
    description = "Analyze the specification and codebase to produce a detailed implementation plan."
    allowed_model_categories = ["llm"]
    default_model_id = "claude-opus-4-6"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        spec = context.get("stage_outputs", {}).get("spec_intake", {})

        return {
            "plan_title": spec.get("title", "Implementation Plan"),
            "steps": [],
            "files_to_modify": [],
            "files_to_create": [],
            "estimated_complexity": "medium",
            "risks": [],
            "model_used": context.get("model", {}).get("id"),
            "status": "planned",
        }
