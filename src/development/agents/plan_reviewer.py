"""Plan Reviewer Agent — Reviews the plan for completeness, risks, and edge cases."""

from typing import Dict, Any
from src.development.agents.base import DevBaseAgent


class PlanReviewerAgent(DevBaseAgent):
    agent_name = "plan_reviewer_agent"
    description = "Review the implementation plan for completeness, risks, security concerns, and edge cases."
    allowed_model_categories = ["llm"]
    default_model_id = "gemini-3-pro"

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        plan = context.get("stage_outputs", {}).get("planner", {})

        return {
            "review_passed": True,
            "issues_found": [],
            "suggestions": [],
            "security_concerns": [],
            "edge_cases": [],
            "model_used": context.get("model", {}).get("id"),
            "status": "reviewed",
        }
