"""Scheduling Agent - Plans the content calendar."""

from typing import Dict, Any
from content_generation.agents.base import BaseAgent


class SchedulingAgent(BaseAgent):
    agent_name = "scheduling_agent"
    description = "Plan the content calendar based on strategy and audience data."

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        strategy = context.get("stage_outputs", {}).get("strategy_assets", {})
        analytics_feedback = context.get("stage_outputs", {}).get("analytics", {})

        return {
            "calendar": input_data.get("calendar", []),
            "posting_frequency": input_data.get("posting_frequency", {
                "reels_per_week": 3,
                "stories_per_day": 1,
            }),
            "channels": input_data.get("channels", []),
            "analytics_adjustments": analytics_feedback.get("schedule_recommendations", []),
            "status": "planned",
        }
