"""Metrics Agent - Tracks performance data and ROI."""

from typing import Dict, Any
from content_generation.agents.base import BaseAgent


class MetricsAgent(BaseAgent):
    agent_name = "metrics_agent"
    description = "Track performance data and ROI for each piece of content."

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        published = context.get("stage_outputs", {}).get("publish", {})

        return {
            "performance_data": input_data.get("performance_data", []),
            "roi_metrics": input_data.get("roi_metrics", {}),
            "engagement_rates": input_data.get("engagement_rates", {}),
            "conversion_data": input_data.get("conversion_data", {}),
            "status": "metrics_collected",
        }
