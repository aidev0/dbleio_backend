"""Strategy & Assets Agent - Gathers brand goals and Shopify assets."""

from typing import Dict, Any
from content_generation.agents.base import BaseAgent


class StrategyAgent(BaseAgent):
    agent_name = "strategy_agent"
    description = "Define brand goals, gather Shopify assets, and set content direction."

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        brand = context.get("brand", {})
        assets = input_data.get("assets", [])
        goals = input_data.get("goals", [])

        return {
            "brand_summary": {
                "name": brand.get("name"),
                "url": brand.get("url"),
                "product_name": brand.get("product_name"),
                "industry": brand.get("industry"),
            },
            "assets_collected": assets,
            "goals": goals,
            "strategy_brief": input_data.get("strategy_brief", ""),
            "status": "ready",
        }
