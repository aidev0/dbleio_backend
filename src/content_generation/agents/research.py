"""Research Agent - Discovers trends and patterns relevant to the audience."""

from typing import Dict, Any
from src.content_generation.agents.base import BaseAgent
from src.research_helpers import (
    get_top_performers,
    build_trend_data,
    extract_brand_from_url,
    fetch_financial_data,
)


class ResearchAgent(BaseAgent):
    agent_name = "research_agent"
    description = "Discover trends and identify patterns relevant to your audience."

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        brand = context.get("brand", {})
        brand_username = brand.get("instagram_username")
        brand_url = brand.get("website_url")
        competitor_usernames = input_data.get("competitor_usernames", [])

        result: Dict[str, Any] = {
            "status": "researched",
        }

        # Brand URL analysis
        if brand_url:
            result["brand_url_analysis"] = extract_brand_from_url(brand_url)

        # Brand Instagram top performers + trends
        if brand_username:
            result["brand_instagram"] = get_top_performers(brand_username, percentile=0.2)
            result["trends"] = {brand_username: build_trend_data(brand_username)}

        # Competitor analysis
        competitor_data = {}
        for comp in competitor_usernames:
            comp_ig = get_top_performers(comp, percentile=0.2)
            competitor_data[comp] = comp_ig
            if "trends" not in result:
                result["trends"] = {}
            result["trends"][comp] = build_trend_data(comp)

        if competitor_data:
            result["competitor_instagram"] = competitor_data

        # Financial data
        financial_companies = input_data.get("financial_companies", [])
        if financial_companies:
            financial = {}
            for co in financial_companies:
                key = co.lower().replace(" ", "_")
                financial[key] = fetch_financial_data(co)
            result["financial"] = financial

        return result
