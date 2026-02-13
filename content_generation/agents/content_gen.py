"""Content Generation Agent - Produces videos, images, and voiceovers using AI."""

from typing import Dict, Any
from content_generation.agents.base import BaseAgent


class ContentGenAgent(BaseAgent):
    agent_name = "content_gen_agent"
    description = "Produce videos, images, and voiceovers using AI."

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        concepts = context.get("stage_outputs", {}).get("concepts", {})

        return {
            "generated_assets": input_data.get("generated_assets", []),
            "videos": input_data.get("videos", []),
            "images": input_data.get("images", []),
            "voiceovers": input_data.get("voiceovers", []),
            "captions": input_data.get("captions", []),
            "status": "content_generated",
        }
