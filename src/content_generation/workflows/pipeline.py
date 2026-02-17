"""
Content Generation Pipeline Definition

15-stage pipeline with stage metadata, types, transitions, and feedback loops.
This is the single source of truth for the pipeline structure.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class StageDefinition:
    key: str
    label: str
    stage_type: str  # "human" | "agent" | "auto"
    description: str
    agent_name: Optional[str] = None  # Which agent handles this stage (for agent/auto types)
    approval_required: bool = False
    reject_target: Optional[str] = None  # Stage key to jump to on rejection
    feedback_target: Optional[str] = None  # Stage key for learning feedback loops


PIPELINE_STAGES: List[StageDefinition] = [
    StageDefinition(
        key="strategy_assets",
        label="Strategy & Assets",
        stage_type="human",
        description="Define brand goals and gather Shopify assets to guide content direction.",
        agent_name="strategy_agent",
    ),
    StageDefinition(
        key="scheduling",
        label="Scheduling",
        stage_type="agent",
        description="Plan the content calendar.",
        agent_name="scheduling_agent",
    ),
    StageDefinition(
        key="research",
        label="Research",
        stage_type="agent",
        description="Discover trends and identify patterns relevant to your audience.",
        agent_name="research_agent",
    ),
    StageDefinition(
        key="concepts",
        label="Concepts",
        stage_type="agent",
        description="Generate ideas and develop scripts for content pieces.",
        agent_name="concepts_agent",
    ),
    StageDefinition(
        key="image_generation",
        label="Image Generation",
        stage_type="agent",
        description="Generate concept art and reference images for each concept before storyboarding.",
        agent_name="image_gen_agent",
    ),
    StageDefinition(
        key="storyboard",
        label="Storyboard",
        stage_type="agent",
        description="Generate detailed storylines with scenes, characters, and visual assets for each concept.",
        agent_name="storyboard_agent",
    ),
    StageDefinition(
        key="video_generation",
        label="Video Generation",
        stage_type="agent",
        description="Produce videos and voiceovers using AI.",
        agent_name="video_gen_agent",
    ),
    StageDefinition(
        key="simulation_testing",
        label="Simulation & Testing",
        stage_type="agent",
        description="Model audience personas and run A/B testing to predict content performance.",
        agent_name="simulation_agent",
    ),
    StageDefinition(
        key="brand_qa",
        label="Brand QA",
        stage_type="human",
        description="Ensure content aligns with brand guidelines and safety requirements.",
        approval_required=True,
        reject_target="concepts",  # QA Reject → Concepts
    ),
    StageDefinition(
        key="fdm_review",
        label="FDM Review",
        stage_type="human",
        description="Team members review, edit, or override AI decisions and run compliance checks.",
        approval_required=True,
        reject_target="concepts",
    ),
    StageDefinition(
        key="publish",
        label="Publish",
        stage_type="auto",
        description="Deploy content across channels (3 reels/week, daily stories).",
        agent_name="publish_agent",
    ),
    StageDefinition(
        key="metrics",
        label="Metrics",
        stage_type="auto",
        description="Track performance data and ROI for each piece of content.",
        agent_name="metrics_agent",
    ),
    StageDefinition(
        key="analytics",
        label="Analytics",
        stage_type="agent",
        description="Generate insights and build predictive models from the data.",
        agent_name="analytics_agent",
        feedback_target="scheduling",  # Learning loop: Analytics → Scheduling
    ),
    StageDefinition(
        key="channel_learning",
        label="Channel-Specific Learning",
        stage_type="agent",
        description="Adapt strategies based on what works on each platform.",
        agent_name="channel_learning_agent",
    ),
    StageDefinition(
        key="ab_testing",
        label="A/B Testing",
        stage_type="agent",
        description="Test content variations to identify top performers and optimize future content.",
        agent_name="ab_testing_agent",
    ),
    StageDefinition(
        key="reinforcement_learning",
        label="Reinforcement Learning",
        stage_type="auto",
        description="Continuously fine-tune the system to improve future outputs.",
        agent_name="reinforcement_agent",
        feedback_target="research",  # RL feedback → Research
    ),
]

STAGE_KEYS = [s.key for s in PIPELINE_STAGES]
STAGE_MAP: Dict[str, StageDefinition] = {s.key: s for s in PIPELINE_STAGES}
STAGE_LABELS: Dict[str, str] = {s.key: s.label for s in PIPELINE_STAGES}


def get_stage_index(key: str) -> int:
    return STAGE_KEYS.index(key)


def get_next_stage(current_key: str) -> Optional[str]:
    idx = get_stage_index(current_key)
    if idx + 1 < len(STAGE_KEYS):
        return STAGE_KEYS[idx + 1]
    return None


def get_reject_target(current_key: str) -> Optional[str]:
    stage = STAGE_MAP.get(current_key)
    return stage.reject_target if stage else None
