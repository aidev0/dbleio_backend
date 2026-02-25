"""
Content Generation Pipeline Definition

15-stage pipeline with stage metadata, types, transitions, and feedback loops.
This is the single source of truth for the pipeline structure.

Categories: Input, Content Generation, Simulation, Review & Publish, Analysis
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class StageDefinition:
    key: str
    label: str
    stage_type: str  # "human" | "agent" | "both"
    description: str
    category: str  # "Input" | "Content Generation" | "Simulation" | "Review & Publish" | "Analysis"
    agent_name: Optional[str] = None  # Which agent handles this stage (for agent/both types)
    approval_required: bool = False
    reject_target: Optional[str] = None  # Stage key to jump to on rejection
    feedback_target: Optional[str] = None  # Stage key for learning feedback loops


PIPELINE_STAGES: List[StageDefinition] = [
    # --- Input ---
    StageDefinition(
        key="brand",
        label="Brand",
        stage_type="human",
        description="Review brand details, URLs, and social profiles.",
        category="Input",
        agent_name="strategy_agent",
    ),
    StageDefinition(
        key="campaign_strategy",
        label="Campaign & Strategy & Assets",
        stage_type="human",
        description="Select campaign, strategy, and manage brand assets.",
        category="Input",
        agent_name="strategy_agent",
    ),
    StageDefinition(
        key="research",
        label="Research",
        stage_type="agent",
        description="Discover trends and identify patterns relevant to your audience.",
        category="Input",
        agent_name="research_agent",
    ),
    StageDefinition(
        key="scheduling",
        label="Scheduling",
        stage_type="human",
        description="Plan the content calendar.",
        category="Input",
        agent_name="scheduling_agent",
    ),
    # --- Content Generation ---
    StageDefinition(
        key="concepts",
        label="Concepts",
        stage_type="both",
        description="Generate ideas and develop scripts for content pieces.",
        category="Content Generation",
        agent_name="concepts_agent",
    ),
    StageDefinition(
        key="image_generation",
        label="Image Generation",
        stage_type="agent",
        description="Generate concept art and reference images for each concept before storyboarding.",
        category="Content Generation",
        agent_name="image_gen_agent",
    ),
    StageDefinition(
        key="storyboard",
        label="Storyboard",
        stage_type="both",
        description="Generate detailed storylines with scenes, characters, and visual assets for each concept.",
        category="Content Generation",
        agent_name="storyboard_agent",
    ),
    StageDefinition(
        key="video_generation",
        label="Video Generation",
        stage_type="agent",
        description="Produce videos and voiceovers using AI.",
        category="Content Generation",
        agent_name="video_gen_agent",
    ),
    # --- Simulation ---
    StageDefinition(
        key="simulation_testing",
        label="Simulation & Testing",
        stage_type="both",
        description="Model audience personas and run A/B testing to predict content performance.",
        category="Simulation",
        agent_name="simulation_agent",
    ),
    StageDefinition(
        key="predictive_modeling",
        label="Predictive Modeling",
        stage_type="agent",
        description="Build predictive models to forecast content performance across channels.",
        category="Simulation",
        agent_name="predictive_modeling_agent",
    ),
    StageDefinition(
        key="content_ranking",
        label="Content Ranking",
        stage_type="agent",
        description="Rank and prioritize content variants based on predicted performance.",
        category="Simulation",
        agent_name="content_ranking_agent",
    ),
    # --- Review & Publish ---
    StageDefinition(
        key="fdm_review",
        label="FDM Review",
        stage_type="human",
        description="Team members review, edit, or override AI decisions and run compliance checks.",
        category="Review & Publish",
        approval_required=True,
        reject_target="concepts",
    ),
    StageDefinition(
        key="brand_qa",
        label="Brand QA",
        stage_type="human",
        description="Ensure content aligns with brand guidelines and safety requirements.",
        category="Review & Publish",
        approval_required=True,
        reject_target="concepts",
    ),
    StageDefinition(
        key="publish",
        label="Publish",
        stage_type="agent",
        description="Deploy content across channels (3 reels/week, daily stories).",
        category="Review & Publish",
        agent_name="publish_agent",
    ),
    # --- Analysis ---
    StageDefinition(
        key="analytics",
        label="A/B Testing & Analytics",
        stage_type="agent",
        description="Run A/B tests, generate insights, and build predictive models from performance data.",
        category="Analysis",
        agent_name="analytics_agent",
        feedback_target="research",  # Learning loop: Analytics -> Research
    ),
]

STAGE_KEYS = [s.key for s in PIPELINE_STAGES]
STAGE_MAP: Dict[str, StageDefinition] = {s.key: s for s in PIPELINE_STAGES}
STAGE_LABELS: Dict[str, str] = {s.key: s.label for s in PIPELINE_STAGES}

# Category groupings: {category_name: [stage_keys]}
STAGE_CATEGORIES: Dict[str, List[str]] = {}
for _s in PIPELINE_STAGES:
    STAGE_CATEGORIES.setdefault(_s.category, []).append(_s.key)


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
