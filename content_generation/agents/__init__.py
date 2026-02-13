"""
Content Generation Agents

Each agent handles a specific pipeline stage and can be invoked as a tool
by the orchestrator. Agents persist their own state via AgentStateStore.
"""

from content_generation.agents.base import BaseAgent
from content_generation.agents.strategy import StrategyAgent
from content_generation.agents.scheduling import SchedulingAgent
from content_generation.agents.research import ResearchAgent
from content_generation.agents.concepts import ConceptsAgent
from content_generation.agents.content_gen import ContentGenAgent
from content_generation.agents.simulation import SimulationAgent
from content_generation.agents.publish import PublishAgent
from content_generation.agents.metrics import MetricsAgent
from content_generation.agents.analytics import AnalyticsAgent
from content_generation.agents.channel_learning import ChannelLearningAgent
from content_generation.agents.ab_testing import ABTestingAgent
from content_generation.agents.reinforcement import ReinforcementAgent

# Registry: agent_name → class
AGENT_REGISTRY = {
    "strategy_agent": StrategyAgent,
    "scheduling_agent": SchedulingAgent,
    "research_agent": ResearchAgent,
    "concepts_agent": ConceptsAgent,
    "content_gen_agent": ContentGenAgent,
    "simulation_agent": SimulationAgent,
    "publish_agent": PublishAgent,
    "metrics_agent": MetricsAgent,
    "analytics_agent": AnalyticsAgent,
    "channel_learning_agent": ChannelLearningAgent,
    "ab_testing_agent": ABTestingAgent,
    "reinforcement_agent": ReinforcementAgent,
}


def get_agent(agent_name: str, workflow_id: str) -> BaseAgent:
    """Factory: instantiate an agent by name."""
    cls = AGENT_REGISTRY.get(agent_name)
    if not cls:
        raise ValueError(f"Unknown agent: {agent_name}")
    return cls(workflow_id=workflow_id)
