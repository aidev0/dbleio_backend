"""
Content Generation Agents

Each agent handles a specific pipeline stage and can be invoked as a tool
by the orchestrator. Agents persist their own state via AgentStateStore.
"""

from src.content_generation.agents.base import BaseAgent
from src.content_generation.agents.strategy import StrategyAgent
from src.content_generation.agents.scheduling import SchedulingAgent
from src.content_generation.agents.research import ResearchAgent
from src.content_generation.agents.concepts import ConceptsAgent
from src.content_generation.agents.content_gen import ContentGenAgent
from src.content_generation.agents.simulation import SimulationAgent
from src.content_generation.agents.publish import PublishAgent
from src.content_generation.agents.metrics import MetricsAgent
from src.content_generation.agents.analytics import AnalyticsAgent
from src.content_generation.agents.channel_learning import ChannelLearningAgent
from src.content_generation.agents.ab_testing import ABTestingAgent
from src.content_generation.agents.reinforcement import ReinforcementAgent

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
