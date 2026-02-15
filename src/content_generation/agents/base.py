"""
Base Agent class for all content generation pipeline agents.

Each agent:
- Has a name matching its pipeline stage
- Can load/save its own state via AgentStateStore
- Receives input_data (prior stage outputs + workflow context)
- Returns output_data (result to store in the node)
- Can be used as a "tool" by the orchestrator
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.content_generation.state import AgentStateStore


class BaseAgent(ABC):
    """Base class for all pipeline stage agents."""

    agent_name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id

    # --- State persistence ---

    def load_state(self) -> dict:
        """Load this agent's persisted state."""
        return AgentStateStore.get_state_data(self.workflow_id, self.agent_name)

    def save_state(self, state: dict):
        """Persist this agent's state."""
        AgentStateStore.save(self.workflow_id, self.agent_name, state)

    def clear_state(self):
        """Clear this agent's state (for restart/retry)."""
        AgentStateStore.clear(self.workflow_id, self.agent_name)

    # --- Execution ---

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            input_data: Direct input for this stage (e.g., from prior stage output)
            context: Full workflow context (brand info, all prior outputs, user session, etc.)

        Returns:
            output_data dict to be stored in the workflow node
        """
        pass

    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with state management.
        Wraps execute() with state load/save.
        """
        # Load any prior state
        prior_state = self.load_state()
        context["agent_prior_state"] = prior_state

        # Execute
        result = await self.execute(input_data, context)

        # Save updated state
        self.save_state({
            "last_input": input_data,
            "last_output": result,
            "execution_count": prior_state.get("execution_count", 0) + 1,
        })

        return result

    def as_tool_definition(self) -> dict:
        """Return a tool-use definition the orchestrator can present to the LLM."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input_data": {
                        "type": "object",
                        "description": "Input data for this agent stage",
                    }
                },
            },
        }
