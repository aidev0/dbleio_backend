"""
Base Agent class for development pipeline agents.

Extends the content_generation pattern with model selection:
- Each agent can be configured with a specific LLM/CLI/Video model
- Model defaults come from pipeline.py, overridable per-workflow via config
- State is persisted via DevAgentStateStore
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.development.state import DevAgentStateStore
from src.development.models import get_model, ModelConfig


class DevBaseAgent(ABC):
    """Base class for all development pipeline agents."""

    agent_name: str = "dev_base_agent"
    description: str = "Base development agent"
    allowed_model_categories: List[str] = ["llm"]
    default_model_id: Optional[str] = None

    def __init__(self, workflow_id: str, model_id: Optional[str] = None):
        self.workflow_id = workflow_id
        # Resolve model: explicit override > default
        self._model_id = model_id or self.default_model_id
        self._model: Optional[ModelConfig] = get_model(self._model_id) if self._model_id else None

    @property
    def model(self) -> Optional[ModelConfig]:
        return self._model

    @property
    def model_string(self) -> Optional[str]:
        return self._model.model_string if self._model else None

    # --- State persistence ---

    def load_state(self) -> dict:
        return DevAgentStateStore.get_state_data(self.workflow_id, self.agent_name)

    def save_state(self, state: dict):
        DevAgentStateStore.save(self.workflow_id, self.agent_name, state)

    def clear_state(self):
        DevAgentStateStore.clear(self.workflow_id, self.agent_name)

    # --- Execution ---

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            input_data: Direct input for this stage (prior stage output)
            context: Full workflow context (project info, all prior outputs, model config, etc.)

        Returns:
            output_data dict to be stored in the workflow node
        """
        pass

    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with state management. Wraps execute() with state load/save."""
        prior_state = self.load_state()
        context["agent_prior_state"] = prior_state
        context["model"] = {
            "id": self._model_id,
            "string": self.model_string,
            "provider": self._model.provider if self._model else None,
            "category": self._model.category if self._model else None,
        }

        result = await self.execute(input_data, context)

        self.save_state({
            "last_input": input_data,
            "last_output": result,
            "model_used": self._model_id,
            "execution_count": prior_state.get("execution_count", 0) + 1,
        })

        return result

    def as_tool_definition(self) -> dict:
        """Return a tool-use definition the orchestrator can present to the LLM."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "model_id": self._model_id,
            "allowed_model_categories": self.allowed_model_categories,
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
