#!/usr/bin/env python3
"""
Agent Executor - Re-export from agents.executor for backward compatibility.
"""
from agents.executor import *  # noqa: F401,F403
from agents.executor import (
    AgentProvider,
    FileEdit,
    AgentIntent,
    AgentConfig,
    AgentResult,
    STAGE_COMMAND_ALLOWLIST,
    parse_agent_output,
    execute_agent,
)
