#!/usr/bin/env python3
"""
Development Setup Agent - Re-export from agents.setup_agent for backward compatibility.
"""
from agents.setup_agent import *  # noqa: F401,F403
from agents.setup_agent import (
    WORKSPACE_BASE,
    workspace_path,
    setup_workspace,
    cleanup_workspace,
)
