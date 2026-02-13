"""
Content Generation Module

Architecture:
- agents/     — One agent per pipeline stage (strategy, research, content, etc.)
- workflows/  — Pipeline definitions, state machine, stage transitions
- orchestrator.py — Bridges UI chat + FDM + workflow executor; uses agents as tools
- state.py    — Robust state management for workflow execution
"""
