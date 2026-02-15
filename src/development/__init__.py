"""
Development Module

Architecture:
- agents/     — One agent per pipeline stage, each configurable with model selection
- workflows/  — Pipeline definitions, state machine, stage transitions
- orchestrator.py — Bridges UI chat + FDM + workflow executor; uses agents as tools
- state.py    — Robust state management for workflow execution
- models.py   — Model registry: LLM, CLI, and video generation model configs
"""
