#!/usr/bin/env python3
"""
Deployer - Re-export from agents.deployer for backward compatibility.
"""
from agents.deployer import *  # noqa: F401,F403
from agents.deployer import (
    deploy,
    run_health_check,
    create_deployment_record,
)
