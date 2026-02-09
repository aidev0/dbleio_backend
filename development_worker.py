#!/usr/bin/env python3
"""
Development Worker - Re-export from agents.worker for backward compatibility.
Run as: python development_worker.py  OR  python -m agents.worker
"""
from agents.worker import poll_and_execute

if __name__ == "__main__":
    import asyncio
    asyncio.run(poll_and_execute())
