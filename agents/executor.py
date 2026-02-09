#!/usr/bin/env python3
"""
Agent Executor - Runs AI agent providers (API and CLI) for development workflow stages.
Supports 6 providers: claude_api, gemini_api, gpt_api, claude_code_cli, gemini_cli, codex_cli.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class AgentProvider(str, Enum):
    CLAUDE_API = "claude_api"
    GEMINI_API = "gemini_api"
    GPT_API = "gpt_api"
    CLAUDE_CODE_CLI = "claude_code_cli"
    GEMINI_CLI = "gemini_cli"
    CODEX_CLI = "codex_cli"


@dataclass
class FileEdit:
    path: str
    content: str
    action: str = "create_or_update"  # create_or_update, delete


@dataclass
class AgentIntent:
    explanation: str = ""
    file_edits: List[FileEdit] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class AgentConfig:
    providers: List[AgentProvider] = field(default_factory=lambda: [AgentProvider.CLAUDE_API])
    mode: str = "first"  # "first" = try in order, "all" = run all in parallel
    timeout: int = 300  # seconds
    max_tokens: int = 8192
    temperature: float = 0.2


@dataclass
class AgentResult:
    provider: AgentProvider
    success: bool
    intent: Optional[AgentIntent] = None
    error: Optional[str] = None
    duration_ms: int = 0
    raw_output: str = ""


# Per-stage command allowlists
STAGE_COMMAND_ALLOWLIST: Dict[str, List[str]] = {
    "setup": ["git", "npm", "pip", "mkdir", "cp", "mv"],
    "plan": [],
    "review": [],
    "implement": ["git", "npm", "pip", "npx", "python", "node"],
    "validate": ["npm", "pytest", "python", "node", "npx"],
    "commit_pr": ["git", "gh"],
    "deploy": ["git", "docker", "heroku", "vercel"],
    "notify": [],
}


def _filter_commands(commands: List[str], stage: str) -> List[str]:
    """Filter commands against the stage allowlist."""
    allowlist = STAGE_COMMAND_ALLOWLIST.get(stage, [])
    if not allowlist:
        return []
    filtered = []
    for cmd in commands:
        base_cmd = cmd.strip().split()[0] if cmd.strip() else ""
        if base_cmd in allowlist:
            filtered.append(cmd)
    return filtered


def parse_agent_output(raw_output: str, stage: str) -> AgentIntent:
    """
    Parse raw agent output into structured intent.
    Extracts file edits (```filename blocks), commands, and explanation.
    """
    intent = AgentIntent(raw_output=raw_output)
    lines = raw_output.split('\n')

    # Extract file edit blocks: ```path/to/file\n...content...\n```
    file_pattern = re.compile(r'^```(\S+\.[\w.]+)\s*$')
    i = 0
    explanation_lines = []
    while i < len(lines):
        match = file_pattern.match(lines[i])
        if match:
            filepath = match.group(1)
            content_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                content_lines.append(lines[i])
                i += 1
            intent.file_edits.append(FileEdit(
                path=filepath,
                content='\n'.join(content_lines)
            ))
            i += 1  # skip closing ```
            continue

        # Extract shell commands: lines starting with $ or > or ```bash blocks
        line = lines[i]
        if line.strip().startswith('$ ') or line.strip().startswith('> '):
            cmd = line.strip()[2:]
            intent.commands.append(cmd)
        elif line.strip() == '```bash' or line.strip() == '```shell' or line.strip() == '```sh':
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                cmd_line = lines[i].strip()
                if cmd_line and not cmd_line.startswith('#'):
                    intent.commands.append(cmd_line)
                i += 1
        else:
            explanation_lines.append(line)
        i += 1

    intent.explanation = '\n'.join(explanation_lines).strip()
    intent.commands = _filter_commands(intent.commands, stage)
    return intent


async def execute_agent(
    config: AgentConfig,
    prompt: str,
    context: str = "",
    workspace_path: str = "",
    stage: str = "implement"
) -> List[AgentResult]:
    """
    Execute AI agents according to config.
    mode="first": try providers in order, return on first success.
    mode="all": run all providers in parallel.
    """
    full_prompt = f"{context}\n\n{prompt}" if context else prompt

    if config.mode == "all":
        tasks = [
            _execute_provider(provider, full_prompt, config, workspace_path, stage)
            for provider in config.providers
        ]
        return await asyncio.gather(*tasks)
    else:
        # "first" mode — try in order
        for provider in config.providers:
            result = await _execute_provider(provider, full_prompt, config, workspace_path, stage)
            if result.success:
                return [result]
        # All failed — return last result
        return [result]


async def _execute_provider(
    provider: AgentProvider,
    prompt: str,
    config: AgentConfig,
    workspace_path: str,
    stage: str
) -> AgentResult:
    """Route to the correct provider executor."""
    start = datetime.utcnow()
    try:
        handlers = {
            AgentProvider.CLAUDE_API: _execute_claude_api,
            AgentProvider.GEMINI_API: _execute_gemini_api,
            AgentProvider.GPT_API: _execute_gpt_api,
            AgentProvider.CLAUDE_CODE_CLI: _execute_claude_code_cli,
            AgentProvider.GEMINI_CLI: _execute_gemini_cli,
            AgentProvider.CODEX_CLI: _execute_codex_cli,
        }
        handler = handlers[provider]
        raw_output = await asyncio.wait_for(
            handler(prompt, config, workspace_path),
            timeout=config.timeout
        )
        intent = parse_agent_output(raw_output, stage)
        elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
        return AgentResult(
            provider=provider,
            success=True,
            intent=intent,
            duration_ms=elapsed,
            raw_output=raw_output
        )
    except asyncio.TimeoutError:
        elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
        return AgentResult(
            provider=provider,
            success=False,
            error=f"Timeout after {config.timeout}s",
            duration_ms=elapsed
        )
    except Exception as e:
        elapsed = int((datetime.utcnow() - start).total_seconds() * 1000)
        return AgentResult(
            provider=provider,
            success=False,
            error=str(e),
            duration_ms=elapsed
        )


# --- API Providers ---

async def _execute_claude_api(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


async def _execute_gemini_api(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    )
    return response.text


async def _execute_gpt_api(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# --- CLI Providers ---

async def _execute_cli(cmd: List[str], prompt: str, workspace_path: str, timeout: int) -> str:
    """Generic CLI executor with stdin pipe."""
    cwd = workspace_path if workspace_path and os.path.isdir(workspace_path) else None
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd
    )
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=prompt.encode()),
        timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CLI exited {proc.returncode}: {stderr.decode()[:500]}")
    return stdout.decode()


async def _execute_claude_code_cli(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    return await _execute_cli(
        ["claude", "--print", "--output-format", "text"],
        prompt, workspace_path, config.timeout
    )


async def _execute_gemini_cli(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    return await _execute_cli(
        ["gemini", "--print"],
        prompt, workspace_path, config.timeout
    )


async def _execute_codex_cli(prompt: str, config: AgentConfig, workspace_path: str) -> str:
    return await _execute_cli(
        ["codex", "--print"],
        prompt, workspace_path, config.timeout
    )
