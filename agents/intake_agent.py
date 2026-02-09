#!/usr/bin/env python3
"""
Intake Agent - Uses Claude Sonnet 4.5 to evaluate client feature requests.
Asks clarifying questions or generates a structured spec summary.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

INTAKE_MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """You are a friendly, sharp product analyst at a software development agency.
A client has submitted a feature request. Your job is to evaluate it and respond.

If the request is UNCLEAR or MISSING critical details:
- Ask 2-4 specific clarifying questions
- Be conversational, not corporate
- Focus on: what exactly they want, who uses it, edge cases, priorities

If the request is CLEAR and has enough detail:
- Generate a brief spec summary (3-5 bullet points)
- Confirm you understand their needs
- Mention what you'll build and rough approach

Respond in JSON format:
{
  "response_type": "clarification" or "confirmation",
  "message": "Your friendly response to the client",
  "questions": ["question1", "question2"],  // only if clarification
  "spec_summary": "Brief summary of what we'll build"  // only if confirmation
}

Always be warm, helpful, and concise. No jargon."""


async def evaluate_request(description: str, project_context: dict = None) -> dict:
    """
    Evaluate a client feature request using Claude Sonnet 4.5.
    Returns: { response_type, message, questions, spec_summary }
    """
    import anthropic
    import json

    client = anthropic.Anthropic()

    context_text = ""
    if project_context:
        context_text = f"\n\nProject context: {project_context.get('name', 'Unknown project')}"
        if project_context.get("description"):
            context_text += f" - {project_context['description']}"

    user_message = f"Client feature request: {description}{context_text}"

    message = client.messages.create(
        model=INTAKE_MODEL,
        max_tokens=1024,
        temperature=0.3,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = message.content[0].text

    # Parse JSON response
    try:
        # Try to extract JSON from the response
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # If not valid JSON, wrap in a confirmation response
        result = {
            "response_type": "confirmation",
            "message": response_text,
            "questions": [],
            "spec_summary": response_text,
        }

    # Ensure required fields
    result.setdefault("response_type", "confirmation")
    result.setdefault("message", "")
    result.setdefault("questions", [])
    result.setdefault("spec_summary", "")

    return result
