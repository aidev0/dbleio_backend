#!/usr/bin/env python3
"""
Chat API endpoint for conversational AI interactions
Supports multiple modes: persona_specific, evaluation_specific, results_analysis, persona_chat
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal, List
import os
from dotenv import load_dotenv
import openai
import anthropic
import google.generativeai as genai

load_dotenv()

# Configure AI providers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    context: Dict[str, Any]
    provider: Literal['openai', 'anthropic', 'google']
    mode: Literal['persona_specific', 'evaluation_specific', 'results_analysis', 'persona_chat']


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str


def build_system_prompt(mode: str, context: Dict[str, Any]) -> str:
    """Build system prompt based on mode and context"""

    if mode == 'persona_specific':
        persona = context.get('persona', {})
        evaluations = context.get('evaluations', [])
        videos = context.get('videos', [])

        return f"""You are an AI assistant helping analyze marketing video performance for a specific customer persona.

PERSONA: {persona.get('name', 'Unknown')}
Demographics: {persona.get('demographics', {})}

EVALUATIONS: You have {len(evaluations)} evaluations from this persona across different AI models.
VIDEOS: You have {len(videos)} videos to reference.

Your role is to:
- Answer questions about this persona's preferences and behavior
- Explain why this persona prefers certain videos
- Provide insights into this persona's decision-making
- Reference specific evaluations and video details when relevant

Be specific, data-driven, and helpful."""

    elif mode == 'evaluation_specific':
        persona = context.get('persona', {})
        evaluation = context.get('evaluation', {})
        videos = context.get('videos', [])

        return f"""You are an AI assistant helping analyze a specific evaluation result.

PERSONA: {persona.get('name', 'Unknown')}
EVALUATION DETAILS: {evaluation}
VIDEOS: {videos}

Your role is to:
- Explain the reasoning behind this specific evaluation
- Answer questions about why certain videos were preferred
- Provide insights into the evaluation methodology
- Help understand the confidence scores and rankings

Be specific and reference the evaluation data."""

    elif mode == 'results_analysis':
        summary = context.get('summary', {})
        evaluations = context.get('evaluations', [])
        personas = context.get('personas', [])
        videos = context.get('videos', [])

        return f"""You are an AI assistant helping analyze overall campaign results.

CAMPAIGN SUMMARY:
- Total evaluations: {summary.get('total_evaluations', 0)}
- Video performance: {summary.get('video_vote_counts', {})}
- Ranking distribution: {summary.get('ranking_distribution', {})}

PERSONAS: {len(personas)} personas
VIDEOS: {len(videos)} videos
EVALUATIONS: {len(evaluations)} total evaluations

Your role is to:
- Provide high-level insights about campaign performance
- Identify trends across personas and videos
- Recommend optimization strategies
- Answer questions about overall results

Be analytical, strategic, and actionable."""

    elif mode == 'persona_chat':
        mentioned_personas = context.get('mentioned_personas', [])
        mentioned_videos = context.get('mentioned_videos', [])

        return f"""You are an AI assistant helping with video marketing campaign analysis.

MENTIONED PERSONAS: {len(mentioned_personas)} persona(s) referenced
MENTIONED VIDEOS: {len(mentioned_videos)} video(s) referenced

Context data available:
{mentioned_personas}
{mentioned_videos}

Your role is to:
- Answer questions about mentioned personas and videos
- Provide insights based on evaluation data
- Help understand preferences and performance
- Reference specific data points when relevant

Be helpful, specific, and data-driven."""

    return "You are a helpful AI assistant for video marketing analysis."


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat requests with different modes and AI providers
    """
    try:
        # Build system prompt
        system_prompt = build_system_prompt(request.mode, request.context)

        # Get conversation history
        conversation_history = request.context.get('conversation_history', [])

        # Call appropriate AI provider
        if request.provider == 'openai':
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": request.message})

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            return ChatResponse(response=response.choices[0].message.content)

        elif request.provider == 'anthropic':
            if not ANTHROPIC_API_KEY:
                raise HTTPException(status_code=500, detail="Anthropic API key not configured")

            # Convert conversation history to Claude format
            claude_messages = []
            for msg in conversation_history:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            claude_messages.append({
                "role": "user",
                "content": request.message
            })

            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=claude_messages
            )

            return ChatResponse(response=response.content[0].text)

        elif request.provider == 'google':
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API key not configured")

            model = genai.GenerativeModel('gemini-2.5-pro')

            # Build conversation
            full_prompt = f"{system_prompt}\n\n"
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n"
            full_prompt += f"User: {request.message}\nAssistant:"

            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2000
                )
            )

            return ChatResponse(response=response.text)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
